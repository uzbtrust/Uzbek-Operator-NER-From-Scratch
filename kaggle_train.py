import os
import sys
import time
import json
import yaml
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast

from data.download_datasets import download_conll, download_wikiann
from data.vocab import Vocabulary, CharVocabulary, TagMap, build_vocabs
from data.preprocess import SavedNERDataset, NERDataset, collate_batch, load_raw_data
from embeddings.load_fasttext import download_vectors, load_vectors, build_embedding_matrix
from model.ner_model import BiLSTMCRF
from training.evaluate import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def setup_data():
    raw_dir = "data/raw"
    proc_dir = "data/processed"

    log.info("=== Step 1: Downloading datasets ===")
    download_conll(raw_dir)
    download_wikiann("en", raw_dir)
    download_wikiann("ru", raw_dir)

    log.info("=== Step 2: Building vocabularies ===")
    data_dirs = [
        Path(raw_dir) / "conll2003",
        Path(raw_dir) / "wikiann_en",
        Path(raw_dir) / "wikiann_ru",
    ]
    word_vocab, char_vocab, tag_map = build_vocabs(data_dirs, proc_dir, min_freq=2)
    return word_vocab, char_vocab, tag_map


def load_embeddings(word_vocab, vectors_dir="embeddings/vectors", max_vectors=500000):
    log.info("=== Step 3: Loading FastText embeddings ===")

    en_path = Path(vectors_dir) / "cc.en.300.vec"
    ru_path = Path(vectors_dir) / "cc.ru.300.vec"

    if not en_path.exists():
        download_vectors("en", vectors_dir)
    if not ru_path.exists():
        download_vectors("ru", vectors_dir)

    en_vecs = load_vectors(en_path, max_vectors)
    ru_vecs = load_vectors(ru_path, max_vectors)

    merged = {}
    merged.update(en_vecs)
    merged.update(ru_vecs)

    return build_embedding_matrix(word_vocab.word2idx, merged)


def make_dataset(raw_dir, name, split, word_vocab, char_vocab, tag_map, max_seq=128, max_word=30):
    path = Path(raw_dir) / name / f"{split}.json"
    samples = load_raw_data(path)
    return NERDataset(samples, word_vocab, char_vocab, tag_map, max_seq, max_word)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_f1, path):
    torch.save({
        "epoch": epoch,
        "best_f1": best_f1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
    }, path)


def train_loop(model, train_loader, val_loader, tag_map, cfg, device, ckpt_dir, stage_name):
    tcfg = cfg["training"]
    use_fp16 = tcfg["fp16"] and device.type == "cuda"

    lr = tcfg.get("current_lr", tcfg["lr"])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=tcfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=tcfg["scheduler_patience"])
    scaler = GradScaler() if use_fp16 else None

    best_f1 = 0.0
    patience_counter = 0
    max_epochs = tcfg.get("current_epochs", tcfg["max_epochs"])

    log.info(f"--- Training stage: {stage_name} (lr={lr}, epochs={max_epochs}) ---")

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            w = batch["word_ids"].to(device)
            c = batch["char_ids"].to(device)
            t_ids = batch["tag_ids"].to(device)
            l = batch["lang_ids"].to(device)
            m = batch["mask"].to(device)
            lens = batch["lengths"].to(device)

            optimizer.zero_grad()

            if use_fp16 and scaler:
                with autocast():
                    loss = model(w, c, l, t_ids, m, lens)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(w, c, l, t_ids, m, lens)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        val_metrics = run_evaluation(model, val_loader, tag_map, device)
        val_f1 = val_metrics["overall"]["f1"]
        scheduler.step(val_f1)

        log.info(f"[{stage_name}] Epoch {epoch+1}/{max_epochs} | loss: {avg_loss:.4f} | val_f1: {val_f1:.4f} | {elapsed:.1f}s")

        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_f1,
                       Path(ckpt_dir) / f"{stage_name}_epoch_{epoch+1}.pt")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_f1,
                          Path(ckpt_dir) / f"{stage_name}_best.pt")
            log.info(f"New best: {best_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= tcfg["early_stop_patience"]:
            log.info(f"Early stopping at epoch {epoch+1}")
            break

    return best_f1


def main():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU count: {torch.cuda.device_count()}")

    word_vocab, char_vocab, tag_map = setup_data()
    pretrained_emb = load_embeddings(word_vocab)

    model = BiLSTMCRF(
        vocab_size=len(word_vocab),
        num_chars=len(char_vocab),
        num_tags=len(tag_map),
        word_dim=cfg["embeddings"]["word_dim"],
        char_dim=cfg["embeddings"]["char_dim"],
        char_filters=cfg["embeddings"]["char_filters"],
        char_kernel=cfg["embeddings"]["char_kernel_size"],
        num_langs=cfg["model"]["num_langs"],
        lang_dim=cfg["embeddings"]["lang_dim"],
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        pretrained_weights=pretrained_emb,
    )
    model.to(device)

    total_p = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {total_p:,}")

    raw_dir = "data/raw"
    bs = cfg["training"]["batch_size"]
    ckpt_dir = "checkpoints"
    Path(ckpt_dir).mkdir(exist_ok=True)

    log.info("=== Stage 1: Train on CoNLL-2003 (EN) ===")
    en_train = make_dataset(raw_dir, "conll2003", "train", word_vocab, char_vocab, tag_map)
    en_val = make_dataset(raw_dir, "conll2003", "validation", word_vocab, char_vocab, tag_map)

    en_train_loader = DataLoader(en_train, batch_size=bs, shuffle=True, collate_fn=collate_batch, num_workers=2, pin_memory=True)
    en_val_loader = DataLoader(en_val, batch_size=bs, shuffle=False, collate_fn=collate_batch, num_workers=2, pin_memory=True)

    cfg["training"]["current_lr"] = cfg["training"]["lr"]
    cfg["training"]["current_epochs"] = cfg["training"]["max_epochs"]
    en_f1 = train_loop(model, en_train_loader, en_val_loader, tag_map, cfg, device, ckpt_dir, "en_conll")

    log.info("=== Stage 2: Fine-tune on WikiANN RU ===")
    ru_train = make_dataset(raw_dir, "wikiann_ru", "train", word_vocab, char_vocab, tag_map)
    ru_val = make_dataset(raw_dir, "wikiann_ru", "validation", word_vocab, char_vocab, tag_map)

    ru_train_loader = DataLoader(ru_train, batch_size=bs, shuffle=True, collate_fn=collate_batch, num_workers=2, pin_memory=True)
    ru_val_loader = DataLoader(ru_val, batch_size=bs, shuffle=False, collate_fn=collate_batch, num_workers=2, pin_memory=True)

    cfg["training"]["current_lr"] = cfg["training"]["finetune_lr"]
    cfg["training"]["current_epochs"] = 30
    ru_f1 = train_loop(model, ru_train_loader, ru_val_loader, tag_map, cfg, device, ckpt_dir, "ru_wikiann")

    log.info("=== Stage 3: Merged fine-tune on EN+RU ===")
    wikiann_en_train = make_dataset(raw_dir, "wikiann_en", "train", word_vocab, char_vocab, tag_map)
    wikiann_en_val = make_dataset(raw_dir, "wikiann_en", "validation", word_vocab, char_vocab, tag_map)

    merged_train = ConcatDataset([en_train, ru_train, wikiann_en_train])
    merged_val = ConcatDataset([en_val, ru_val, wikiann_en_val])

    merged_train_loader = DataLoader(merged_train, batch_size=bs, shuffle=True, collate_fn=collate_batch, num_workers=2, pin_memory=True)
    merged_val_loader = DataLoader(merged_val, batch_size=bs, shuffle=False, collate_fn=collate_batch, num_workers=2, pin_memory=True)

    cfg["training"]["current_lr"] = cfg["training"]["finetune_lr"]
    cfg["training"]["current_epochs"] = 20
    merged_f1 = train_loop(model, merged_train_loader, merged_val_loader, tag_map, cfg, device, ckpt_dir, "merged")

    log.info("=== Final Evaluation ===")
    en_test = make_dataset(raw_dir, "conll2003", "test", word_vocab, char_vocab, tag_map)
    en_test_loader = DataLoader(en_test, batch_size=64, shuffle=False, collate_fn=collate_batch)

    ru_test = make_dataset(raw_dir, "wikiann_ru", "test", word_vocab, char_vocab, tag_map)
    ru_test_loader = DataLoader(ru_test, batch_size=64, shuffle=False, collate_fn=collate_batch)

    en_metrics = run_evaluation(model, en_test_loader, tag_map, device)
    ru_metrics = run_evaluation(model, ru_test_loader, tag_map, device)

    log.info(f"EN Test F1: {en_metrics['overall']['f1']:.4f}")
    log.info(f"RU Test F1: {ru_metrics['overall']['f1']:.4f}")

    results = {
        "en_conll_best_f1": en_f1,
        "ru_wikiann_best_f1": ru_f1,
        "merged_best_f1": merged_f1,
        "en_test": en_metrics,
        "ru_test": ru_metrics,
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info("All done. Results saved to results/training_results.json")


if __name__ == "__main__":
    main()
