import os
import sys
import time
import json
import yaml
import logging
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import NERDataset, collate_batch, load_raw_data
from data.vocab import Vocabulary, CharVocabulary, TagMap
from model.ner_model import BiLSTMCRF
from training.evaluate import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def freeze_layers(model, freeze_embeddings=True, freeze_encoder=False):
    if freeze_embeddings:
        for param in model.embedding.word_embedding.parameters():
            param.requires_grad = False
        log.info("Froze word embeddings")

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        log.info("Froze BiLSTM encoder")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable: {trainable:,} / {total:,} parameters")


def train_one_epoch(model, loader, optimizer, scaler, device, grad_clip, use_fp16):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        word_ids = batch["word_ids"].to(device)
        char_ids = batch["char_ids"].to(device)
        tag_ids = batch["tag_ids"].to(device)
        lang_ids = batch["lang_ids"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"].to(device)

        optimizer.zero_grad()

        if use_fp16 and scaler is not None:
            with autocast():
                loss = model(word_ids, char_ids, lang_ids, tag_ids, mask, lengths)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(word_ids, char_ids, lang_ids, tag_ids, mask, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def finetune(cfg, model, train_data, val_data, tag_map, device, ckpt_dir,
             lr=5e-5, max_epochs=20, patience=5):
    use_fp16 = cfg["training"]["fp16"] and device.type == "cuda"
    grad_clip = cfg["training"]["grad_clip"]

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)
    scaler = GradScaler() if use_fp16 else None

    train_loader = DataLoader(
        train_data, batch_size=cfg["training"]["batch_size"],
        shuffle=True, collate_fn=collate_batch, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=cfg["training"]["batch_size"],
        shuffle=False, collate_fn=collate_batch, num_workers=2, pin_memory=True,
    )

    best_f1 = 0.0
    patience_counter = 0
    history = []

    for epoch in range(max_epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, grad_clip, use_fp16)
        elapsed = time.time() - t0

        val_metrics = run_evaluation(model, val_loader, tag_map, device)
        val_f1 = val_metrics["overall"]["f1"]

        scheduler.step(val_f1)
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch + 1,
            "loss": float(avg_loss),
            "val_f1": float(val_f1),
            "lr": float(lr_now),
        })

        log.info(
            f"Epoch {epoch+1}/{max_epochs} | "
            f"loss: {avg_loss:.4f} | val_f1: {val_f1:.4f} | "
            f"lr: {lr_now:.2e} | {elapsed:.1f}s"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_path = Path(ckpt_dir) / "domain_finetuned_best.pt"
            torch.save({
                "epoch": epoch,
                "best_f1": best_f1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, best_path)
            log.info(f"New best F1: {best_f1:.4f} -> saved {best_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            log.info(f"Early stopping at epoch {epoch+1}")
            break

    log.info(f"Domain fine-tuning done. Best F1: {best_f1:.4f}")
    return best_f1, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--domain_data", required=True)
    parser.add_argument("--vocab_dir", default="data/processed")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    word_vocab = Vocabulary.load(Path(args.vocab_dir) / "word_vocab.json")
    char_vocab = CharVocabulary.load(Path(args.vocab_dir) / "char_vocab.json")
    tag_map = TagMap.load(Path(args.vocab_dir) / "tag_map.json")

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
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    log.info(f"Loaded base model from {args.checkpoint}")

    if args.freeze_embeddings or args.freeze_encoder:
        freeze_layers(model, args.freeze_embeddings, args.freeze_encoder)

    domain_dir = Path(args.domain_data)
    max_seq = cfg["data"]["max_seq_len"]
    max_word = cfg["data"]["max_word_len"]

    train_samples = load_raw_data(domain_dir / "train.json")
    val_samples = load_raw_data(domain_dir / "validation.json")

    train_data = NERDataset(train_samples, word_vocab, char_vocab, tag_map, max_seq, max_word)
    val_data = NERDataset(val_samples, word_vocab, char_vocab, tag_map, max_seq, max_word)

    log.info(f"Domain data: train={len(train_data)}, val={len(val_data)}")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    best_f1, history = finetune(
        cfg, model, train_data, val_data, tag_map, device,
        args.ckpt_dir, args.lr, args.epochs, args.patience,
    )

    results = {
        "domain_finetune": {
            "best_f1": float(best_f1),
            "base_checkpoint": args.checkpoint,
            "lr": args.lr,
            "freeze_embeddings": args.freeze_embeddings,
            "freeze_encoder": args.freeze_encoder,
            "history": history,
        }
    }

    results_path = Path(args.ckpt_dir) / "domain_finetune_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
