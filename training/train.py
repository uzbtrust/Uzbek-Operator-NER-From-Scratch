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

from data.preprocess import SavedNERDataset, collate_batch
from data.vocab import Vocabulary, CharVocabulary, TagMap
from embeddings.load_fasttext import merge_embeddings
from model.ner_model import BiLSTMCRF
from training.evaluate import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg, word_vocab, char_vocab, tag_map, pretrained_emb=None):
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
    return model


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_f1, path):
    torch.save({
        "epoch": epoch,
        "best_f1": best_f1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
    }, path)
    log.info(f"Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    log.info(f"Resumed from epoch {ckpt['epoch']} (best F1: {ckpt['best_f1']:.4f})")
    return ckpt["epoch"], ckpt["best_f1"]


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


def train(cfg, train_data, val_data, tag_map, model, device, ckpt_dir, resume_path=None):
    tcfg = cfg["training"]
    use_fp16 = tcfg["fp16"] and device.type == "cuda"

    optimizer = optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=tcfg["scheduler_patience"])
    scaler = GradScaler() if use_fp16 else None

    start_epoch = 0
    best_f1 = 0.0

    if resume_path and Path(resume_path).exists():
        start_epoch, best_f1 = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, device)
        start_epoch += 1

    train_loader = DataLoader(train_data, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=collate_batch, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=tcfg["batch_size"], shuffle=False, collate_fn=collate_batch, num_workers=2, pin_memory=True)

    patience_counter = 0

    for epoch in range(start_epoch, tcfg["max_epochs"]):
        t0 = time.time()
        avg_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, tcfg["grad_clip"], use_fp16)
        elapsed = time.time() - t0

        val_metrics = run_evaluation(model, val_loader, tag_map, device)
        val_f1 = val_metrics["overall"]["f1"]

        scheduler.step(val_f1)
        lr_now = optimizer.param_groups[0]["lr"]

        log.info(f"Epoch {epoch+1}/{tcfg['max_epochs']} | loss: {avg_loss:.4f} | val_f1: {val_f1:.4f} | lr: {lr_now:.6f} | {elapsed:.1f}s")

        ckpt_path = Path(ckpt_dir) / f"epoch_{epoch+1}.pt"
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_f1, ckpt_path)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_path = Path(ckpt_dir) / "best_model.pt"
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_f1, best_path)
            log.info(f"New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= tcfg["early_stop_patience"]:
            log.info(f"Early stopping at epoch {epoch+1}")
            break

    log.info(f"Training done. Best F1: {best_f1:.4f}")
    return best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--vocab_dir", default="data/processed")
    parser.add_argument("--vectors_dir", default="embeddings/vectors")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.epochs:
        cfg["training"]["max_epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    word_vocab = Vocabulary.load(Path(args.vocab_dir) / "word_vocab.json")
    char_vocab = CharVocabulary.load(Path(args.vocab_dir) / "char_vocab.json")
    tag_map = TagMap.load(Path(args.vocab_dir) / "tag_map.json")

    pretrained_emb = None
    if not args.no_pretrained:
        try:
            pretrained_emb = merge_embeddings(word_vocab.word2idx, args.vectors_dir)
            log.info("Loaded pretrained embeddings")
        except Exception as e:
            log.warning(f"Could not load pretrained embeddings: {e}")

    model = build_model(cfg, word_vocab, char_vocab, tag_map, pretrained_emb)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model params: {total_params:,} total, {trainable:,} trainable")

    train_data = SavedNERDataset(args.train_data)
    val_data = SavedNERDataset(args.val_data)
    log.info(f"Train: {len(train_data)} | Val: {len(val_data)}")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    best_f1 = train(cfg, train_data, val_data, tag_map, model, device, args.ckpt_dir, args.resume)

    results = {"best_f1": best_f1, "config": cfg}
    results_path = Path(args.ckpt_dir) / "train_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
