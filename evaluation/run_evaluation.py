import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import SavedNERDataset, collate_batch
from data.vocab import Vocabulary, CharVocabulary, TagMap
from model.ner_model import BiLSTMCRF

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_model(cfg, vocab_dir, checkpoint_path, device):
    word_vocab = Vocabulary.load(Path(vocab_dir) / "word_vocab.json")
    char_vocab = CharVocabulary.load(Path(vocab_dir) / "char_vocab.json")
    tag_map = TagMap.load(Path(vocab_dir) / "tag_map.json")

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
        dropout=0.0,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return model, tag_map


def predict_dataset(model, loader, tag_map, device):
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for batch in loader:
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"].to(device)
            tag_ids = batch["tag_ids"]

            pred_ids = model.predict(word_ids, char_ids, lang_ids, mask, lengths)

            for i in range(len(lengths)):
                ln = lengths[i].item()
                gold_seq = [tag_map.decode(tag_ids[i][t].item()) for t in range(ln)]
                pred_seq = [tag_map.decode(pred_ids[i][t].item()) for t in range(ln)]
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    return all_golds, all_preds


def compute_metrics(golds, preds):
    f1 = float(f1_score(golds, preds))
    prec = float(precision_score(golds, preds))
    rec = float(recall_score(golds, preds))

    report_dict = classification_report(golds, preds, output_dict=True)
    report_text = classification_report(golds, preds)

    per_entity = {}
    for key, vals in report_dict.items():
        if key in ("micro avg", "macro avg", "weighted avg"):
            continue
        if isinstance(vals, dict):
            per_entity[key] = {
                "precision": float(vals.get("precision", 0)),
                "recall": float(vals.get("recall", 0)),
                "f1": float(vals.get("f1-score", 0)),
                "support": int(vals.get("support", 0)),
            }

    averages = {}
    for avg_type in ("micro avg", "macro avg", "weighted avg"):
        if avg_type in report_dict and isinstance(report_dict[avg_type], dict):
            averages[avg_type] = {
                "precision": float(report_dict[avg_type].get("precision", 0)),
                "recall": float(report_dict[avg_type].get("recall", 0)),
                "f1": float(report_dict[avg_type].get("f1-score", 0)),
                "support": int(report_dict[avg_type].get("support", 0)),
            }

    return {
        "overall": {"f1": f1, "precision": prec, "recall": rec},
        "per_entity": per_entity,
        "averages": averages,
        "report": report_text,
    }


def compute_confusion_matrix(golds, preds, tag_map):
    tag_list = sorted(tag_map.tag2idx.keys())
    tag_to_idx = {t: i for i, t in enumerate(tag_list)}
    n = len(tag_list)
    matrix = [[0] * n for _ in range(n)]

    for gold_seq, pred_seq in zip(golds, preds):
        for g, p in zip(gold_seq, pred_seq):
            gi = tag_to_idx.get(g, 0)
            pi = tag_to_idx.get(p, 0)
            matrix[gi][pi] += 1

    return {"labels": tag_list, "matrix": matrix}


def evaluate_checkpoint(cfg, checkpoint_path, test_path, vocab_dir, device, batch_size=64):
    model, tag_map = load_model(cfg, vocab_dir, checkpoint_path, device)
    test_data = SavedNERDataset(test_path)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    log.info(f"Evaluating {checkpoint_path} on {len(test_data)} samples...")
    golds, preds = predict_dataset(model, test_loader, tag_map, device)
    metrics = compute_metrics(golds, preds)
    confusion = compute_confusion_matrix(golds, preds, tag_map)

    return {**metrics, "confusion_matrix": confusion}


def run_full_evaluation(cfg, checkpoints, test_sets, vocab_dir, device, batch_size=64):
    results = {}

    for ckpt_name, ckpt_path in checkpoints.items():
        if not Path(ckpt_path).exists():
            log.warning(f"Checkpoint not found: {ckpt_path}")
            continue

        results[ckpt_name] = {}

        for test_name, test_path in test_sets.items():
            if not Path(test_path).exists():
                log.warning(f"Test set not found: {test_path}")
                continue

            log.info(f"Evaluating [{ckpt_name}] on [{test_name}]...")
            eval_result = evaluate_checkpoint(cfg, ckpt_path, test_path, vocab_dir, device, batch_size)

            report = eval_result.pop("report")
            results[ckpt_name][test_name] = eval_result

            log.info(f"  F1={eval_result['overall']['f1']:.4f}  P={eval_result['overall']['precision']:.4f}  R={eval_result['overall']['recall']:.4f}")
            log.info(f"\n{report}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--vocab_dir", default="data/processed")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--test_dir", default="data/processed")
    parser.add_argument("--output", default="results/evaluation_results.json")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = {
        "en_conll": str(Path(args.ckpt_dir) / "en_conll_best.pt"),
        "ru_wikiann": str(Path(args.ckpt_dir) / "ru_wikiann_best.pt"),
        "merged": str(Path(args.ckpt_dir) / "merged_best.pt"),
    }

    test_sets = {
        "conll2003_test": str(Path(args.test_dir) / "conll2003_test.pt"),
        "wikiann_en_test": str(Path(args.test_dir) / "wikiann_en_test.pt"),
        "wikiann_ru_test": str(Path(args.test_dir) / "wikiann_ru_test.pt"),
    }

    results = run_full_evaluation(cfg, checkpoints, test_sets, args.vocab_dir, device, args.batch_size)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Full evaluation saved to {out_path}")


if __name__ == "__main__":
    main()
