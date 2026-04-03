import os
import sys
import json
import logging
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import SavedNERDataset, collate_batch
from data.vocab import Vocabulary, CharVocabulary, TagMap
from model.ner_model import BiLSTMCRF

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_evaluation(model, loader, tag_map, device):
    model.eval()
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

    f1 = f1_score(all_golds, all_preds)
    prec = precision_score(all_golds, all_preds)
    rec = recall_score(all_golds, all_preds)

    report = classification_report(all_golds, all_preds, output_dict=True)

    metrics = {
        "overall": {"f1": f1, "precision": prec, "recall": rec},
        "per_entity": {}
    }

    for key, vals in report.items():
        if key in ("micro avg", "macro avg", "weighted avg"):
            continue
        if isinstance(vals, dict):
            metrics["per_entity"][key] = {
                "precision": vals.get("precision", 0),
                "recall": vals.get("recall", 0),
                "f1-score": vals.get("f1-score", 0),
                "support": vals.get("support", 0),
            }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--vocab_dir", default="data/processed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/evaluation_results.json")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    test_data = SavedNERDataset(args.test_data)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    log.info(f"Evaluating on {len(test_data)} samples...")
    metrics = run_evaluation(model, test_loader, tag_map, device)

    log.info(f"Overall F1: {metrics['overall']['f1']:.4f}")
    log.info(f"Precision: {metrics['overall']['precision']:.4f}")
    log.info(f"Recall: {metrics['overall']['recall']:.4f}")

    for ent, vals in metrics["per_entity"].items():
        log.info(f"  {ent}: P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1-score']:.3f} ({vals['support']})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
