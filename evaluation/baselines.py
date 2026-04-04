import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocess import SavedNERDataset, collate_batch
from data.vocab import TagMap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class MajorityBaseline:
    def __init__(self):
        self.majority_tag = "O"

    def fit(self, loader, tag_map):
        tag_counts = Counter()
        for batch in loader:
            tag_ids = batch["tag_ids"]
            lengths = batch["lengths"]
            for i in range(len(lengths)):
                for t in range(lengths[i].item()):
                    tag = tag_map.decode(tag_ids[i][t].item())
                    tag_counts[tag] += 1

        self.majority_tag = tag_counts.most_common(1)[0][0]
        log.info(f"Majority tag: {self.majority_tag} ({tag_counts[self.majority_tag]} occurrences)")

    def predict(self, loader, tag_map):
        all_preds = []
        all_golds = []

        for batch in loader:
            tag_ids = batch["tag_ids"]
            lengths = batch["lengths"]

            for i in range(len(lengths)):
                ln = lengths[i].item()
                gold_seq = [tag_map.decode(tag_ids[i][t].item()) for t in range(ln)]
                pred_seq = [self.majority_tag] * ln
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

        return all_golds, all_preds


class RandomBaseline:
    def __init__(self, seed=42):
        self.tags = []
        self.seed = seed

    def fit(self, tag_map):
        self.tags = list(tag_map.tag2idx.keys())

    def predict(self, loader, tag_map):
        import random
        random.seed(self.seed)

        all_preds = []
        all_golds = []

        for batch in loader:
            tag_ids = batch["tag_ids"]
            lengths = batch["lengths"]

            for i in range(len(lengths)):
                ln = lengths[i].item()
                gold_seq = [tag_map.decode(tag_ids[i][t].item()) for t in range(ln)]
                pred_seq = [random.choice(self.tags) for _ in range(ln)]
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

        return all_golds, all_preds


class FrequencyBaseline:
    def __init__(self):
        self.word_tag_map = {}
        self.default_tag = "O"

    def fit(self, train_loader, tag_map, word_vocab):
        word_tag_counts = {}

        for batch in train_loader:
            word_ids = batch["word_ids"]
            tag_ids = batch["tag_ids"]
            lengths = batch["lengths"]

            for i in range(len(lengths)):
                for t in range(lengths[i].item()):
                    wid = word_ids[i][t].item()
                    tid = tag_ids[i][t].item()
                    tag = tag_map.decode(tid)

                    if wid not in word_tag_counts:
                        word_tag_counts[wid] = Counter()
                    word_tag_counts[wid][tag] += 1

        for wid, counts in word_tag_counts.items():
            self.word_tag_map[wid] = counts.most_common(1)[0][0]

        log.info(f"Frequency baseline built with {len(self.word_tag_map)} word-tag entries")

    def predict(self, loader, tag_map):
        all_preds = []
        all_golds = []

        for batch in loader:
            word_ids = batch["word_ids"]
            tag_ids = batch["tag_ids"]
            lengths = batch["lengths"]

            for i in range(len(lengths)):
                ln = lengths[i].item()
                gold_seq = [tag_map.decode(tag_ids[i][t].item()) for t in range(ln)]
                pred_seq = []
                for t in range(ln):
                    wid = word_ids[i][t].item()
                    pred_seq.append(self.word_tag_map.get(wid, self.default_tag))
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

        return all_golds, all_preds


def evaluate_baseline(golds, preds, name):
    f1 = float(f1_score(golds, preds))
    prec = float(precision_score(golds, preds))
    rec = float(recall_score(golds, preds))

    report_dict = classification_report(golds, preds, output_dict=True)

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

    log.info(f"[{name}] F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}")

    return {
        "name": name,
        "overall": {"f1": f1, "precision": prec, "recall": rec},
        "per_entity": per_entity,
    }


def run_all_baselines(test_path, train_path, vocab_dir, batch_size=64):
    tag_map = TagMap.load(Path(vocab_dir) / "tag_map.json")

    test_data = SavedNERDataset(test_path)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    results = {}

    majority = MajorityBaseline()
    train_data = SavedNERDataset(train_path)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    majority.fit(train_loader, tag_map)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    golds, preds = majority.predict(test_loader, tag_map)
    results["majority"] = evaluate_baseline(golds, preds, "Majority (always O)")

    rand_baseline = RandomBaseline()
    rand_baseline.fit(tag_map)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    golds, preds = rand_baseline.predict(test_loader, tag_map)
    results["random"] = evaluate_baseline(golds, preds, "Random")

    from data.vocab import Vocabulary
    word_vocab = Vocabulary.load(Path(vocab_dir) / "word_vocab.json")

    freq_baseline = FrequencyBaseline()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    freq_baseline.fit(train_loader, tag_map, word_vocab)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    golds, preds = freq_baseline.predict(test_loader, tag_map)
    results["frequency"] = evaluate_baseline(golds, preds, "Word Frequency")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--vocab_dir", default="data/processed")
    parser.add_argument("--output", default="results/baseline_results.json")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    results = run_all_baselines(args.test_data, args.train_data, args.vocab_dir, args.batch_size)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Baseline results saved to {out_path}")


if __name__ == "__main__":
    main()
