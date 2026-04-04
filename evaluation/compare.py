import json
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def format_table(headers, rows, col_width=14):
    header_line = "".join(h.ljust(col_width) for h in headers)
    separator = "-" * len(header_line)
    lines = [header_line, separator]

    for row in rows:
        line = "".join(str(v).ljust(col_width) for v in row)
        lines.append(line)

    return "\n".join(lines)


def build_comparison(model_results, baseline_results, training_results):
    comparison = {
        "model_vs_baselines": {},
        "stage_progression": {},
        "cross_lingual": {},
    }

    if baseline_results:
        for baseline_name, baseline_data in baseline_results.items():
            comparison["model_vs_baselines"][baseline_name] = {
                "baseline_f1": baseline_data["overall"]["f1"],
                "baseline_precision": baseline_data["overall"]["precision"],
                "baseline_recall": baseline_data["overall"]["recall"],
            }

    if training_results:
        for stage_name in ["stage_1_en_conll", "stage_2_ru_wikiann", "stage_3_merged"]:
            if stage_name in training_results:
                stage = training_results[stage_name]
                comparison["stage_progression"][stage_name] = {
                    "best_f1": float(stage["best_f1"]),
                    "num_epochs": len(stage["history"]),
                    "final_loss": float(stage["history"][-1]["loss"]),
                }

        if "test_en" in training_results:
            comparison["cross_lingual"]["english"] = training_results["test_en"]["overall"]
        if "test_ru" in training_results:
            comparison["cross_lingual"]["russian"] = training_results["test_ru"]["overall"]

    return comparison


def print_comparison(comparison, model_f1=None):
    print("\n" + "=" * 60)
    print("NER MODEL EVALUATION SUMMARY")
    print("=" * 60)

    if comparison.get("model_vs_baselines"):
        print("\n--- Model vs Baselines ---")
        headers = ["Method", "F1", "Precision", "Recall"]
        rows = []

        if model_f1:
            rows.append(["BiLSTM-CRF", f"{model_f1:.4f}", "-", "-"])

        for name, data in comparison["model_vs_baselines"].items():
            rows.append([
                name,
                f"{data['baseline_f1']:.4f}",
                f"{data['baseline_precision']:.4f}",
                f"{data['baseline_recall']:.4f}",
            ])

        print(format_table(headers, rows))

    if comparison.get("stage_progression"):
        print("\n--- Training Stage Progression ---")
        headers = ["Stage", "Best F1", "Epochs", "Final Loss"]
        rows = []

        for stage, data in comparison["stage_progression"].items():
            short_name = stage.replace("stage_", "S").replace("_", " ")
            rows.append([
                short_name,
                f"{data['best_f1']:.4f}",
                str(data['num_epochs']),
                f"{data['final_loss']:.4f}",
            ])

        print(format_table(headers, rows))

    if comparison.get("cross_lingual"):
        print("\n--- Cross-Lingual Performance ---")
        headers = ["Language", "F1", "Precision", "Recall"]
        rows = []

        for lang, data in comparison["cross_lingual"].items():
            rows.append([
                lang.capitalize(),
                f"{data['f1']:.4f}",
                f"{data['precision']:.4f}",
                f"{data['recall']:.4f}",
            ])

        print(format_table(headers, rows))

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_results", default="results/training_results.json")
    parser.add_argument("--baseline_results", default=None)
    parser.add_argument("--evaluation_results", default=None)
    parser.add_argument("--output", default="results/comparison_report.json")
    args = parser.parse_args()

    training_results = None
    baseline_results = None
    eval_results = None

    if Path(args.training_results).exists():
        training_results = load_results(args.training_results)

    if args.baseline_results and Path(args.baseline_results).exists():
        baseline_results = load_results(args.baseline_results)

    if args.evaluation_results and Path(args.evaluation_results).exists():
        eval_results = load_results(args.evaluation_results)

    model_f1 = None
    if training_results and "test_en" in training_results:
        model_f1 = training_results["test_en"]["overall"]["f1"]

    comparison = build_comparison(eval_results, baseline_results, training_results)
    print_comparison(comparison, model_f1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    log.info(f"Comparison report saved to {out_path}")


if __name__ == "__main__":
    main()
