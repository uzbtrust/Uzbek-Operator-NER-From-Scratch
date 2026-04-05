import json
import logging
import argparse
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CONLL_TAG_MAP = {
    0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"
}

WIKIANN_TAG_MAP = {
    0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC"
}

CONLL_PARQUET = {
    "train": "https://huggingface.co/datasets/conll2003/resolve/refs%2Fconvert%2Fparquet/conll2003/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/conll2003/resolve/refs%2Fconvert%2Fparquet/conll2003/validation/0000.parquet",
    "test": "https://huggingface.co/datasets/conll2003/resolve/refs%2Fconvert%2Fparquet/conll2003/test/0000.parquet",
}

WIKIANN_PARQUET = "https://huggingface.co/datasets/wikiann/resolve/refs%2Fconvert%2Fparquet/{lang}/{split}/0000.parquet"


def _load_parquet(data_files):
    return load_dataset("parquet", data_files=data_files)


def download_conll(output_dir):
    log.info("Downloading CoNLL-2003 (parquet)...")
    ds = _load_parquet(CONLL_PARQUET)

    for split in ["train", "validation", "test"]:
        samples = []
        for row in ds[split]:
            tokens = row["tokens"]
            tags = [CONLL_TAG_MAP[t] for t in row["ner_tags"]]
            samples.append({"tokens": tokens, "tags": tags, "lang": "en"})

        out_path = Path(output_dir) / "conll2003" / f"{split}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        log.info(f"Saved {len(samples)} samples to {out_path}")


def download_wikiann(lang, output_dir):
    log.info(f"Downloading WikiANN ({lang}) (parquet)...")
    data_files = {
        split: WIKIANN_PARQUET.format(lang=lang, split=split)
        for split in ["train", "validation", "test"]
    }
    ds = _load_parquet(data_files)

    for split in ["train", "validation", "test"]:
        samples = []
        for row in ds[split]:
            tokens = row["tokens"]
            tags = [WIKIANN_TAG_MAP[t] for t in row["ner_tags"]]
            samples.append({"tokens": tokens, "tags": tags, "lang": lang})

        out_path = Path(output_dir) / f"wikiann_{lang}" / f"{split}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        log.info(f"Saved {len(samples)} samples to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/raw")
    args = parser.parse_args()

    download_conll(args.output_dir)
    download_wikiann("en", args.output_dir)
    download_wikiann("ru", args.output_dir)
    log.info("All datasets downloaded.")


if __name__ == "__main__":
    main()
