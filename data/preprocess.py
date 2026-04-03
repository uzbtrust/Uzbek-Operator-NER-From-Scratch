import json
import logging
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from data.vocab import Vocabulary, CharVocabulary, TagMap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LANG_MAP = {"en": 0, "ru": 1}


class NERDataset(Dataset):
    def __init__(self, samples, word_vocab, char_vocab, tag_map, max_seq_len=128, max_word_len=30):
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.tag_map = tag_map
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len
        self.data = self._process(samples)

    def _process(self, samples):
        processed = []
        for sample in samples:
            tokens = [t.lower() for t in sample["tokens"]][:self.max_seq_len]
            tags = sample["tags"][:self.max_seq_len]
            lang = LANG_MAP.get(sample.get("lang", "en"), 0)

            word_ids = [self.word_vocab.encode(t) for t in tokens]
            char_ids = [self.char_vocab.encode_word(t, self.max_word_len) for t in tokens]
            tag_ids = [self.tag_map.encode(t) for t in tags]

            processed.append({
                "word_ids": torch.tensor(word_ids, dtype=torch.long),
                "char_ids": torch.tensor(char_ids, dtype=torch.long),
                "tag_ids": torch.tensor(tag_ids, dtype=torch.long),
                "lang_id": torch.tensor(lang, dtype=torch.long),
                "length": len(tokens),
            })
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SavedNERDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_batch(batch):
    word_ids = pad_sequence([s["word_ids"] for s in batch], batch_first=True, padding_value=0)
    tag_ids = pad_sequence([s["tag_ids"] for s in batch], batch_first=True, padding_value=0)
    lengths = torch.tensor([s["length"] for s in batch], dtype=torch.long)
    lang_ids = torch.stack([s["lang_id"] for s in batch])

    max_len = word_ids.size(1)
    max_word_len = batch[0]["char_ids"].size(1)
    char_ids = torch.zeros(len(batch), max_len, max_word_len, dtype=torch.long)
    for i, s in enumerate(batch):
        seq_len = s["char_ids"].size(0)
        char_ids[i, :seq_len] = s["char_ids"]

    mask = torch.zeros(len(batch), max_len, dtype=torch.float)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1.0

    return {
        "word_ids": word_ids,
        "char_ids": char_ids,
        "tag_ids": tag_ids,
        "lang_ids": lang_ids,
        "mask": mask,
        "lengths": lengths,
    }


def load_raw_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_dataset(raw_dir, split, word_vocab, char_vocab, tag_map, max_seq_len=128, max_word_len=30):
    samples = load_raw_data(Path(raw_dir) / f"{split}.json")
    return NERDataset(samples, word_vocab, char_vocab, tag_map, max_seq_len, max_word_len)


def process_and_save(raw_dirs, vocab_dir, output_dir, max_seq_len=128, max_word_len=30):
    word_vocab = Vocabulary.load(Path(vocab_dir) / "word_vocab.json")
    char_vocab = CharVocabulary.load(Path(vocab_dir) / "char_vocab.json")
    tag_map = TagMap.load(Path(vocab_dir) / "tag_map.json")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for raw_dir in raw_dirs:
        name = Path(raw_dir).name
        for split in ["train", "validation", "test"]:
            fpath = Path(raw_dir) / f"{split}.json"
            if not fpath.exists():
                continue

            ds = create_dataset(raw_dir, split, word_vocab, char_vocab, tag_map, max_seq_len, max_word_len)
            save_path = out / f"{name}_{split}.pt"
            torch.save(ds.data, save_path)
            log.info(f"Saved {len(ds)} samples to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--vocab_dir", default="data/processed")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--max_word_len", type=int, default=30)
    args = parser.parse_args()

    raw_dirs = [
        Path(args.raw_dir) / "conll2003",
        Path(args.raw_dir) / "wikiann_en",
        Path(args.raw_dir) / "wikiann_ru",
    ]

    process_and_save(raw_dirs, args.vocab_dir, args.output_dir, args.max_seq_len, args.max_word_len)


if __name__ == "__main__":
    main()
