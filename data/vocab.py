import json
import logging
import argparse
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_CHAR = "<pad>"

TAG_LIST = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
    "B-LOC", "I-LOC", "B-MISC", "I-MISC"
]


class Vocabulary:
    def __init__(self):
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self.word_freq = Counter()

    def add_tokens(self, tokens):
        for tok in tokens:
            self.word_freq[tok] += 1

    def build(self, min_freq=2):
        idx = len(self.word2idx)
        for word, count in self.word_freq.most_common():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        log.info(f"Vocabulary built: {len(self.word2idx)} words (min_freq={min_freq})")

    def encode(self, token):
        return self.word2idx.get(token, self.word2idx[UNK_TOKEN])

    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "word_freq": dict(self.word_freq)}, f, ensure_ascii=False)
        log.info(f"Vocabulary saved to {path}")

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        vocab.word_freq = Counter(data.get("word_freq", {}))
        log.info(f"Vocabulary loaded from {path}: {len(vocab)} words")
        return vocab


class CharVocabulary:
    def __init__(self):
        self.char2idx = {PAD_CHAR: 0}

    def build_from_words(self, words):
        idx = len(self.char2idx)
        for word in words:
            for ch in word:
                if ch not in self.char2idx:
                    self.char2idx[ch] = idx
                    idx += 1
        log.info(f"Character vocabulary built: {len(self.char2idx)} chars")

    def encode(self, char):
        return self.char2idx.get(char, 0)

    def encode_word(self, word, max_len):
        indices = [self.encode(ch) for ch in word[:max_len]]
        indices += [0] * (max_len - len(indices))
        return indices

    def __len__(self):
        return len(self.char2idx)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab.char2idx = data
        log.info(f"Character vocabulary loaded: {len(vocab)} chars")
        return vocab


class TagMap:
    def __init__(self, tags=None):
        tags = tags or TAG_LIST
        self.tag2idx = {tag: i for i, tag in enumerate(tags)}
        self.idx2tag = {i: tag for i, tag in enumerate(tags)}

    def encode(self, tag):
        return self.tag2idx.get(tag, 0)

    def decode(self, idx):
        return self.idx2tag.get(idx, "O")

    def __len__(self):
        return len(self.tag2idx)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.tag2idx, f)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        tm = cls()
        tm.tag2idx = data
        tm.idx2tag = {v: k for k, v in data.items()}
        return tm


def build_vocabs(data_dirs, output_dir, min_freq=2):
    word_vocab = Vocabulary()
    all_words = set()

    for data_dir in data_dirs:
        train_file = Path(data_dir) / "train.json"
        if not train_file.exists():
            log.warning(f"Train file not found: {train_file}")
            continue

        with open(train_file, "r", encoding="utf-8") as f:
            samples = json.load(f)

        for sample in samples:
            lowered = [t.lower() for t in sample["tokens"]]
            word_vocab.add_tokens(lowered)
            all_words.update(lowered)

    word_vocab.build(min_freq=min_freq)

    char_vocab = CharVocabulary()
    char_vocab.build_from_words(all_words)

    tag_map = TagMap()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    word_vocab.save(out / "word_vocab.json")
    char_vocab.save(out / "char_vocab.json")
    tag_map.save(out / "tag_map.json")

    return word_vocab, char_vocab, tag_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--min_freq", type=int, default=2)
    args = parser.parse_args()

    data_dirs = [
        Path(args.raw_dir) / "conll2003",
        Path(args.raw_dir) / "wikiann_en",
        Path(args.raw_dir) / "wikiann_ru",
    ]

    build_vocabs(data_dirs, args.output_dir, min_freq=args.min_freq)


if __name__ == "__main__":
    main()
