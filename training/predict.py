import os
import sys
import re
import json
import yaml
import logging
import argparse
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vocab import Vocabulary, CharVocabulary, TagMap
from model.ner_model import BiLSTMCRF

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LANG_MAP = {"en": 0, "ru": 1}
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def detect_lang(text):
    cyrillic_count = len(CYRILLIC_RE.findall(text))
    return "ru" if cyrillic_count > len(text.split()) * 0.3 else "en"


def tokenize(text):
    tokens = text.strip().split()
    return tokens


class NERPredictor:
    def __init__(self, config_path, checkpoint_path, vocab_dir):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.word_vocab = Vocabulary.load(Path(vocab_dir) / "word_vocab.json")
        self.char_vocab = CharVocabulary.load(Path(vocab_dir) / "char_vocab.json")
        self.tag_map = TagMap.load(Path(vocab_dir) / "tag_map.json")

        self.model = BiLSTMCRF(
            vocab_size=len(self.word_vocab),
            num_chars=len(self.char_vocab),
            num_tags=len(self.tag_map),
            word_dim=self.cfg["embeddings"]["word_dim"],
            char_dim=self.cfg["embeddings"]["char_dim"],
            char_filters=self.cfg["embeddings"]["char_filters"],
            char_kernel=self.cfg["embeddings"]["char_kernel_size"],
            num_langs=self.cfg["model"]["num_langs"],
            lang_dim=self.cfg["embeddings"]["lang_dim"],
            hidden_size=self.cfg["model"]["hidden_size"],
            num_layers=self.cfg["model"]["num_layers"],
            dropout=0.0,
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()
        log.info("Model loaded and ready")

    def predict_text(self, text):
        tokens = tokenize(text)
        if not tokens:
            return []

        lang = detect_lang(text)
        lang_id = LANG_MAP.get(lang, 0)
        max_word_len = self.cfg["data"]["max_word_len"]

        lowered = [t.lower() for t in tokens]
        word_ids = torch.tensor([[self.word_vocab.encode(t) for t in lowered]], dtype=torch.long)
        char_ids = torch.tensor([[self.char_vocab.encode_word(t, max_word_len) for t in lowered]], dtype=torch.long)
        lang_ids = torch.tensor([lang_id], dtype=torch.long)
        mask = torch.ones(1, len(tokens), dtype=torch.float)
        lengths = torch.tensor([len(tokens)], dtype=torch.long)

        word_ids = word_ids.to(self.device)
        char_ids = char_ids.to(self.device)
        lang_ids = lang_ids.to(self.device)
        mask = mask.to(self.device)
        lengths = lengths.to(self.device)

        with torch.no_grad():
            pred_ids = self.model.predict(word_ids, char_ids, lang_ids, mask, lengths)

        tags = [self.tag_map.decode(pred_ids[0][i].item()) for i in range(len(tokens))]
        return list(zip(tokens, tags))

    def extract_entities(self, text):
        tagged = self.predict_text(text)
        entities = []
        current_entity = None
        current_tokens = []

        for token, tag in tagged:
            if tag.startswith("B-"):
                if current_entity:
                    entities.append({"type": current_entity, "text": " ".join(current_tokens)})
                current_entity = tag[2:]
                current_tokens = [token]
            elif tag.startswith("I-") and current_entity == tag[2:]:
                current_tokens.append(token)
            else:
                if current_entity:
                    entities.append({"type": current_entity, "text": " ".join(current_tokens)})
                current_entity = None
                current_tokens = []

        if current_entity:
            entities.append({"type": current_entity, "text": " ".join(current_tokens)})

        return entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab_dir", default="data/processed")
    parser.add_argument("--input", default=None)
    parser.add_argument("--input_file", default=None)
    args = parser.parse_args()

    predictor = NERPredictor(args.config, args.checkpoint, args.vocab_dir)

    if args.input:
        entities = predictor.extract_entities(args.input)
        tagged = predictor.predict_text(args.input)
        print("\nTagged output:")
        for token, tag in tagged:
            print(f"  {token:20s} {tag}")
        print("\nExtracted entities:")
        for ent in entities:
            print(f"  [{ent['type']}] {ent['text']}")

    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        all_entities = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            entities = predictor.extract_entities(line)
            all_entities.extend(entities)
            for ent in entities:
                print(f"  [{ent['type']}] {ent['text']}")

        print(f"\nTotal entities found: {len(all_entities)}")

    else:
        print("Interactive mode (type 'quit' to exit):")
        while True:
            text = input("\n> ")
            if text.lower() in ("quit", "exit", "q"):
                break
            entities = predictor.extract_entities(text)
            tagged = predictor.predict_text(text)
            for token, tag in tagged:
                if tag != "O":
                    print(f"  {token:20s} {tag}")
            if entities:
                print("Entities:", json.dumps(entities, ensure_ascii=False))
            else:
                print("No entities found")


if __name__ == "__main__":
    main()
