import os
import sys
import re
import json
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vocab import Vocabulary, CharVocabulary, TagMap
from model.ner_model import BiLSTMCRF

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LANG_MAP = {"en": 0, "ru": 1}
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.0


@dataclass
class NERResult:
    text: str
    entities: List[Entity] = field(default_factory=list)
    lang: str = "en"
    tokens: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "text": self.text,
            "lang": self.lang,
            "entities": [asdict(e) for e in self.entities],
            "tokens": self.tokens,
            "tags": self.tags,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def get_entities_by_type(self, entity_type):
        return [e for e in self.entities if e.label == entity_type]

    def has_entities(self):
        return len(self.entities) > 0


class NEREngine:
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
        log.info(f"NER engine loaded on {self.device}")

    def detect_lang(self, text):
        cyrillic = len(CYRILLIC_RE.findall(text))
        return "ru" if cyrillic > len(text.split()) * 0.3 else "en"

    def tokenize(self, text):
        return text.strip().split()

    def predict(self, text):
        tokens = self.tokenize(text)
        if not tokens:
            return NERResult(text=text)

        lang = self.detect_lang(text)
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
        entities = self._extract_entities(tokens, tags, text)

        return NERResult(
            text=text,
            entities=entities,
            lang=lang,
            tokens=tokens,
            tags=tags,
        )

    def _extract_entities(self, tokens, tags, original_text):
        entities = []
        current_label = None
        current_tokens = []
        current_start = 0

        char_pos = 0
        token_positions = []
        for token in tokens:
            idx = original_text.lower().find(token.lower(), char_pos)
            if idx == -1:
                idx = char_pos
            token_positions.append(idx)
            char_pos = idx + len(token)

        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if tag.startswith("B-"):
                if current_label:
                    entity_text = " ".join(current_tokens)
                    entities.append(Entity(
                        text=entity_text,
                        label=current_label,
                        start=current_start,
                        end=token_positions[i - 1] + len(tokens[i - 1]),
                    ))
                current_label = tag[2:]
                current_tokens = [token]
                current_start = token_positions[i]
            elif tag.startswith("I-") and current_label == tag[2:]:
                current_tokens.append(token)
            else:
                if current_label:
                    entity_text = " ".join(current_tokens)
                    entities.append(Entity(
                        text=entity_text,
                        label=current_label,
                        start=current_start,
                        end=token_positions[i - 1] + len(tokens[i - 1]),
                    ))
                current_label = None
                current_tokens = []

        if current_label:
            entity_text = " ".join(current_tokens)
            entities.append(Entity(
                text=entity_text,
                label=current_label,
                start=current_start,
                end=token_positions[-1] + len(tokens[-1]),
            ))

        return entities

    def batch_predict(self, texts):
        return [self.predict(text) for text in texts]


class RAGEntityFilter:
    ENTITY_PRIORITY = {
        "PER": 3,
        "ORG": 3,
        "LOC": 2,
        "MISC": 1,
    }

    def __init__(self, min_confidence=0.0, allowed_types=None):
        self.min_confidence = min_confidence
        self.allowed_types = allowed_types

    def filter_entities(self, ner_result):
        entities = ner_result.entities

        if self.allowed_types:
            entities = [e for e in entities if e.label in self.allowed_types]

        if self.min_confidence > 0:
            entities = [e for e in entities if e.confidence >= self.min_confidence]

        return entities

    def deduplicate(self, entities):
        seen = {}
        for entity in entities:
            key = (entity.text.lower(), entity.label)
            if key not in seen:
                seen[key] = entity
        return list(seen.values())

    def rank_entities(self, entities):
        return sorted(
            entities,
            key=lambda e: self.ENTITY_PRIORITY.get(e.label, 0),
            reverse=True,
        )


class RAGQueryEnricher:
    def __init__(self, ner_engine, entity_filter=None):
        self.ner_engine = ner_engine
        self.entity_filter = entity_filter or RAGEntityFilter()

    def enrich_query(self, query):
        ner_result = self.ner_engine.predict(query)
        entities = self.entity_filter.filter_entities(ner_result)
        entities = self.entity_filter.deduplicate(entities)
        entities = self.entity_filter.rank_entities(entities)

        enriched = {
            "original_query": query,
            "language": ner_result.lang,
            "entities": [asdict(e) for e in entities],
            "entity_types": list(set(e.label for e in entities)),
            "entity_texts": [e.text for e in entities],
        }

        expanded_query = self._expand_query(query, entities)
        enriched["expanded_query"] = expanded_query

        return enriched

    def _expand_query(self, query, entities):
        if not entities:
            return query

        entity_context = []
        for entity in entities:
            entity_context.append(f"[{entity.label}: {entity.text}]")

        return f"{query} {' '.join(entity_context)}"

    def build_retrieval_context(self, query):
        enrichment = self.enrich_query(query)

        context = {
            "query": enrichment["expanded_query"],
            "filters": {},
            "boost_terms": enrichment["entity_texts"],
            "metadata": {
                "language": enrichment["language"],
                "detected_entities": enrichment["entities"],
            },
        }

        for entity in enrichment["entities"]:
            label = entity["label"]
            if label not in context["filters"]:
                context["filters"][label] = []
            context["filters"][label].append(entity["text"])

        return context

    def process_batch(self, queries):
        return [self.build_retrieval_context(q) for q in queries]


def create_pipeline(config_path="configs/config.yaml",
                    checkpoint_path="checkpoints/merged_best.pt",
                    vocab_dir="data/processed",
                    allowed_entity_types=None):
    engine = NEREngine(config_path, checkpoint_path, vocab_dir)
    entity_filter = RAGEntityFilter(allowed_types=allowed_entity_types)
    enricher = RAGQueryEnricher(engine, entity_filter)
    return enricher


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/merged_best.pt")
    parser.add_argument("--vocab_dir", default="data/processed")
    parser.add_argument("--input", default=None)
    args = parser.parse_args()

    pipeline = create_pipeline(args.config, args.checkpoint, args.vocab_dir)

    if args.input:
        result = pipeline.build_retrieval_context(args.input)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Interactive mode (type 'quit' to exit):")
        while True:
            text = input("\n> ")
            if text.lower() in ("quit", "exit", "q"):
                break
            result = pipeline.build_retrieval_context(text)
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
