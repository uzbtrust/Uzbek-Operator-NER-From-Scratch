import gzip
import logging
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FASTTEXT_URLS = {
    "en": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz",
    "ru": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz",
}


def download_vectors(lang, output_dir):
    url = FASTTEXT_URLS[lang]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gz_path = out_dir / f"cc.{lang}.300.vec.gz"
    vec_path = out_dir / f"cc.{lang}.300.vec"

    if vec_path.exists():
        log.info(f"FastText vectors already exist: {vec_path}")
        return vec_path

    if not gz_path.exists():
        log.info(f"Downloading FastText ({lang}) from {url}...")
        urllib.request.urlretrieve(url, gz_path)
        log.info(f"Downloaded to {gz_path}")

    log.info(f"Extracting {gz_path}...")
    with gzip.open(gz_path, "rt", encoding="utf-8") as fin:
        with open(vec_path, "w", encoding="utf-8") as fout:
            for line in fin:
                fout.write(line)

    log.info(f"Extracted to {vec_path}")
    return vec_path


def load_vectors(vec_path, max_vectors=500000):
    log.info(f"Loading vectors from {vec_path}...")
    vectors = {}
    with open(vec_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for i, line in enumerate(f):
            if i >= max_vectors:
                break
            parts = line.rstrip().split(" ")
            word = parts[0].lower()
            if word not in vectors:
                try:
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    if len(vec) == 300:
                        vectors[word] = vec
                except ValueError:
                    continue

    log.info(f"Loaded {len(vectors)} vectors from {vec_path}")
    return vectors


def build_embedding_matrix(word2idx, vectors, embed_dim=300):
    vocab_size = len(word2idx)
    matrix = np.random.uniform(-0.05, 0.05, (vocab_size, embed_dim)).astype(np.float32)
    matrix[0] = np.zeros(embed_dim)

    found = 0
    for word, idx in word2idx.items():
        if word in vectors:
            matrix[idx] = vectors[word]
            found += 1

    log.info(f"Initialized {found}/{vocab_size} word embeddings from pretrained vectors")
    return torch.tensor(matrix)


def load_and_build(word2idx, lang, vectors_dir, max_vectors=500000):
    vec_path = Path(vectors_dir) / f"cc.{lang}.300.vec"
    if not vec_path.exists():
        vec_path = download_vectors(lang, vectors_dir)

    vectors = load_vectors(vec_path, max_vectors)
    return build_embedding_matrix(word2idx, vectors)


def merge_embeddings(word2idx, vectors_dir, max_vectors=500000):
    en_vecs = load_vectors(Path(vectors_dir) / "cc.en.300.vec", max_vectors)
    ru_vecs = load_vectors(Path(vectors_dir) / "cc.ru.300.vec", max_vectors)

    merged = {}
    merged.update(en_vecs)
    merged.update(ru_vecs)

    return build_embedding_matrix(word2idx, merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["en", "ru", "both"], default="both")
    parser.add_argument("--output_dir", default="embeddings/vectors")
    parser.add_argument("--max_vectors", type=int, default=500000)
    args = parser.parse_args()

    if args.lang == "both":
        download_vectors("en", args.output_dir)
        download_vectors("ru", args.output_dir)
    else:
        download_vectors(args.lang, args.output_dir)


if __name__ == "__main__":
    main()
