import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Bumped when the on-disk format/metric changes. v2 = inner-product index over
# unit-normalized embeddings (cosine). v1 stores (no meta.json, L2 metric over
# un-normalized vectors) are refused on load and must be rebuilt.
STORE_VERSION = 2


class StoreCorruptError(RuntimeError):
    """Persisted store is missing, stale-format, or internally inconsistent."""


class VectorStore:
    def __init__(self, dim=None):
        if dim is not None:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None
        self.texts = []

    def add(self, embeddings, texts):
        arr = np.array(embeddings).astype('float32')
        if self.index is None:
            # infer dim
            dim = arr.shape[1]
            self.index = faiss.IndexFlatIP(int(dim))

        self.index.add(arr)
        self.texts.extend(texts)

    def search(self, query_embedding, k=5):
        if self.index is None or self.index.ntotal == 0:
            return []
        q = np.array(query_embedding).astype('float32')
        if q.ndim == 1:
            q = q.reshape(1, -1)
        _dists, ids = self.index.search(q, k)
        # Guard i < len(texts): a desynced store must not crash queries.
        return [self.texts[i] for i in ids[0] if 0 <= i < len(self.texts)]

    def save(self, dir_path):
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        # Write each artifact to a temp file then atomically replace, so a
        # crash mid-save never leaves a half-written index/texts pair behind.
        if self.index is not None:
            tmp_idx = p / 'index.faiss.tmp'
            faiss.write_index(self.index, str(tmp_idx))
            os.replace(tmp_idx, p / 'index.faiss')
        tmp_txt = p / 'texts.json.tmp'
        with open(tmp_txt, 'w', encoding='utf-8') as f:
            json.dump(self.texts, f, ensure_ascii=False)
        os.replace(tmp_txt, p / 'texts.json')
        tmp_meta = p / 'meta.json.tmp'
        with open(tmp_meta, 'w', encoding='utf-8') as f:
            json.dump({'version': STORE_VERSION, 'count': len(self.texts)}, f)
        os.replace(tmp_meta, p / 'meta.json')

    @classmethod
    def load(cls, dir_path):
        p = Path(dir_path)
        if not p.exists():
            raise FileNotFoundError(f"Vector store dir not found: {dir_path}")

        meta_path = p / 'meta.json'
        if not meta_path.exists():
            raise StoreCorruptError(
                f"{dir_path} is a pre-v{STORE_VERSION} store (L2 metric, un-normalized "
                "embeddings). Rebuild it: python scripts/build_index_all.py"
            )
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        if meta.get('version') != STORE_VERSION:
            raise StoreCorruptError(
                f"{dir_path} has store version {meta.get('version')}, expected "
                f"{STORE_VERSION}. Rebuild it: python scripts/build_index_all.py"
            )

        inst = cls()
        idx_path = p / 'index.faiss'
        if idx_path.exists():
            inst.index = faiss.read_index(str(idx_path))
        texts_path = p / 'texts.json'
        if texts_path.exists():
            with open(texts_path, encoding='utf-8') as f:
                inst.texts = json.load(f)

        ntotal = inst.index.ntotal if inst.index is not None else 0
        if ntotal != len(inst.texts):
            raise StoreCorruptError(
                f"{dir_path} is inconsistent: index has {ntotal} vectors but "
                f"{len(inst.texts)} texts. Rebuild it: python scripts/build_index_all.py"
            )
        return inst
