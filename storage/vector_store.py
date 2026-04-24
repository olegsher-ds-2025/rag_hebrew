import faiss
import numpy as np
import json
from pathlib import Path


class VectorStore:
    def __init__(self, dim=None):
        if dim is not None:
            self.index = faiss.IndexFlatL2(dim)
        else:
            self.index = None
        self.texts = []

    def add(self, embeddings, texts):
        arr = np.array(embeddings).astype('float32')
        if self.index is None:
            # infer dim
            dim = arr.shape[1]
            self.index = faiss.IndexFlatL2(int(dim))

        self.index.add(arr)
        self.texts.extend(texts)

    def search(self, query_embedding, k=5):
        q = np.array(query_embedding).astype('float32')
        if q.ndim == 1:
            q = q.reshape(1, -1)
        D, I = self.index.search(q, k)
        return [self.texts[i] for i in I[0] if i != -1]

    def save(self, dir_path):
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        # save faiss index
        if self.index is not None:
            faiss.write_index(self.index, str(p / 'index.faiss'))
        # save texts
        with open(p / 'texts.json', 'w', encoding='utf-8') as f:
            json.dump(self.texts, f, ensure_ascii=False)

    @classmethod
    def load(cls, dir_path):
        p = Path(dir_path)
        if not p.exists():
            raise FileNotFoundError(f"Vector store dir not found: {dir_path}")
        inst = cls()
        idx_path = p / 'index.faiss'
        if idx_path.exists():
            inst.index = faiss.read_index(str(idx_path))
        texts_path = p / 'texts.json'
        if texts_path.exists():
            with open(texts_path, 'r', encoding='utf-8') as f:
                inst.texts = json.load(f)
        return inst