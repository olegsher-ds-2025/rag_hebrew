from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        # E5 models require task-specific prefixes for best performance
        self._use_e5_prefix = "e5" in model_name.lower()

    def encode(self, texts, is_query=False):
        if self._use_e5_prefix:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        return self.model.encode(texts, show_progress_bar=False)
