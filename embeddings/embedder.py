import logging

logger = logging.getLogger(__name__)


class Embedder:
    """
    SentenceTransformer wrapper with lazy loading and a batch-size knob.

    The ~2 GB model is NOT built until the first encode()/warmup() — importing
    this module (or constructing an Embedder) never pulls in torch, which keeps
    scripts, tests and API startup cheap until embeddings are actually needed.
    """

    def __init__(self, model_name, batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size
        # E5 models require task-specific prefixes for best performance
        self._use_e5_prefix = "e5" in model_name.lower()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy: pulls torch
            logger.info("Loading embedding model %s ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def warmup(self):
        """Force the model to load now (e.g. at API startup for readiness)."""
        _ = self.model

    def encode(self, texts, is_query=False):
        if self._use_e5_prefix:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        # Unit-normalize so inner-product search (IndexFlatIP) ranks by cosine
        # similarity — what E5 (and sentence-transformers generally) is trained
        # for. Un-normalized L2 distance is NOT rank-equivalent to cosine.
        return self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=self.batch_size,
        )
