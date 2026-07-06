DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

TOP_K = 10

# Reciprocal Rank Fusion constant for merging vector + keyword result lists.
# 60 is the value from the original RRF paper (Cormack et al., 2009); larger
# values flatten the contribution of rank position.
RRF_K = 60
