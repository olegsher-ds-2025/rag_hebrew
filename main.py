"""
Quickstart demo: incrementally index one file and run a test query.

Non-destructive — appends to the existing indices via index_new_files()
(files already indexed are skipped). To rebuild everything from data/raw/,
use scripts/build_index_all.py.
"""

import sys

from rag.pipeline import RAGPipeline


def main(file_path: str = "data/raw/45.pdf") -> None:
    rag = RAGPipeline()
    added = rag.index_new_files([file_path])
    print(f"Indexed {added} new chunks from {file_path}")
    print(rag.query("מה כתוב במסמך?"))


if __name__ == "__main__":
    main(*sys.argv[1:2])
