import argparse
from pathlib import Path
from typing import Sequence

from src.common.processor import DocumentProcessor
from src.common.embedding_generator import EmbeddingGenerator
from src.common.vector_store import VectorStore

# --------------------------------------------------------------------------- #
# Initialize shared services
# --------------------------------------------------------------------------- #
processor = DocumentProcessor()
embedder = EmbeddingGenerator()
vector_store = VectorStore()


def update_index_from_path(path: str | Path) -> None:
    """Ingest files under *path*, embed them, and upsert into the vector store."""
    processed = processor.process_document(Path(path))
    embedded = embedder.generate_embeddings(processed)
    vector_store.upsert_vectors(embedded)


def remove_vectors(ids: Sequence[str]) -> None:
    """Delete vectors with the given *ids* from the vector store."""
    vector_store.delete_vectors(list(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage your document vector index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # `update` sub-command
    update_parser = subparsers.add_parser(
        "update", help="Ingest documents and update the index."
    )
    update_parser.add_argument(
        "--path",
        "-p",
        required=True,
        metavar="DIR",
        help="Root directory containing documents to ingest.",
    )

    # `remove` sub-command
    remove_parser = subparsers.add_parser(
        "remove", help="Remove vectors from the index."
    )
    remove_parser.add_argument(
        "--ids",
        "-i",
        required=True,
        nargs="+",
        metavar="ID",
        help="One or more vector IDs to delete.",
    )

    args = parser.parse_args()

    if args.command == "update":
        update_index_from_path(args.path)
    elif args.command == "remove":
        remove_vectors(args.ids)


if __name__ == "__main__":
    main()
