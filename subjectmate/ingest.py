"""Embed course files and ZIP archives into the Qdrant knowledge base."""

import argparse

from subjectmate.settings import Settings
from subjectmate.vectorstore import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-embed everything instead of only files added since the last build",
    )
    args = parser.parse_args()
    settings = Settings()
    manifest = build_index(
        settings, progress=lambda message: print(message, flush=True), full=args.full
    )
    print(
        f"Index holds {manifest['document_count']} sections as "
        f"{manifest['chunk_count']} chunks in Qdrant collection "
        f"'{settings.qdrant_collection}' at {settings.qdrant_url}"
    )


if __name__ == "__main__":
    main()
