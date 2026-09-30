"""Copy the local Qdrant collection (vectors + payloads) to a Qdrant Cloud cluster.

Nothing is re-embedded: points are read from the local server and written unchanged.
The target collection is replaced.

Usage:
  python scripts/upload_to_qdrant_cloud.py --url https://xxxx.cloud.qdrant.io:6333 --api-key <key>
The URL and key default to QDRANT_CLOUD_URL / QDRANT_CLOUD_API_KEY, then QDRANT_API_KEY.
"""

import argparse
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subjectmate.settings import Settings  # noqa: E402
from subjectmate.vectorstore import SUBJECT_KEY  # noqa: E402

BATCH = 256


def main() -> int:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.getenv("QDRANT_CLOUD_URL"))
    parser.add_argument(
        "--api-key", default=os.getenv("QDRANT_CLOUD_API_KEY") or os.getenv("QDRANT_API_KEY")
    )
    parser.add_argument("--source", default="http://localhost:6333", help="Local Qdrant URL")
    parser.add_argument("--collection", default=settings.qdrant_collection)
    args = parser.parse_args()
    if not args.url or not args.api_key:
        parser.error("Give --url and --api-key (or set QDRANT_CLOUD_URL and QDRANT_CLOUD_API_KEY)")

    source = QdrantClient(url=args.source, timeout=120)
    target = QdrantClient(url=args.url, api_key=args.api_key, timeout=300)
    info = source.get_collection(args.collection)
    total = source.count(args.collection, exact=True).count
    print(f"Copying {total:,} points of '{args.collection}' to {args.url}")

    if target.collection_exists(args.collection):
        target.delete_collection(args.collection)
    target.create_collection(args.collection, vectors_config=info.config.params.vectors)
    target.create_payload_index(args.collection, SUBJECT_KEY, models.PayloadSchemaType.KEYWORD)

    copied, offset = 0, None
    while True:
        points, offset = source.scroll(
            args.collection, limit=BATCH, offset=offset, with_payload=True, with_vectors=True
        )
        if points:
            target.upsert(
                args.collection,
                points=[
                    models.PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                    for point in points
                ],
                wait=True,
            )
            copied += len(points)
            print(f"  {copied:,}/{total:,}", flush=True)
        if offset is None:
            break

    uploaded = target.count(args.collection, exact=True).count
    print(f"Done: cloud collection holds {uploaded:,} points (local {total:,})")
    return 0 if uploaded == total else 1


if __name__ == "__main__":
    sys.exit(main())
