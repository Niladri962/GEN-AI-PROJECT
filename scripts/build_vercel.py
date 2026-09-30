"""Prepare `vercel_app/` for deployment.

Copies the dependency-light `subjectmate` modules the serverless API imports, the logo,
and pre-downloads the fastembed ONNX models into `vercel_app/models/` so cold starts
never download them. Re-run after changing those modules or the model settings.

Usage: python scripts/build_vercel.py
Then:  cd vercel_app && npx vercel --prod
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from subjectmate.settings import Settings  # noqa: E402

APP = ROOT / "vercel_app"
# The API needs only these; the rest of the package pulls in PyTorch and LangChain.
MODULES = ["__init__.py", "settings.py", "rag.py", "ranking.py", "cloud.py"]


def directory_mb(path: Path) -> float:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / 1e6


def main() -> None:
    package = APP / "subjectmate"
    shutil.rmtree(package, ignore_errors=True)
    package.mkdir(parents=True)
    for name in MODULES:
        shutil.copy2(ROOT / "subjectmate" / name, package / name)
    shutil.copy2(ROOT / "assets" / "logo.svg", APP / "public" / "logo.svg")
    print(f"Copied {len(MODULES)} modules and the logo")

    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    settings = Settings()
    models = APP / "models"
    TextEmbedding(settings.embedding_model, cache_dir=str(models))
    print(f"Bundled embedding model {settings.embedding_model}")
    if settings.reranker_model:
        TextCrossEncoder(settings.reranker_model, cache_dir=str(models))
        print(f"Bundled re-ranker {settings.reranker_model}")
    for lock in models.rglob("*.lock"):
        lock.unlink()
    print(f"vercel_app/ is {directory_mb(APP):.0f} MB before Python packages (limit 500 MB with them)")


if __name__ == "__main__":
    main()
