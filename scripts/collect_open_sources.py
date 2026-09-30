"""Download openly licensed reference material into `all data/<subject>/`.

Sources (all permit reuse; attribution is written at the top of each text file):
- Python documentation (PSF License), from the official text archive
- scikit-learn user guide (BSD-3-Clause)
- PostgreSQL documentation: tutorial and core SQL chapters (PostgreSQL License)
- OpenStax Introductory Statistics 2e (CC BY 4.0)

Files that already exist are skipped, so the script can be re-run safely. Afterwards run
`python -m subjectmate.ingest` to embed only the new files.

Usage: python scripts/collect_open_sources.py [--only python,sklearn,postgres,openstax]
"""

import argparse
import io
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent / "all data"
USER_AGENT = "SubjectMate/1.0 (college course-notes RAG project; local educational use)"
DELAY_SECONDS = 0.4

PYTHON = "python programming"
DB = "databases & data warehouses"
PS = "mathematical foundations-i (probability & statistics)"
ML = "machine learning"


PYTHON_DOCS_ARCHIVE = "https://docs.python.org/3/archives/python-3.14-docs-text.zip"
PYTHON_LIBRARY_PAGES = {
    "functions", "stdtypes", "exceptions", "constants", "collections", "itertools",
    "functools", "operator", "re", "string", "datetime", "json", "csv", "math", "random",
    "statistics", "os", "pathlib", "typing", "dataclasses", "abc", "enum", "copy",
    "heapq", "bisect", "array", "unittest", "doctest", "timeit", "sqlite3",
}

SKLEARN_GUIDE = "https://scikit-learn.org/stable/user_guide.html"
POSTGRES_BASE = "https://www.postgresql.org/docs/current/"
POSTGRES_CHAPTERS = [
    "tutorial.html", "sql-syntax.html", "ddl.html", "dml.html", "queries.html",
    "indexes.html", "mvcc.html", "performance-tips.html",
]
OPENSTAX_STATISTICS = (
    "https://assets.openstax.org/oscms-prodcms/media/documents/"
    "introductory-statistics-2e_-_WEB.pdf"
)


class Collector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.written = 0
        self.skipped = 0
        self.failed: list[str] = []

    def get(self, url: str, delay: float = DELAY_SECONDS, **kwargs) -> requests.Response:
        """GET with a polite delay; on 429/503 wait (Retry-After if given) and retry."""
        for attempt in range(6):
            time.sleep(delay)
            response = self.session.get(url, timeout=120, **kwargs)
            if response.status_code not in {429, 503}:
                response.raise_for_status()
                return response
            retry_after = response.headers.get("Retry-After", "")
            wait = int(retry_after) if retry_after.isdigit() else 15 * 2**attempt
            print(f"    rate limited; waiting {wait}s", flush=True)
            time.sleep(wait)
        response.raise_for_status()
        return response

    def save_text(self, path: Path, title: str, source: str, license_name: str, body: str) -> None:
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if len(body) < 200:
            self.failed.append(f"{title} (too little text)")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"{title}\nSource: {source}\nLicense: {license_name}\n\n{body}\n", encoding="utf-8"
        )
        self.written += 1

    def exists(self, path: Path) -> bool:
        if path.exists():
            self.skipped += 1
            return True
        return False


def safe_name(title: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", title).strip(". ")[:150]


BLOCK_TAGS = [
    "p", "li", "pre", "tr", "dt", "dd", "div", "section", "table", "blockquote",
    "h1", "h2", "h3", "h4", "h5", "h6", "br",
]


def html_to_text(html: str, selector: str) -> str:
    """Readable text: paragraphs on their own lines, source line wrapping removed."""
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one(selector) or soup.body or soup
    for tag in root.select(
        "script, style, nav, footer, .headerlink, .id_link, .navheader, .navfooter"
    ):
        tag.decompose()
    for string in list(root.find_all(string=True)):
        if string.find_parent("pre") is None:
            string.replace_with(re.sub(r"\s+", " ", str(string)))
    for tag in root.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    lines = (line.strip() for line in root.get_text().splitlines())
    return "\n".join(line for line in lines if line)



def collect_python_docs(collector: Collector) -> None:
    archive = zipfile.ZipFile(io.BytesIO(collector.get(PYTHON_DOCS_ARCHIVE).content))
    for name in archive.namelist():
        parts = name.split("/")
        if len(parts) != 3 or not name.endswith(".txt"):
            continue
        section, stem = parts[1], parts[2][:-4]
        wanted = (
            section in {"tutorial", "howto"}
            or (section == "reference" and stem != "index")
            or (section == "faq" and stem in {"programming", "general", "design"})
            or (section == "library" and stem in PYTHON_LIBRARY_PAGES)
        )
        if not wanted or stem == "index":
            continue
        path = ROOT / PYTHON / "python-docs" / f"{section} - {stem}.txt"
        if collector.exists(path):
            continue
        url = f"https://docs.python.org/3/{section}/{stem}.html"
        collector.save_text(
            path, f"Python documentation: {section}/{stem}", url, "PSF License",
            archive.read(name).decode("utf-8", errors="replace"),
        )
    print("  Python docs: done", flush=True)


def collect_sklearn(collector: Collector) -> None:
    soup = BeautifulSoup(collector.get(SKLEARN_GUIDE).text, "html.parser")
    links = []
    for anchor in soup.select("article a[href]"):
        href = anchor["href"].split("#")[0]
        if href.startswith("modules/") and href.endswith(".html") and href not in links:
            links.append(href)
    for href in links:
        stem = href.rsplit("/", 1)[-1][:-5]
        path = ROOT / ML / "scikit-learn-guide" / f"{stem}.txt"
        if collector.exists(path):
            continue
        url = urljoin(SKLEARN_GUIDE, href)
        try:
            text = html_to_text(collector.get(url).text, "article")
            collector.save_text(path, f"scikit-learn user guide: {stem}", url, "BSD-3-Clause", text)
        except Exception as exc:
            collector.failed.append(f"scikit-learn: {stem} ({exc})")
    print(f"  scikit-learn: {len(links)} pages", flush=True)


def collect_postgres(collector: Collector) -> None:
    pages: list[str] = []
    for chapter in POSTGRES_CHAPTERS:
        pages.append(chapter)
        soup = BeautifulSoup(collector.get(POSTGRES_BASE + chapter).text, "html.parser")
        for anchor in soup.select("div.toc a[href]"):
            href = anchor["href"].split("#")[0]
            if href.endswith(".html") and "/" not in href and href not in pages:
                pages.append(href)
    for href in pages:
        stem = href[:-5]
        path = ROOT / DB / "postgresql-docs" / f"{stem}.txt"
        if collector.exists(path):
            continue
        url = POSTGRES_BASE + href
        try:
            text = html_to_text(collector.get(url).text, "#docContent")
            collector.save_text(path, f"PostgreSQL documentation: {stem}", url, "PostgreSQL License", text)
        except Exception as exc:
            collector.failed.append(f"PostgreSQL: {stem} ({exc})")
    print(f"  PostgreSQL: {len(pages)} pages", flush=True)


def collect_openstax(collector: Collector) -> None:
    path = ROOT / PS / "Introductory Statistics 2e - OpenStax.pdf"
    if collector.exists(path):
        return
    content = collector.get(OPENSTAX_STATISTICS).content
    if not content.startswith(b"%PDF"):
        collector.failed.append("OpenStax: download was not a PDF")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    collector.written += 1
    print("  OpenStax Introductory Statistics: done", flush=True)


SOURCES = {
    "python": collect_python_docs,
    "sklearn": collect_sklearn,
    "postgres": collect_postgres,
    "openstax": collect_openstax,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="Comma-separated subset of: " + ",".join(SOURCES))
    args = parser.parse_args()
    chosen = args.only.split(",") if args.only else list(SOURCES)
    collector = Collector()
    for name in chosen:
        print(f"Collecting {name}...", flush=True)
        try:
            SOURCES[name](collector)
        except Exception as exc:
            collector.failed.append(f"{name}: {exc}")
    print(f"\nWrote {collector.written} files, skipped {collector.skipped} existing.")
    if collector.failed:
        print(f"{len(collector.failed)} item(s) failed:")
        for item in collector.failed:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
