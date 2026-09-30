import hashlib
import io
import posixpath
import re
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Iterable
from xml.etree import ElementTree

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".zip"}
MAX_ARCHIVE_DEPTH = 8
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DRAWING_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
SPREADSHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
OFFICE_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def discover_source_files(source_dirs: Iterable[Path]) -> list[tuple[Path, str]]:
    """Find supported source files and return each with a stable display path."""
    discovered: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        paths = [source_dir] if source_dir.is_file() else sorted(source_dir.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            relative = path.name if source_dir.is_file() else path.relative_to(source_dir).as_posix()
            discovered.append((path, f"{source_dir.name}/{relative}"))
    return discovered


def _source_subject(relative_path: str, fallback: str) -> str:
    parts = PurePosixPath(relative_path).parts
    return parts[0] if len(parts) > 1 else fallback


def _base_metadata(
    source_name: str,
    source_path: str,
    subject: str,
    location_type: str,
    location: str,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "source": source_name,
        "source_path": source_path,
        "source_id": f"{source_path}:{location_type}:{location}",
        "subject": subject,
        "location_type": location_type,
        "location": location,
    }
    if location_type == "page":
        metadata["page"] = int(location)
    return metadata


def _add_document(
    documents: list[Document],
    text: str,
    source_name: str,
    source_path: str,
    subject: str,
    location_type: str,
    location: str,
) -> None:
    text = text.strip()
    if text:
        documents.append(
            Document(
                page_content=text,
                metadata=_base_metadata(
                    source_name, source_path, subject, location_type, location
                ),
            )
        )


def _read_payload(
    extension: str,
    payload: bytes,
    source_path: str,
    subject: str,
    documents: list[Document],
) -> None:
    source_name = PurePosixPath(source_path.split("!/")[-1]).name
    if extension == ".pdf":
        reader = PdfReader(io.BytesIO(payload))
        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            _add_document(
                documents,
                text,
                source_name,
                source_path,
                subject,
                "page",
                str(page_index),
            )
        return

    if extension in {".txt", ".md"}:
        _add_document(
            documents,
            payload.decode("utf-8", errors="replace"),
            source_name,
            source_path,
            subject,
            "document",
            "document",
        )
        return

    if extension == ".docx":
        with zipfile.ZipFile(io.BytesIO(payload)) as package:
            root = ElementTree.fromstring(package.read("word/document.xml"))
        content = [
            "".join(node.text or "" for node in paragraph.iter(f"{WORD_NS}t"))
            for paragraph in root.iter(f"{WORD_NS}p")
        ]
        _add_document(
            documents,
            "\n".join(content),
            source_name,
            source_path,
            subject,
            "document",
            "document",
        )
        return

    if extension == ".pptx":
        with zipfile.ZipFile(io.BytesIO(payload)) as package:
            slides = []
            for name in package.namelist():
                match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
                if match:
                    slides.append((int(match.group(1)), name))
            for slide_number, name in sorted(slides):
                root = ElementTree.fromstring(package.read(name))
                content = [
                    node.text or "" for node in root.iter(f"{DRAWING_NS}t")
                ]
                _add_document(
                    documents,
                    "\n".join(content),
                    source_name,
                    source_path,
                    subject,
                    "slide",
                    str(slide_number),
                )
        return

    if extension == ".xlsx":
        with zipfile.ZipFile(io.BytesIO(payload)) as package:
            shared_strings = _xlsx_shared_strings(package)
            workbook = ElementTree.fromstring(package.read("xl/workbook.xml"))
            relationships = ElementTree.fromstring(
                package.read("xl/_rels/workbook.xml.rels")
            )
            targets = {
                relation.attrib["Id"]: relation.attrib["Target"]
                for relation in relationships
                if "Id" in relation.attrib and "Target" in relation.attrib
            }
            for sheet in workbook.iter(f"{SPREADSHEET_NS}sheet"):
                relation_id = sheet.attrib.get(f"{OFFICE_REL_NS}id")
                target = targets.get(relation_id or "")
                if not target:
                    continue
                sheet_path = _xlsx_part_path(target)
                root = ElementTree.fromstring(package.read(sheet_path))
                rows = _xlsx_rows(root, shared_strings)
                _add_document(
                    documents,
                    "\n".join(rows),
                    source_name,
                    source_path,
                    subject,
                    "sheet",
                    sheet.attrib.get("name", sheet_path),
                )
        return

    if extension == ".zip":
        if not payload:
            return
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                _read_archive(archive, source_path, subject, documents, 1)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid ZIP archive: {source_path}") from exc


def _xlsx_shared_strings(package: zipfile.ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(package.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(node.text or "" for node in item.iter(f"{SPREADSHEET_NS}t"))
        for item in root.iter(f"{SPREADSHEET_NS}si")
    ]


def _xlsx_part_path(target: str) -> str:
    normalized = target.replace("\\", "/")
    if normalized.startswith("/"):
        return normalized.lstrip("/")
    return posixpath.normpath(posixpath.join("xl", normalized))


def _xlsx_rows(
    root: ElementTree.Element, shared_strings: list[str]
) -> list[str]:
    rows = []
    for row_index, row in enumerate(root.iter(f"{SPREADSHEET_NS}row"), start=1):
        values = []
        for cell in row.findall(f"{SPREADSHEET_NS}c"):
            cell_type = cell.attrib.get("t", "")
            value_node = cell.find(f"{SPREADSHEET_NS}v")
            if cell_type == "inlineStr":
                value = "".join(
                    node.text or "" for node in cell.iter(f"{SPREADSHEET_NS}t")
                )
            elif value_node is not None and value_node.text is not None:
                value = value_node.text
                if cell_type == "s":
                    value = shared_strings[int(value)]
            else:
                continue
            if value:
                values.append(f"{cell.attrib.get('r', '')}={value}")
        if values:
            row_number = row.attrib.get("r", str(row_index))
            rows.append(f"row {row_number}: " + " | ".join(values))
    return rows


def _read_archive(
    archive: zipfile.ZipFile,
    archive_path: str,
    subject: str,
    documents: list[Document],
    depth: int,
) -> None:
    if depth > MAX_ARCHIVE_DEPTH:
        raise ValueError(f"ZIP nesting exceeds {MAX_ARCHIVE_DEPTH} levels: {archive_path}")
    for entry in sorted(archive.infolist(), key=lambda item: item.filename.lower()):
        if entry.is_dir():
            continue
        entry_path = PurePosixPath(entry.filename)
        if "__MACOSX" in entry_path.parts or entry_path.name.startswith("._"):
            continue
        extension = entry_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue
        nested_subject = _source_subject(entry.filename, subject)
        member_path = f"{archive_path}!/{entry_path.as_posix()}"
        with archive.open(entry) as member:
            payload = member.read()
        if extension == ".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(payload)) as nested_archive:
                    _read_archive(
                        nested_archive,
                        member_path,
                        nested_subject,
                        documents,
                        depth + 1,
                    )
            except zipfile.BadZipFile as exc:
                raise ValueError(f"Invalid ZIP archive: {member_path}") from exc
        else:
            _read_payload(extension, payload, member_path, nested_subject, documents)


def _archive_subject(path: Path) -> str:
    subject = path.stem
    if subject.lower().endswith("_resources"):
        subject = subject[: -len("_resources")]
    return subject.replace("_", " ").strip()


def load_course_documents(
    source_dirs: Iterable[Path], only: set[str] | None = None
) -> list[Document]:
    """Extract text and source locations from course files and ZIP archives.

    `only` limits extraction to these display paths (as returned by discover_source_files).
    """
    source_dirs = list(source_dirs)
    documents: list[Document] = []
    source_files = discover_source_files(source_dirs)
    if only is not None:
        source_files = [item for item in source_files if item[1] in only]
    if not source_files:
        locations = ", ".join(str(path) for path in source_dirs)
        raise FileNotFoundError(f"No supported course files found in: {locations}")

    for path, display_path in source_files:
        subject = _archive_subject(path) if path.suffix.lower() == ".zip" else _source_subject(
            display_path.split("/", 1)[-1], "Uncategorized"
        )
        try:
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    _read_archive(archive, display_path, subject, documents, 1)
            else:
                _read_payload(
                    path.suffix.lower(), path.read_bytes(), display_path, subject, documents
                )
        except Exception as exc:
            raise ValueError(f"Unable to extract {display_path}: {exc}") from exc
    if not documents:
        raise ValueError("No searchable text found in the supported course files")
    return documents


def load_page_documents(pdf_dir: Path) -> list[Document]:
    """Backward-compatible wrapper for loading course files from one directory."""
    return load_course_documents([pdf_dir])


def merge_short_sections(documents: list[Document], min_chars: int = 200) -> list[Document]:
    """Merge short pages or slides into the next one from the same file.

    Title slides and bare examples are too thin to retrieve well on their own; attached to
    the following page or slide they keep their context. Merged sections cite the range.
    """
    merged: list[Document] = []
    pending: Document | None = None
    for document in documents:
        metadata = document.metadata
        mergeable = metadata.get("location_type") in {"page", "slide"}
        if pending is not None:
            same_file = (
                mergeable
                and pending.metadata["source_path"] == metadata["source_path"]
                and pending.metadata["location_type"] == metadata["location_type"]
            )
            if same_file:
                first = str(pending.metadata["location"]).split("-")[0]
                location = f"{first}-{metadata['location']}"
                document = Document(
                    page_content=f"{pending.page_content}\n\n{document.page_content}",
                    metadata={
                        **metadata,
                        "location": location,
                        "source_id": f"{metadata['source_path']}:{metadata['location_type']}:{location}",
                        **({"page": pending.metadata["page"]} if "page" in metadata else {}),
                    },
                )
            else:
                merged.append(pending)
            pending = None
        if mergeable and len(document.page_content) < min_chars:
            pending = document
        else:
            merged.append(document)
    if pending is not None:
        merged.append(pending)
    return merged


def deduplicate_chunks(chunks: list[Document], min_words: int = 3) -> list[Document]:
    """Drop repeated chunks (e.g. the same PDF shipped twice) and chunks with almost no words."""
    seen: set[str] = set()
    unique: list[Document] = []
    for chunk in chunks:
        text = re.sub(r"\s+", " ", chunk.page_content.lower()).strip()
        if text in seen or len(re.findall(r"[a-z]{2,}", text)) < min_words:
            continue
        seen.add(text)
        unique.append(chunk)
    return unique


def split_page_documents(
    pages: list[Document], chunk_size: int = 850, chunk_overlap: int = 120
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        start = int(chunk.metadata.get("start_index", 0))
        identity = "|".join(
            [
                str(chunk.metadata["source_id"]),
                str(start),
                hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest(),
            ]
        )
        chunk.metadata["chunk_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return chunks
