from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from langchain_core.documents import Document

from subjectmate.documents import (
    deduplicate_chunks,
    load_course_documents,
    merge_short_sections,
    split_page_documents,
)


def _make_zip(files: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_chunks_keep_pdf_and_page_metadata():
    page = Document(
        page_content="A short definition. " * 20,
        metadata={"source": "Lecture.pdf", "page": 4, "source_id": "Lecture.pdf:p4"},
    )

    chunks = split_page_documents([page], chunk_size=80, chunk_overlap=12)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "Lecture.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 4 for chunk in chunks)
    assert all(chunk.metadata["source_id"] == "Lecture.pdf:p4" for chunk in chunks)
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)


def test_extracts_office_files_from_nested_archives(tmp_path):
    word_xml = b"""<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Word course material</w:t></w:r></w:p></w:body></w:document>"""
    slide_xml = b"""<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide course material</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"""
    workbook_xml = b"""<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Results" sheetId="1" r:id="rId1"/></sheets></workbook>"""
    relationships_xml = b"""<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="worksheet"/></Relationships>"""
    shared_strings_xml = b"""<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>Spreadsheet course material</t></si></sst>"""
    worksheet_xml = b"""<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>row value</t></is></c></row></sheetData></worksheet>"""
    docx = _make_zip({"word/document.xml": word_xml})
    nested_zip = _make_zip({"notes.docx": docx})
    pptx = _make_zip({"ppt/slides/slide1.xml": slide_xml})
    xlsx = _make_zip(
        {
            "xl/workbook.xml": workbook_xml,
            "xl/_rels/workbook.xml.rels": relationships_xml,
            "xl/sharedStrings.xml": shared_strings_xml,
            "xl/worksheets/sheet1.xml": worksheet_xml,
        }
    )
    course_zip = _make_zip(
        {
            "notes.docx": docx,
            "lecture.pptx": pptx,
            "scores.xlsx": xlsx,
            "nested.zip": nested_zip,
        }
    )
    material_dir = tmp_path / "materials"
    material_dir.mkdir()
    (material_dir / "sample_resources.zip").write_bytes(course_zip)

    documents = load_course_documents([material_dir])
    by_source = {document.metadata["source"]: document for document in documents}
    nested_document = next(
        document
        for document in documents
        if "nested.zip!/notes.docx" in document.metadata["source_path"]
    )

    assert "Word course material" in by_source["notes.docx"].page_content
    assert "Slide course material" in by_source["lecture.pptx"].page_content
    assert "Spreadsheet course material" in by_source["scores.xlsx"].page_content
    assert by_source["lecture.pptx"].metadata["location_type"] == "slide"
    assert by_source["lecture.pptx"].metadata["location"] == "1"
    assert by_source["scores.xlsx"].metadata["location"] == "Results"
    assert nested_document.metadata["subject"] == "sample"


def test_loose_text_files_take_their_folder_as_subject(tmp_path):
    folder = tmp_path / "materials" / "python programming"
    folder.mkdir(parents=True)
    (folder / "Tutorial.txt").write_text("Lists are mutable sequences.", encoding="utf-8")

    [document] = load_course_documents([tmp_path / "materials"])

    assert document.page_content == "Lists are mutable sequences."
    assert document.metadata["subject"] == "python programming"
    assert document.metadata["location_type"] == "document"


def _page(source_path: str, number: int, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source_path.rsplit("/", 1)[-1],
            "source_path": source_path,
            "source_id": f"{source_path}:page:{number}",
            "location_type": "page",
            "location": str(number),
            "page": number,
        },
    )


def test_short_pages_merge_forward_within_the_same_file():
    body = "Generative AI systems create new content from learned patterns. " * 5
    documents = [
        _page("a.pdf", 1, "Part A"),
        _page("a.pdf", 2, "Foundations"),
        _page("a.pdf", 3, body),
        _page("a.pdf", 4, "End"),
        _page("b.pdf", 1, body),
    ]

    merged = merge_short_sections(documents, min_chars=50)

    assert [document.metadata["location"] for document in merged] == ["1-3", "4", "1"]
    assert merged[0].page_content.startswith("Part A\n\nFoundations\n\n")
    assert merged[0].metadata["page"] == 1
    assert merged[0].metadata["source_id"] == "a.pdf:page:1-3"
    assert merged[1].metadata["source_path"] == "a.pdf"


def test_duplicate_and_wordless_chunks_are_dropped():
    text = "A data warehouse stores integrated data under a unified schema."
    chunks = [
        Document(page_content=text, metadata={}),
        Document(page_content="  " + text.upper() + "\n", metadata={}),
        Document(page_content="16 / 14 / 7 3 / 1", metadata={}),
    ]

    assert deduplicate_chunks(chunks) == [chunks[0]]
