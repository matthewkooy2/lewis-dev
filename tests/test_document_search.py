from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.rag import DocumentSearch


UNSAFE_QUERY_STRINGS = [
    "../",
    "../../",
    "C:\\",
    "C:\\Users\\Example",
    "%USERPROFILE%",
    "Desktop",
    "Downloads",
    "Documents",
    "OneDrive",
    "AppData",
    ".ssh",
]


def test_searches_markdown_file(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "note.md").write_text(
        "Project Alpha has a local planning note.",
        encoding="utf-8",
    )

    results = DocumentSearch(docs).search("Alpha")

    assert results[0].path == "note.md"
    assert results[0].score == 1
    assert "Alpha" in results[0].snippet


def test_searches_text_file(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "meeting.txt").write_text(
        "The budget review mentions Project Beta twice. Beta is important.",
        encoding="utf-8",
    )

    results = DocumentSearch(docs).search("Beta")

    assert results[0].path == "meeting.txt"
    assert results[0].score == 2
    assert "Beta" in results[0].snippet


def test_searches_pdf_file(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    pdf_path = docs / "brief.pdf"
    write_simple_pdf(pdf_path, "Project Gamma PDF note")

    results = DocumentSearch(docs).search("Gamma")

    assert results[0].path == "brief.pdf"
    assert "Gamma" in results[0].snippet


def test_no_documents_folder_returns_no_results(tmp_path: Path) -> None:
    search = DocumentSearch(tmp_path / "missing-documents")

    assert search.list_files() == []
    assert search.search("anything") == []


def test_no_matching_results(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "note.md").write_text("Project Alpha only.", encoding="utf-8")

    results = DocumentSearch(docs).search("Zanzibar")

    assert results == []


def test_path_traversal_candidate_is_blocked(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("escapephrase", encoding="utf-8")
    search = DocumentSearch(docs)

    assert search._validated_supported_file(docs / ".." / "outside.md") is None
    assert search.search("escapephrase") == []


def test_unsafe_query_strings_do_not_search_outside_documents(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    outside = tmp_path / "Desktop" / "outside.md"
    outside.parent.mkdir()
    outside.write_text("escapephrase", encoding="utf-8")
    (docs / "inside.md").write_text("ordinary local note", encoding="utf-8")
    search = DocumentSearch(docs)

    for unsafe_query in UNSAFE_QUERY_STRINGS:
        assert search.search(f"{unsafe_query} escapephrase") == []


def test_hidden_and_secret_like_files_are_skipped(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    docs.mkdir()
    hidden_dir = docs / ".ssh"
    hidden_dir.mkdir()
    (hidden_dir / "note.md").write_text("escapephrase", encoding="utf-8")
    (docs / ".hidden.md").write_text("escapephrase", encoding="utf-8")
    (docs / "api_key_notes.md").write_text("escapephrase", encoding="utf-8")
    (docs / "safe.md").write_text("ordinary local note", encoding="utf-8")

    search = DocumentSearch(docs)

    assert search.list_files() == ["safe.md"]
    assert search.search("escapephrase") == []


def write_simple_pdf(path: Path, text: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    stream = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 50 250 Td ({escaped}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = stream
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica"),
                        }
                    )
                }
            )
        }
    )
    with path.open("wb") as handle:
        writer.write(handle)
