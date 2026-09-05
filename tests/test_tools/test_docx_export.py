"""DocxExportTool: path sandboxing, filename safety, and a real .docx write."""

from __future__ import annotations

import pytest

from src.tools.builtin.docx_export import DocxExportTool, _safe_filename


def test_safe_filename_strips_directory_components():
    assert _safe_filename("../../evil.docx", "V-101") == "evil.docx"
    assert _safe_filename("C:\\Windows\\evil.docx", "V-101") == "evil.docx"


def test_safe_filename_adds_extension_when_missing():
    assert _safe_filename("approval_note", "V-101").endswith(".docx")


def test_safe_filename_generates_one_when_blank():
    name = _safe_filename(None, "V-101")
    assert name.startswith("approval_note_V-101_")
    assert name.endswith(".docx")


async def test_export_writes_a_real_docx(tmp_path):
    pytest.importorskip("docx")
    tool = DocxExportTool(root=tmp_path)
    result = await tool.run(
        title="V-101 Inspection Approval Note",
        equipment_id="V-101",
        summary="Observed pressure exceeds the approved operating limit.",
        findings=[
            "Observed pressure 14.5 bar vs. approved limit 12.0 bar.",
            "Last inspection recorded no corrosion.",
        ],
        recommendation="Schedule a follow-up inspection before continued operation.",
    )
    assert result.ok is True
    path = tmp_path / result.output["filename"]
    assert path.is_file()
    assert path.stat().st_size > 0

    from docx import Document
    doc = Document(str(path))
    full_text = "\n".join(p.text for p in doc.paragraphs)
    assert "V-101 Inspection Approval Note" in full_text
    assert "14.5 bar" in full_text
    assert "follow-up inspection" in full_text
    # sign-off table present
    assert len(doc.tables) == 1
    assert doc.tables[0].rows[0].cells[0].text == "Prepared by"


async def test_output_filename_cannot_escape_root(tmp_path):
    pytest.importorskip("docx")
    tool = DocxExportTool(root=tmp_path)
    # _safe_filename already strips directory components before the
    # containment check runs, so this proves the escape is blocked, not
    # just that the traversal string happens to look different afterwards.
    result = await tool.run(title="x", output_filename="../outside.docx")
    assert result.ok is True
    assert (tmp_path / "outside.docx").is_file()
    assert not (tmp_path.parent / "outside.docx").is_file()
