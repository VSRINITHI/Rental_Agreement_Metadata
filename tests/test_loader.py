import docx

from src.loader import load_docx, strip_layout_tags


def test_docx_keeps_document_order_and_tables(tmp_path):
    d = docx.Document()
    d.add_paragraph("before table")
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text = "Rent"
    t.rows[0].cells[1].text = "Rs. 9000"
    d.add_paragraph("after table")
    p = tmp_path / "a.docx"
    d.save(p)
    assert load_docx(p).split("\n") == ["before table", "Rent | Rs. 9000", "after table"]


def test_strip_layout_tags_only_removes_layout_markers():
    raw = "<x_0.1><y_0.2>Rent<x_0.3><y_0.4><class_Text>is Rs.<x_1>9000 <u>note</u> <foo>"
    out = strip_layout_tags(raw)
    assert out == "Rent is Rs.9000 <u>note</u> <foo>"


def test_strip_layout_tags_separates_words():
    assert strip_layout_tags("Rent<x_0.1><y_0.2>Paid") == "Rent Paid"
