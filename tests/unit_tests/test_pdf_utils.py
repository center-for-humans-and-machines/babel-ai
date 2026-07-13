import os
import tempfile
from typing import List

from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from utils.pdf_utils import (
    clamp_requested_pages,
    extract_pdf_selection,
    normalize_unique_order,
)


def _make_pdf_with_text(pages: List[str], path: str) -> None:
    """
    Create a simple PDF where each page contains the provided text.
    """
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    for text in pages:
        c.setFont("Helvetica", 12)
        c.drawString(72, height - 72, text)
        c.showPage()
    c.save()


def test_normalize_unique_order_preserves_first_occurrence():
    nums = [3, 4, 3, 2, 4, 1, 1, 5]
    assert normalize_unique_order(nums) == [3, 4, 2, 1, 5]


def test_clamp_requested_pages_filters_and_orders():
    total = 10
    req = [0, 1, 2, 11, 10, 5, 5]
    assert clamp_requested_pages(req, total) == [1, 2, 10, 5]


def test_extract_pdf_selection_combines_fixed_and_search_pages():
    # Build a 12-page PDF with distinct content per page
    pages = [f"Page {i} content" for i in range(1, 13)]
    # Add search terms on pages 6, 11, 12 (1-based)
    pages[5] = "This has Iteration 42 marker"
    pages[10] = "Note: Iteration 43 appears here"
    pages[11] = "Ending with Iteration 44 marker"

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src.pdf")
        out = os.path.join(td, "out.pdf")
        _make_pdf_with_text(pages, src)

        fixed = [3, 4, 8, 9, 10]
        terms = ["Iteration 42", "Iteration 43", "Iteration 44"]

        selected = extract_pdf_selection(src, fixed, terms, out)

        # Expect fixed pages + pages 6, 11, 12 in order
        assert selected == [3, 4, 8, 9, 10, 6, 11, 12]

        # Verify output file pages count matches selection
        reader = PdfReader(out)
        assert len(reader.pages) == len(selected)
