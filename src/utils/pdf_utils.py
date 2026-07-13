"""
Utility functions for extracting selected pages from PDF files.

This module provides small, functional helpers to:
- extract specific 1-based pages
- locate pages containing target substrings (e.g., 'Iteration 42')
- write a new PDF with the selected pages in order (deduplicated)

All functions are pure (no I/O) except `write_pages_to_pdf`.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set

from pypdf import PdfReader, PdfWriter


def normalize_unique_order(numbers: Sequence[int]) -> List[int]:
    """
    Return items preserving order while removing duplicates.
    """
    seen: Set[int] = set()
    ordered: List[int] = []
    for num in numbers:
        if num not in seen:
            seen.add(num)
            ordered.append(num)
    return ordered


def clamp_requested_pages(
    requested_pages_1based: Iterable[int], total_pages: int
) -> List[int]:
    """
    Keep only valid 1-based page indices within [1, total_pages].
    Preserve original order and uniqueness.
    """
    filtered = [
        p for p in requested_pages_1based if 1 <= int(p) <= int(total_pages)
    ]
    return normalize_unique_order(filtered)


def find_pages_with_terms(
    reader: PdfReader, terms: Iterable[str]
) -> List[int]:
    """
    Return 1-based pages whose extracted text contains any of the terms.
    """
    lower_terms = [t.lower() for t in terms]
    matched_pages: List[int] = []

    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        lower_text = text.lower()
        if any(term in lower_text for term in lower_terms):
            matched_pages.append(idx)

    return matched_pages


def write_pages_to_pdf(
    reader: PdfReader, target_pages_1based: Sequence[int], output_path: str
) -> str:
    """
    Write selected pages (1-based) to a new PDF at `output_path`.
    Returns the output path.
    """
    writer = PdfWriter()
    total = len(reader.pages)
    for p in target_pages_1based:
        if 1 <= p <= total:
            # pypdf uses 0-based indexing for pages
            writer.add_page(reader.pages[p - 1])
    with open(output_path, "wb") as f:
        writer.write(f)
    return output_path


def extract_pdf_selection(
    source_path: str,
    fixed_pages_1based: Iterable[int],
    search_terms: Iterable[str],
    output_path: str,
) -> List[int]:
    """
    Extract a combined selection of pages and write to `output_path`.

    Returns the final 1-based page list included in the output.
    """
    reader = PdfReader(source_path)
    total_pages = len(reader.pages)

    requested_pages = list(fixed_pages_1based)
    requested_pages = clamp_requested_pages(requested_pages, total_pages)

    found_pages = find_pages_with_terms(reader, search_terms)
    combined = normalize_unique_order([*requested_pages, *found_pages])

    write_pages_to_pdf(reader, combined, output_path)
    return combined
