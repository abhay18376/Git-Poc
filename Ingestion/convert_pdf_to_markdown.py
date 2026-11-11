"""Convert a PDF to Markdown with Docling.

Install Docling in the active environment:
    python -m pip install docling

Convert the project's sample PDF:
    python Ingestion/convert_pdf_to_markdown.py

Convert another PDF and choose the output path:
    python Ingestion/convert_pdf_to_markdown.py input.pdf output.md

When the output path is omitted, Markdown is written to ``source/<input name>.md``.
Docling may download its models the first time it processes a document.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "source" / "pdf" / "Walking Tour Mobile Friendly RD final.pdf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "source"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PDF file to Markdown using Docling.",
        epilog=(
            "Install Docling with 'python -m pip install docling'. "
            "Example: python Ingestion/convert_pdf_to_markdown.py"
        ),
    )
    parser.add_argument(
        "input_pdf",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"PDF to convert (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "output_md",
        nargs="?",
        type=Path,
        help="Markdown output path (default: source/<input filename>.md)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_pdf = args.input_pdf

    if not input_pdf.is_absolute():
        input_pdf = Path.cwd() / input_pdf
    input_pdf = input_pdf.resolve()

    if not input_pdf.is_file():
        print(f"Error: input PDF does not exist: {input_pdf}", file=sys.stderr)
        return 1

    output_md = args.output_md
    if output_md is None:
        output_md = DEFAULT_OUTPUT_DIR / f"{input_pdf.stem}.md"
    elif not output_md.is_absolute():
        output_md = Path.cwd() / output_md
    output_md = output_md.resolve()

    try:
        from docling.document_converter import DocumentConverter

        print(f"Converting: {input_pdf}")
        document = DocumentConverter().convert(input_pdf).document
        markdown = document.export_to_markdown()

        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
    except ImportError:
        print(
            "Error: Docling is not installed. Install it with: python -m pip install docling",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"Error: PDF conversion failed: {exc}", file=sys.stderr)
        return 1

    print(f"Markdown saved to: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
