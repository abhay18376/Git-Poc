"""Split a Markdown document into contextual chunks based on its headings.

Create chunks from the project's converted walking-tour document:
    python Ingestion/chunk_markdown_by_section.py

Choose an input file and output file:
    python Ingestion/chunk_markdown_by_section.py source/document.md source/chunks/document.chunks.json

Each JSON chunk includes its heading hierarchy, plain section content, and the
section Markdown. This lets a RAG pipeline retrieve a section with its context.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "source" / "Walking Tour Mobile Friendly RD final.md"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "source" / "chunks"
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Markdown into heading-based JSON chunks for RAG.",
    )
    parser.add_argument(
        "input_md",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Markdown file to chunk (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "output_json",
        nargs="?",
        type=Path,
        help="JSON output path (default: source/chunks/<input name>.chunks.json)",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve a path relative to the directory where the command is run."""
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def make_chunk(
    chunk_id: int,
    heading: str,
    level: int,
    heading_path: list[str],
    lines: list[str],
) -> dict[str, object]:
    """Build one RAG-ready chunk from a heading and its section body."""
    content = "\n".join(lines).strip()
    markdown = f"{'#' * level} {heading}"
    if content:
        markdown = f"{markdown}\n\n{content}"

    return {
        "id": f"section-{chunk_id:03d}",
        "heading": heading,
        "level": level,
        "heading_path": heading_path,
        "content": content,
        "markdown": markdown,
    }


def chunk_markdown(markdown: str) -> list[dict[str, object]]:
    """Return one chunk per Markdown heading, retaining parent-heading context."""
    chunks: list[dict[str, object]] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading: str | None = None
    current_level: int | None = None
    current_lines: list[str] = []

    for line in markdown.splitlines():
        match = HEADING_PATTERN.match(line)
        if not match:
            current_lines.append(line)
            continue

        if current_heading is not None and current_level is not None:
            chunks.append(
                make_chunk(
                    len(chunks) + 1,
                    current_heading,
                    current_level,
                    [title for _, title in heading_stack],
                    current_lines,
                )
            )

        level = len(match.group(1))
        heading = match.group(2).strip()
        heading_stack = [(old_level, title) for old_level, title in heading_stack if old_level < level]
        heading_stack.append((level, heading))
        current_heading = heading
        current_level = level
        current_lines = []

    if current_heading is not None and current_level is not None:
        chunks.append(
            make_chunk(
                len(chunks) + 1,
                current_heading,
                current_level,
                [title for _, title in heading_stack],
                current_lines,
            )
        )

    return chunks


def main() -> int:
    args = parse_args()
    input_md = resolve_path(args.input_md)
    if not input_md.is_file():
        print(f"Error: input Markdown file does not exist: {input_md}", file=sys.stderr)
        return 1

    output_json = args.output_json
    if output_json is None:
        output_json = DEFAULT_OUTPUT_DIR / f"{input_md.stem}.chunks.json"
    output_json = resolve_path(output_json)

    try:
        chunks = chunk_markdown(input_md.read_text(encoding="utf-8"))
        if not chunks:
            print("Error: no Markdown headings were found to create chunks.", file=sys.stderr)
            return 1

        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"Error: could not create chunks: {exc}", file=sys.stderr)
        return 1

    print(f"Created {len(chunks)} section chunks: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
