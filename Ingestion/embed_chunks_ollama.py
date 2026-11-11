"""Embed heading-based chunk JSON/JSONL files with a local Ollama model.

Default:
    python Ingestion/embed_chunks_ollama.py

The default input is the JSONL created during chunk conversion. Each output
line preserves the chunk fields and adds ``embedding`` and ``embedding_model``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "source" / "chunks" / "Walking Tour Mobile Friendly RD final.chunks.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "source" / "embeddings" / "Walking Tour Mobile Friendly RD final.embeddings.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed chunk records using Ollama's local API.")
    parser.add_argument("input_file", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output_file", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="nomic-embed-text", help="Ollama embedding model")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--timeout", type=float, default=120, help="Per-request timeout in seconds")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError("Input JSON must contain an array of chunk objects.")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Every input record must be a JSON object.")
    return records


def embed(text: str, model: str, host: str, timeout: float) -> list[float]:
    payload = json.dumps({"model": model, "input": text}).encode("utf-8")
    request = Request(
        f"{host.rstrip('/')}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    vectors = result.get("embeddings")
    if not vectors or not vectors[0]:
        raise ValueError("Ollama returned no embedding vector.")
    return vectors[0]


def main() -> int:
    args = parse_args()
    source = args.input_file.resolve()
    destination = args.output_file.resolve()
    if not source.is_file():
        print(f"Error: input file not found: {source}", file=sys.stderr)
        return 1

    try:
        records = load_records(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with destination.open("w", encoding="utf-8", newline="\n") as output:
            for line_number, record in enumerate(records, start=1):
                text = record.get("content") or record.get("markdown") or ""
                if not isinstance(text, str) or not text.strip():
                    print(f"Skipping record {line_number}: no nonempty content/markdown text.", file=sys.stderr)
                    continue
                try:
                    vector = embed(text, args.model, args.host, args.timeout)
                except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                    raise RuntimeError(f"Could not embed record {line_number}: {exc}") from exc
                output_record = dict(record)
                output_record["embedding"] = vector
                output_record["embedding_model"] = args.model
                output.write(json.dumps(output_record, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
                print(f"Embedded {count}/{len(records)}: {record.get('id', line_number)}")
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {count} embedded records to: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
