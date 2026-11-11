"""Upsert locally generated chunk embeddings into the Pinecone index in .env.

Run from the project root:
    python Ingestion/upsert_embeddings_pinecone.py

Input is JSONL with each record containing an ``id`` and ``embedding`` field.
Records are translated to Pinecone Documents API documents: ``id`` becomes
``_id``, ``embedding`` becomes the schema's dense vector field, and remaining
fields are stored alongside the vector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "source" / "embeddings" / "Walking Tour Mobile Friendly RD final.embeddings.jsonl"
ENV_FILE = PROJECT_ROOT / ".env"


def read_env(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without echoing their values."""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip().strip('"').strip("'")
    return values


def metadata_value(value: object, field: str) -> object:
    """Validate Pinecone-compatible metadata values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(
        f"Metadata field {field!r} must be a string, number, boolean, or list of strings."
    )


def read_documents(path: Path, vector_field: str) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object.")
            vector_id = record.get("id")
            embedding = record.get("embedding")
            if not isinstance(vector_id, str) or not vector_id:
                raise ValueError(f"Line {line_number} is missing a nonempty string 'id'.")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(f"Line {line_number} is missing an 'embedding' array.")
            if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in embedding):
                raise ValueError(f"Line {line_number} has a non-numeric embedding value.")

            document = {"_id": vector_id, vector_field: embedding}
            document.update({
                key: metadata_value(value, key)
                for key, value in record.items()
                if key not in {"id", "embedding"} and value is not None
            })
            documents.append(document)
    if not documents:
        raise ValueError("Input contains no documents to upsert.")
    return documents


def upsert_batch(
    host: str,
    api_key: str,
    namespace: str,
    documents: list[dict[str, object]],
    timeout: float,
) -> None:
    payload = json.dumps({"documents": documents}).encode("utf-8")
    namespace_path = quote(namespace, safe="")
    request = Request(
        f"{host.rstrip('/')}/namespaces/{namespace_path}/documents/upsert",
        data=payload,
        headers={
            "Api-Key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Pinecone-API-Version": "2026-07",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    upserted = result.get("upsertedCount")
    if upserted is not None and upserted != len(documents):
        raise RuntimeError(f"Pinecone acknowledged {upserted} of {len(documents)} documents in a batch.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upsert embedded chunk records to Pinecone.")
    parser.add_argument("input_file", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--vector-field",
        help="Dense vector field from the index schema (default: Pinecone_Vector_Field or 'dense').",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input_file if args.input_file.is_absolute() else PROJECT_ROOT / args.input_file
    if args.batch_size < 1:
        print("Error: --batch-size must be at least 1.", file=sys.stderr)
        return 1
    try:
        config = read_env(ENV_FILE)
        api_key = config.get("pinecone_api_key", "")
        host = config.get("pinecone_host", "")
        namespace = config.get("pinecone_namespace", "")
        vector_field = args.vector_field or config.get("pinecone_vector_field", "dense")
        if not api_key or not host:
            raise ValueError(".env must define Pinecone_API_Key and Pinecone_Host.")
        if not namespace:
            raise ValueError(".env must define a nonempty Pinecone_Namespace for the Documents API.")
        if not source.is_file():
            raise ValueError(f"Input JSONL file not found: {source}")
        documents = read_documents(source, vector_field)

        # The index host determines the target index; the name is used for a clear run summary.
        index_name = config.get("pinecone_index_name", "(host configured in .env)")
        for start in range(0, len(documents), args.batch_size):
            batch = documents[start : start + args.batch_size]
            upsert_batch(host, api_key, namespace, batch, args.timeout)
            print(f"Upserted {min(start + len(batch), len(documents))}/{len(documents)} documents")
        print(
            f"Finished: {len(documents)} documents sent to index {index_name!r}, "
            f"namespace {namespace!r}, vector field {vector_field!r}."
        )
    except HTTPError as exc:
        # Pinecone's response body explains validation failures (for example,
        # dimension mismatch) and does not contain the request's API key.
        detail = exc.read(2000).decode("utf-8", "replace").strip()
        print(f"Pinecone rejected the upsert: HTTP {exc.code} {exc.reason}", file=sys.stderr)
        if detail:
            print(f"Pinecone response: {detail}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"Could not reach Pinecone: {type(exc.reason).__name__}", file=sys.stderr)
        return 1
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
