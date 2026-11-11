"""Retrieve Pinecone chunks, ask Groq to answer from them, and print both."""

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
SIMILARITY_THRESHOLD = 0.6
OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


def read_env():
    """Read the simple KEY=VALUE settings from the project .env file."""
    settings = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            settings[key.strip().lower()] = value.strip().strip('"').strip("'")
    return settings


def get_embedding(question):
    """Get one embedding vector from the local Ollama server."""
    body = json.dumps({"model": EMBEDDING_MODEL, "input": question}).encode("utf-8")
    request = Request(
        f"{OLLAMA_HOST}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)["embeddings"][0]


def search_pinecone(settings, embedding):
    """Search the index's `dense` vector field and return its three best matches."""
    host = settings["pinecone_host"].rstrip("/")
    namespace = quote(settings["pinecone_namespace"], safe="")
    url = f"{host}/namespaces/{namespace}/documents/search"
    body = json.dumps({
        "score_by": [{"type": "dense_vector", "field": "dense", "values": embedding}],
        "top_k": 3,
        "include_fields": ["heading", "content", "heading_path"],
    }).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Api-Key": settings["pinecone_api_key"],
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": "2026-07",
        },
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response).get("matches", [])


def generate_grounded_answer(question, matches, settings):
    """Ask the configured Groq model to answer using only retrieved chunks."""
    chunks = []
    for rank, match in enumerate(matches, start=1):
        chunks.append(
            f"[Chunk {rank}] Heading: {match.get('heading', '(no heading)')}\n"
            f"Heading path: {match.get('heading_path', '')}\n"
            f"Content:\n{match.get('content', '')}"
        )
    prompt = (
        "Answer the user's question using only the reference chunks below. "
        "If the chunks do not contain enough information, say so clearly. "
        "Do not add facts from outside the chunks. Cite supporting chunk numbers.\n\n"
        f"Question: {question}\n\nReference chunks:\n" + "\n\n".join(chunks)
    )
    model = settings["model_name"]
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": False,
    }).encode("utf-8")
    request = Request(
        GROQ_CHAT_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {settings['groq_api_key']}",
            "Content-Type": "application/json",
            "User-Agent": "rag-demo-pinecone/1.0",
        },
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    choices = result.get("choices", [])
    answer = choices[0].get("message", {}).get("content", "").strip() if choices else ""
    if not answer:
        raise ValueError(f"Groq model {model!r} returned an empty answer.")
    return answer


def main():
    question = input("Enter your question: ").strip()
    if not question:
        print("Please enter a question.")
        return

    try:
        settings = read_env()
        embedding = get_embedding(question)
        matches = search_pinecone(settings, embedding)

        matches = [
            match for match in matches
            if match.get("_score", 0) > SIMILARITY_THRESHOLD
        ]
        if not matches:
            print(f"No matches scored above {SIMILARITY_THRESHOLD:.1f}.")
            return

        answer = generate_grounded_answer(question, matches, settings)
        print(f"\nGrounded answer ({settings['model_name']} via Groq):\n")
        print(answer)
        print(f"\nRetrieved chunks above {SIMILARITY_THRESHOLD:.1f}:\n")
        for rank, match in enumerate(matches, start=1):
            heading = match.get("heading", "(no heading)")
            content = match.get("content", "")
            print(f"{rank}. {heading}")
            print(f"   Similarity score: {match.get('_score', 'n/a')}")
            if content:
                print(f"   {content}\n")
    except HTTPError as error:
        request_host = error.geturl().split("/", 3)[2]
        print(f"Request to {request_host} failed: HTTP {error.code} {error.reason}")
        detail = error.read(1500).decode("utf-8", "replace")
        if detail:
            print(detail)
    except URLError as error:
        print(f"Could not connect: {type(error.reason).__name__}")
    except (OSError, KeyError, IndexError, json.JSONDecodeError, ValueError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
