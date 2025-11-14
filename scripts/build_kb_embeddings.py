import os
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    text = text or ""
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + chunk_size)
        chunks.append(text[i:end])
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return chunks


def build_embeddings_for_texts(texts: List[str], model: str, api_key: str) -> List[List[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    # Batch requests in reasonable chunks to avoid payload limits
    embeddings: List[List[float]] = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        for d in resp.data:
            embeddings.append(d.embedding)
    return embeddings


def main():
    load_dotenv()
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    kb_dir = data_dir / "kb"
    index_path = data_dir / "kb_index.json"
    out_path = data_dir / "kb_embeddings.json"

    api_key = os.getenv("OPENAI_API_KEY")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set in environment.")
        return

    kb_index = load_json(index_path) or []
    if not kb_index:
        print("No KB index found; nothing to embed.")
        return

    records: List[Dict[str, Any]] = []
    for entry in kb_index:
        rel = entry.get("path") or ""
        if not rel:
            continue
        src_path = data_dir / rel
        if not src_path.exists():
            print(f"Skip missing KB file: {src_path}")
            continue
        # Prefer full text extraction if available; fall back to snippet
        try:
            text = src_path.read_text(encoding="utf-8")
        except Exception:
            text = entry.get("snippet") or ""
        chunks = chunk_text(text)
        for idx, ch in enumerate(chunks):
            records.append({
                "id": f"{entry.get('id')}_{idx}",
                "title": entry.get("title"),
                "path": rel,
                "chunk_index": idx,
                "text": ch,
            })

    if not records:
        print("No KB text chunks created.")
        return

    print(f"Embedding {len(records)} KB chunks…")
    texts = [r["text"] for r in records]
    vectors = build_embeddings_for_texts(texts, embed_model, api_key)
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out_items = []
    for r, v in zip(records, vectors):
        item = dict(r)
        item["embedding"] = v
        item["created_at"] = now
        out_items.append(item)
    save_json(out_path, {"model": embed_model, "count": len(out_items), "items": out_items})
    print(f"Saved embeddings → {out_path}")


if __name__ == "__main__":
    main()