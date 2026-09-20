"""Build a FAISS index from the Daraz policy PDFs.

Usage: python ingest.py --data_dir daraz_knowledge_base --out_dir faiss_index
Department = the first subfolder under data_dir (returns, delivery, ...).
"""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(data_dir: Path, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    records = []
    pdfs = sorted(data_dir.rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs")

    for pdf in pdfs:
        rel = pdf.relative_to(data_dir)
        department = rel.parts[0] if len(rel.parts) > 1 else "general"
        try:
            reader = PdfReader(str(pdf))
            pages = [(i, (p.extract_text() or "").strip())
                     for i, p in enumerate(reader.pages, start=1)]
        except Exception as e:
            print(f"  ! Skipping {rel}: {e}")
            continue

        n_before = len(records)
        for page_no, text in pages:
            if not text:
                continue
            for chunk in splitter.split_text(text):
                records.append({
                    "id": len(records),
                    "department": department,
                    "source_file": pdf.name,
                    "page": page_no,
                    "text": chunk,
                })
        print(f"  {rel}: {len(records) - n_before} chunks")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="daraz_knowledge_base")
    ap.add_argument("--out_dir", default="faiss_index")
    ap.add_argument("--chunk_size", type=int, default=800)
    ap.add_argument("--chunk_overlap", type=int, default=150)
    args = ap.parse_args()

    records = load_chunks(Path(args.data_dir), args.chunk_size, args.chunk_overlap)
    if not records:
        raise SystemExit("No text extracted. Are the PDFs scanned images (need OCR)?")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [r["text"] for r in records],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via inner product
    ).astype("float32")

    index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
    ids = np.array([r["id"] for r in records], dtype="int64")
    index.add_with_ids(embeddings, ids)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "index.faiss"))
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(out / "config.json", "w") as f:
        json.dump({"model": MODEL_NAME, "dim": int(embeddings.shape[1])}, f)

    print(f"\nSaved {index.ntotal} vectors to {out}/")


if __name__ == "__main__":
    main()
