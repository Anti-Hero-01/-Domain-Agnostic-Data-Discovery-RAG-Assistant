# backend/embeddings.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

MODEL_NAME = 'all-MiniLM-L6-v2'  # small and fast
model = SentenceTransformer(MODEL_NAME)

INDEX_FILE = 'faiss.index'
META_FILE = 'index_meta.json'

# metadata map: id -> {doc_id, chunk_text, file_name, offset}
if os.path.exists(META_FILE):
    with open(META_FILE, 'r', encoding='utf-8') as f:
        meta = json.load(f)
else:
    meta = {}

# dimension
DIM = model.get_sentence_embedding_dimension()

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatIP(DIM)  # inner product; will normalize

def _embed(texts):
    emb = model.encode(texts, convert_to_numpy=True)
    # normalize for cosine similarity
    faiss.normalize_L2(emb)
    return emb

def index_chunks(doc_id, chunks):
    if not chunks:
        return
    emb = _embed(chunks)
    start = index.ntotal
    index.add(emb)
    # update meta
    for i, c in enumerate(chunks):
        meta[str(start + i)] = {'doc_id': doc_id, 'chunk': c}
    # persist
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def query_vectorstore(query, top_k=5):
    if index.ntotal == 0:
        return []
    q_emb = _embed([query])
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append({'score': float(score), 'id': int(idx), 'meta': meta.get(str(idx)), 'chunk': meta.get(str(idx))['chunk']})
    return results

def generate_answer(question, top_chunks):
    # Build a simple prompt and call OpenAI if API key present; otherwise return concatenated snippets
    ctx = '\n\n'.join([f"Source: {c['meta']['doc_id']}\nText:\n{c['chunk']}" for c in top_chunks])
    prompt = f"You are a helpful assistant. Use the following context to answer the question. Include a short answer and give citations in square brackets referencing the doc_id.\n\nCONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nAnswer concisely, and if you don't know say you don't know."

    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key
            res = openai.ChatCompletion.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400
            )
            return res['choices'][0]['message']['content']
        except Exception as e:
            return f"OpenAI call failed: {e}\n\nContext used:\n{ctx[:2000]}"
    else:
        # fallback: return top snippets (shortened)
        return "\n\n".join([c['chunk'][:800] for c in top_chunks])
