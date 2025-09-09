import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from core.config import Settings


class VectorStore:
    def __init__(self):
        settings = Settings()
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.vector_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.index = None
        self.documents: List[str] = []

        # Paths for persistence
        self.index_path = os.path.join(settings.VECTOR_DB_PATH, "faiss.index")
        self.docs_path = os.path.join(settings.VECTOR_DB_PATH, "documents.json")

        os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load index + documents if available, otherwise create new."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            if os.path.exists(self.docs_path):
                with open(self.docs_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.vector_dim)

    def save(self):
        """Persist FAISS index and documents to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.docs_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def add_document(self, content: str, chunk_size: int = 512):
        """Split content into chunks, embed them, and add to FAISS + memory store."""
        chunks = self._chunk_text(content, chunk_size)
        if not chunks:
            return

        embeddings = self.model.encode(
            chunks,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        self.index.add(np.array(embeddings, dtype=np.float32))
        self.documents.extend(chunks)
        self.save()  # ✅ auto-save after adding

    def search(self, query: str, k: int = 5) -> List[Dict[str, float]]:
        """Search for the top-k most similar chunks to the query."""
        if not self.documents:
            raise ValueError("No documents in vector store. Add documents before searching.")

        # prevent asking FAISS for more results than exist
        k = min(k, len(self.documents))

        query_vector = self.model.encode([query], convert_to_numpy=True)[0].reshape(1, -1)
        distances, indices = self.index.search(query_vector.astype(np.float32), k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.documents):
                results.append({
                    "content": self.documents[idx],
                    "score": float(1 / (1 + dist))  # similarity score
                })
        return results

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately `chunk_size` characters."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_size + word_len > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_len
            else:
                current_chunk.append(word)
                current_size += word_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
