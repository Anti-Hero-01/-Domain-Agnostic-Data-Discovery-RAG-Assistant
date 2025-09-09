import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
from core.config import Settings

class VectorStore:
    def __init__(self):
        settings = Settings()
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.vector_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.index = self._load_or_create_index()
        self.documents = []

    def _load_or_create_index(self):
        settings = Settings()
        index_path = f"{settings.VECTOR_DB_PATH}/faiss.index"
        
        if os.path.exists(index_path):
            return faiss.read_index(index_path)
        
        return faiss.IndexFlatL2(self.vector_dim)

    def add_document(self, content: str, chunk_size: int = 512):
        # Split content into chunks and embed
        chunks = self._chunk_text(content, chunk_size)
        embeddings = self.model.encode(chunks)
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(chunks)

    def search(self, query: str, k: int = 5) -> List[Dict[str, float]]:
        query_vector = self.model.encode([query])[0].reshape(1, -1)
        distances, indices = self.index.search(query_vector.astype(np.float32), k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append({
                    "content": self.documents[idx],
                    "score": float(1 / (1 + dist))
                })
        
        return results

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1
            if current_size > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks