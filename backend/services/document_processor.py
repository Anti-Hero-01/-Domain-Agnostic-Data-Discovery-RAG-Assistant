import pandas as pd
import pdfplumber
from typing import Dict, Any
import spacy
from services.knowledge_graph import KnowledgeGraph
from services.vector_store import VectorStore

class DocumentProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.kg = KnowledgeGraph()
        self.vector_store = VectorStore()

    async def process_file(self, file) -> Dict[str, Any]:
        content = await self._extract_content(file)
        entities = self._extract_entities(content)
        
        # Store in knowledge graph
        self.kg.add_entities(entities)
        
        # Store embeddings
        self.vector_store.add_document(content)
        
        return {
            "filename": file.filename,
            "entities": entities,
            "status": "processed"
        }

    async def _extract_content(self, file) -> str:
        ext = file.filename.split(".")[-1].lower()
        content = ""
        
        if ext == "pdf":
            with pdfplumber.open(file.file) as pdf:
                for page in pdf.pages:
                    content += page.extract_text()
        elif ext == "txt":
            content = await file.read()
            content = content.decode("utf-8")
        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(file.file)
            content = df.to_string()
        
        return content

    def _extract_entities(self, text: str) -> Dict[str, list]:
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        return entities