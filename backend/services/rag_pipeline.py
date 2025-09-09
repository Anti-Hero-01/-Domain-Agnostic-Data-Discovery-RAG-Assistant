from typing import Dict, List
from services.knowledge_graph import KnowledgeGraph
from services.vector_store import VectorStore
from transformers import pipeline
import openai
from core.config import Settings

class RAGPipeline:
    def __init__(self):
        settings = Settings()
        self.kg = KnowledgeGraph()
        self.vector_store = VectorStore()
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
    async def process_query(self, question: str, domain: str = None, role: str = None) -> Dict:
        # 1. Get relevant entities from knowledge graph
        kg_results = self.kg.query_subgraph(question)
        
        # 2. Get relevant documents from vector store
        vector_results = self.vector_store.search(question, k=3)
        
        # 3. Combine context
        context = self._combine_context(kg_results, vector_results)
        
        # 4. Generate answer using LLM
        response = await self._generate_answer(question, context, domain, role)
        
        return {
            "answer": response["answer"],
            "sources": response["sources"],
            "confidence": response["confidence"]
        }
    
    def _combine_context(self, kg_results: List[Dict], vector_results: List[Dict]) -> str:
        # Combine knowledge graph and vector store results
        context_parts = []
        
        # Add knowledge graph context
        if kg_results:
            kg_context = "Knowledge Graph Information:\n"
            for result in kg_results:
                kg_context += f"- {result['type']}: {result['value']}\n"
            context_parts.append(kg_context)
        
        # Add vector store context
        if vector_results:
            vs_context = "Related Content:\n"
            for result in vector_results:
                vs_context += f"- {result['content']} (confidence: {result['score']:.2f})\n"
            context_parts.append(vs_context)
        
        return "\n".join(context_parts)
    
    async def _generate_answer(self, question: str, context: str, domain: str, role: str) -> Dict:
        # Prepare prompt with context and metadata
        prompt = self._prepare_prompt(question, context, domain, role)
        
        # Generate answer using OpenAI
        response = await self._get_llm_response(prompt)
        
        return response
    
    def _prepare_prompt(self, question: str, context: str, domain: str, role: str) -> str:
        prompt_parts = [
            "Based on the following context and metadata, please answer the question.",
            f"\nContext:\n{context}",
            f"\nDomain: {domain if domain else 'General'}",
            f"Role: {role if role else 'General'}",
            f"\nQuestion: {question}",
            "\nPlease provide a concise and accurate answer with relevant citations from the context."
        ]
        return "\n".join(prompt_parts)
    
    async def _get_llm_response(self, prompt: str) -> Dict:
        response = await self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on the given context."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": self._extract_sources(response.choices[0].message.content),
            "confidence": response.choices[0].message.finish_reason == "stop"
        }
    
    @staticmethod
    def _extract_sources(answer: str) -> List[str]:
        # Extract citation references from the answer
        # This is a simplified implementation
        sources = []
        if "Source:" in answer:
            sources = [src.strip() for src in answer.split("Source:")[1:]]
        return sources