from typing import Dict, List
from services.knowledge_graph import KnowledgeGraph
from services.vector_store import VectorStore
from openai import OpenAI


class RAGPipeline:
    def __init__(self, openai_client=None):
        # OpenAI client (injectable for tests)
        self.openai_client = openai_client or OpenAI()
        self.kg = KnowledgeGraph()
        self.vector_store = VectorStore()

    async def process_query(self, question: str, domain: str = None, role: str = None) -> Dict:
        """
        Main RAG pipeline: fetches KG & vector results, combines context, and generates an answer.
        """
        # 1. Get relevant entities from knowledge graph
        kg_results = self.kg.query_subgraph(question)

        # 2. Get relevant documents from vector store
        vector_results = self.vector_store.search(question, k=3)

        # 3. Combine context
        context = self._combine_context(kg_results, vector_results)

        # 4. Generate answer using LLM (async)
        response = await self._generate_answer(question, context, domain, role)

        return {
            "answer": response["answer"],
            "sources": response["sources"],
            "confidence": response["confidence"],
        }

    def _combine_context(self, kg_results: List[Dict], vector_results: List[Dict]) -> str:
        context_parts = []

        if kg_results:
            kg_context = "Knowledge Graph Information:\n"
            for result in kg_results:
                kg_context += f"- {result['type']}: {result['value']}\n"
            context_parts.append(kg_context)

        if vector_results:
            vs_context = "Related Content:\n"
            for result in vector_results:
                vs_context += f"- {result['content']} (similarity: {result['score']:.2f})\n"
            context_parts.append(vs_context)

        return "\n".join(context_parts)

    async def _generate_answer(self, question: str, context: str, domain: str, role: str) -> Dict:
        """
        Async method to generate answer from OpenAI API.
        """
        prompt = self._prepare_prompt(question, context, domain, role)

        # Await the OpenAI API call
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on the given context."},
                {"role": "user", "content": prompt},
            ],
        )

        # Access the message from the response
        message = response.choices[0].message

        return {
            "answer": message.content,
            "sources": self._extract_sources(message.content),
            "confidence": 1.0 if response.choices[0].finish_reason == "stop" else 0.0,
        }

    def _prepare_prompt(self, question: str, context: str, domain: str, role: str) -> str:
        """
        Prepares prompt for the LLM based on context, domain, role, and question.
        """
        prompt_parts = [
            "Based on the following context and metadata, please answer the question.",
            f"\nContext:\n{context}",
            f"\nDomain: {domain if domain else 'General'}",
            f"Role: {role if role else 'General'}",
            f"\nQuestion: {question}",
            "\nPlease provide a concise and accurate answer. If possible, cite relevant sources.",
        ]
        return "\n".join(prompt_parts)

    @staticmethod
    def _extract_sources(answer: str) -> List[str]:
        """
        Naive parser to extract sources from the answer.
        """
        sources = []
        if "Source:" in answer:
            sources = [src.strip() for src in answer.split("Source:")[1:]]
        return sources
