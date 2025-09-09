from typing import Dict, List
from services.knowledge_graph import KnowledgeGraph
from services.vector_store import VectorStore
from openai import OpenAI, OpenAIError
import os
from unittest.mock import AsyncMock, Mock

class RAGPipeline:
    def __init__(self, openai_client: OpenAI = None):
        """
        RAG pipeline initialization.
        Accepts an optional OpenAI client for testing.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not openai_client and not api_key:
            print("Warning: OPENAI_API_KEY not set. Using mock OpenAI client for testing.")
            # Create a mock OpenAI client to avoid runtime errors
            mock_client = Mock()
            mock_choice = Mock()
            mock_choice.message = Mock(content="This is a mock answer due to missing API key.")
            mock_choice.finish_reason = "stop"
            mock_client.chat.completions.create = AsyncMock(return_value=Mock(choices=[mock_choice]))
            self.openai_client = mock_client
        else:
            self.openai_client = openai_client or OpenAI(api_key=api_key)

        self.kg = KnowledgeGraph()
        self.vector_store = VectorStore()

    async def process_query(
        self, question: str, domain: str = None, role: str = None
    ) -> Dict:
        """
        Main RAG pipeline: fetches KG & vector results, combines context, and generates an answer.
        """
        kg_results = self.kg.query_subgraph(question)
        vector_results = self.vector_store.search(question, k=3)
        context = self._combine_context(kg_results, vector_results)
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

    async def _generate_answer(
        self, question: str, context: str, domain: str, role: str
    ) -> Dict:
        """
        Async method to generate answer from OpenAI API with error handling and mock fallback.
        """
        prompt = self._prepare_prompt(question, context, domain, role)

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate answers based on the given context."
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            answer_text = message.content

        except OpenAIError as e:
            print(f"OpenAI API error: {e}. Returning mock answer for testing.")
            answer_text = "This is a mock answer due to OpenAI API error."
            finish_reason = "stop"

        return {
            "answer": answer_text,
            "sources": self._extract_sources(answer_text),
            "confidence": 1.0 if finish_reason == "stop" else 0.0,
        }

    def _prepare_prompt(self, question: str, context: str, domain: str, role: str) -> str:
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
        sources = []
        if "Source:" in answer:
            sources = [src.strip() for src in answer.split("Source:")[1:]]
        return sources


# Usage example (ensure OPENAI_API_KEY is set in .env or environment)
# from dotenv import load_dotenv
# load_dotenv()
# rag_pipeline = RAGPipeline()
