import asyncio
import os
from fastapi import UploadFile
from services.document_processor import DocumentProcessor
from services.rag_pipeline import RAGPipeline
import pytest
from pathlib import Path

async def test_document_processing():
    # Initialize processor
    processor = DocumentProcessor()
    
    # Create a test text file
    test_content = """
    Apple Inc. is a technology company headquartered in Cupertino, California.
    Tim Cook is the CEO of Apple. The company was founded by Steve Jobs and Steve Wozniak.
    Apple produces various products including the iPhone, iPad, and MacBook.
    """
    
    test_file_path = Path("test_document.txt")
    test_file_path.write_text(test_content)
    
    # Create mock UploadFile
    class MockFile:
        def __init__(self, filename, content):
            self.filename = filename
            self._content = content
            
        async def read(self):
            return self._content.encode('utf-8')
            
        @property
        def file(self):
            return self
    
    # Process test file
    mock_file = MockFile("test_document.txt", test_content)
    result = await processor.process_file(mock_file)
    
    # Verify results
    assert "entities" in result
    assert "filename" in result
    assert result["status"] == "processed"
    
    # Test RAG pipeline
    rag = RAGPipeline()
    test_question = "Who is the CEO of Apple?"
    answer = await rag.process_query(test_question)
    
    # Verify RAG response
    assert "answer" in answer
    assert "sources" in answer
    assert "confidence" in answer
    
    # Cleanup
    if test_file_path.exists():
        test_file_path.unlink()

    return result, answer

if __name__ == "__main__":
    # Run the test
    result, answer = asyncio.run(test_document_processing())
    
    print("\nDocument Processing Result:")
    print("==========================")
    print(f"Filename: {result['filename']}")
    print("\nExtracted Entities:")
    for entity_type, entities in result['entities'].items():
        print(f"{entity_type}: {', '.join(entities)}")
    
    print("\nRAG Pipeline Result:")
    print("===================")
    print(f"Answer: {answer['answer']}")
    print(f"Confidence: {answer['confidence']}")
    print("\nSources:")
    for source in answer['sources']:
        print(f"- {source}")