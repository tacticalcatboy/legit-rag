from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI
from .models import SearchResult, RAGResponse, Citation
import json
from .config import Settings

class BaseAnswerGenerator(ABC):
    @abstractmethod
    def generate_answer(self, query: str, context: List[SearchResult]) -> RAGResponse:
        """
        Generate answer with citations.
        Returns a RAGResponse containing the answer, citations, and confidence score.
        """
        pass 

class LLMAnswerGenerator(BaseAnswerGenerator):
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        settings = Settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
    
    def generate_answer(self, query: str, context: List[SearchResult]) -> RAGResponse:
        # Format context for the prompt
        formatted_context = "\n\n".join([
            f"Context {i+1}:\n{result.text}"
            for i, result in enumerate(context)
        ])
        
        prompt = """Using the provided context, answer the question. Your response must be in JSON format with these fields:
        1. "answer": Your detailed response
        2. "citations": A list of objects, each with:
           - "text": The relevant quote from the context
           - "relevance_score": A float between 0-1 indicating how relevant this citation is
        3. "confidence_score": A float between 0-1 indicating your overall confidence
        
        Only use information from the provided context. If you're unsure, reflect that in the confidence score.
        
        Context:
        {context}
        
        Question: {query}
        
        Respond with only the JSON object, no other text."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user", 
                "content": prompt.format(context=formatted_context, query=query)
            }],
            temperature=0,
            max_tokens=1000
        )
        
        result = response.choices[0].message.content
        parsed = json.loads(result)
        
        citations = [
            Citation(
                text=cite["text"],
                metadata={},  # Could be enhanced with context metadata
                relevance_score=cite["relevance_score"]
            )
            for cite in parsed["citations"]
        ]
        
        return RAGResponse(
            answer=parsed["answer"],
            citations=citations,
            confidence_score=parsed["confidence_score"]
        ) 