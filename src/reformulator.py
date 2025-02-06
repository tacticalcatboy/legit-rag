from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from openai import OpenAI
import json
from .config import Settings

class BaseQueryReformulator(ABC):
    @dataclass
    class ReformulatedQuery:
        refined_text: str
        keywords: List[str]
        
    @abstractmethod
    def reformulate(self, query: str) -> ReformulatedQuery:
        """Reformulate the query and generate keywords."""
        pass

class LLMQueryReformulator(BaseQueryReformulator):
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        settings = Settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
    
    def reformulate(self, query: str) -> BaseQueryReformulator.ReformulatedQuery:
        prompt = """Given the user query, reformulate it to be more precise and extract key search terms.
        Return your response in this JSON format:
        {
            "refined_query": "reformulated question",
            "keywords": ["key1", "key2", "key3"]
        }
        
        Only return the JSON object, no other text.
        
        User Query: {query}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": prompt.format(query=query)
            }],
            temperature=0,
            response_format={ "type": "json_object" }  # Force JSON response
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return self.ReformulatedQuery(
                refined_text=result["refined_query"],
                keywords=result["keywords"]
            )
        except json.JSONDecodeError as e:
            # Log the actual response if JSON parsing fails
            print(f"Failed to parse JSON. Response was: {response.choices[0].message.content}")
            raise 