from abc import abstractmethod
from enum import Enum
from openai import OpenAI
from .base_component import BaseComponent
from .config import Settings

class QueryIntent(Enum):
    ANSWER = "answer"          # Direct answer using RAG
    CLARIFY = "clarify"       # Need more information
    REJECT = "reject"         # Cannot/should not answer

class BaseRequestRouter(BaseComponent):
    """Base class for routing user queries"""
    def __init__(self):
        super().__init__(name="router")
    
    async def _execute(self, query: str) -> QueryIntent:
        """Execute routing"""
        return await self.route(query)
    
    @abstractmethod
    async def route(self, query: str) -> QueryIntent:
        """Determine the intent of the query."""
        pass

class LLMRequestRouter(BaseRequestRouter):
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        super().__init__()
        settings = Settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
        
    def route_query(self, query: str) -> QueryIntent:
        prompt = """You are a query router. Analyze the following query and determine how it should be handled.
        Return EXACTLY ONE of these values (nothing else): ANSWER, CLARIFY, or REJECT
        
        Guidelines:
        - ANSWER: If the query is clear, specific, and can be answered with factual information
        - CLARIFY: If the query is ambiguous, vague, or needs more context
        - REJECT: If the query is inappropriate, harmful, or completely out of scope
        
        Query: {query}
        
        Decision:"""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt.format(query=query)}],
            temperature=0,
            max_tokens=10
        )
        
        decision = response.choices[0].message.content.strip().upper()
        return QueryIntent[decision] 