from typing import Optional
from .models import RAGResponse, QueryIntent
from .components import (
    BaseRequestRouter,
    BaseQueryReformulator,
    BaseRetriever,
    BaseCompletionChecker,
    BaseAnswerGenerator
)

class RAGWorkflow:
    def __init__(
        self,
        router: BaseRequestRouter,
        reformulator: BaseQueryReformulator,
        retriever: BaseRetriever,
        completion_checker: BaseCompletionChecker,
        answer_generator: BaseAnswerGenerator,
        completion_threshold: float = 0.7,
    ):
        self.router = router
        self.reformulator = reformulator
        self.retriever = retriever
        self.completion_checker = completion_checker
        self.answer_generator = answer_generator
        self.completion_threshold = completion_threshold
    
    async def process_query(self, query: str) -> Optional[RAGResponse]:
        # Route
        intent = await self.router.execute(query)
        if intent != QueryIntent.ANSWER:
            return None
        
        # Reformulate
        reformulated = await self.reformulator.execute(query)
        
        # Retrieve
        context = await self.retriever.execute(
            reformulated.refined_text,
            reformulated.keywords
        )
        
        # Check completion
        can_complete = await self.completion_checker.execute(query, context)
        if not can_complete:
            return None
        
        # Generate answer
        response = await self.answer_generator.execute(query, context)
        return response 