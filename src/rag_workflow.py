from typing import Optional
from .models import RAGResponse, QueryIntent
from .components import (
    BaseRequestRouter,
    BaseQueryReformulator,
    BaseRetriever,
    BaseCompletionChecker,
    BaseAnswerGenerator
)
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    def process_query(self, query: str) -> Optional[RAGResponse]:
        # Route
        logger.info("Step 1: Routing query...")
        intent = self.router.execute(query)
        logger.info(f"Query intent: {intent}")
        if intent != QueryIntent.ANSWER:
            return None
        
        # Reformulate
        logger.info("Step 2: Reformulating query...")
        reformulated = self.reformulator.execute(query)
        logger.info(f"Reformulated query: {reformulated.refined_text}")
        logger.info(f"Generated keywords: {reformulated.keywords}")        
        
        # Retrieve
        logger.info("Step 3: Retrieving context...")
        context = self.retriever.execute(
            reformulated.refined_text,
            reformulated.keywords
        )

        logger.info(f"Found {len(context)} search results")
        for i, result in enumerate(context):
            logger.info(f"Result {i+1}: Score={result.score}, Text={result.text[:100]}...")

        
        # Check completion
        logger.info("Step 4: Checking completion capability...")
        completion_confidence = self.completion_checker.execute(query, context)
        if completion_confidence < self.completion_threshold:
            logger.info(f"Completion confidence is too low.{completion_score} < {self.completion_threshold}")
            return None
        
        # Generate answer
        logger.info("Step 5: Generating answer...")
        try:
            response = self.answer_generator.execute(query, context)
            logger.info(f"Generated answer with confidence score: {response.confidence_score}")
            logger.info(f"Number of citations: {len(response.citations)}")
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise 