from typing import Optional, Tuple, List, Dict, Any
import logging
import sys
from ..models import RAGResponse, QueryIntent
from ..components import (
    BaseRequestRouter,
    BaseQueryReformulator,
    BaseRetriever,
    BaseCompletionChecker,
    BaseAnswerGenerator
)
from .base import BaseWorkflow
from ..logging.base import StepLog

# Configure logging to write to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class RAGWorkflow(BaseWorkflow):
    def __init__(
        self,
        router: BaseRequestRouter,
        reformulator: BaseQueryReformulator,
        retriever: BaseRetriever,
        completion_checker: BaseCompletionChecker,
        answer_generator: BaseAnswerGenerator,
        completion_threshold: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="rag_workflow", metadata=metadata)
        self.router = router
        self.reformulator = reformulator
        self.retriever = retriever
        self.completion_checker = completion_checker
        self.answer_generator = answer_generator
        self.completion_threshold = completion_threshold
    
    def _execute(self, query: str) -> Tuple[Optional[RAGResponse], List[StepLog]]:
        step_logs: List[StepLog] = []
        
        # Route
        intent, route_log = self.router.execute(query)
        logger.info(route_log)

        step_logs.append(route_log)
        if intent != QueryIntent.ANSWER:
            return None, step_logs
        
        # Reformulate
        reformulated, reform_log = self.reformulator.execute(query)
        step_logs.append(reform_log)
        
        # Retrieve
        context, retrieve_log = self.retriever.execute(
            reformulated.refined_text,
            reformulated.keywords
        )
        step_logs.append(retrieve_log)
        logger.info(retrieve_log)
        # Check completion
        completion_score, check_log = self.completion_checker.execute(query, context)
        logger.info(check_log)
        step_logs.append(check_log)

        if completion_score < self.completion_threshold:
            return None, step_logs
        
        # Generate answer
        response, generate_log = self.answer_generator.execute(query, context)
        logger.info(generate_log)

        step_logs.append(generate_log)
        return response, step_logs 