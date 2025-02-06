from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
from .models import RAGResponse, SearchResult
from .router import BaseRequestRouter, QueryIntent
from .reformulator import BaseQueryReformulator
from .retriever import BaseRetriever
from .completion_checker import BaseCompletionChecker
from .answer_generator import BaseAnswerGenerator
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
        completion_threshold: float = 0.7
    ):
        self.router = router
        self.reformulator = reformulator
        self.retriever = retriever
        self.completion_checker = completion_checker
        self.answer_generator = answer_generator
        self.completion_threshold = completion_threshold
    
    def process_query(self, query: str) -> Optional[RAGResponse]:
        logger.info(f"Processing query: {query}")
        
        # Step 1: Route the query
        logger.info("Step 1: Routing query...")
        intent = self.router.route_query(query)
        logger.info(f"Query intent: {intent}")
        if intent != QueryIntent.ANSWER:
            logger.info(f"Query rejected due to intent: {intent}")
            return None
            
        # Step 2: Reformulate query
        logger.info("Step 2: Reformulating query...")
        reformulated = self.reformulator.reformulate(query)
        logger.info(f"Reformulated query: {reformulated.refined_text}")
        logger.info(f"Generated keywords: {reformulated.keywords}")
        
        # Step 3: Retrieve relevant context
        logger.info("Step 3: Retrieving context...")
        search_results = self.retriever.hybrid_search(
            reformulated.refined_text,
            reformulated.keywords
        )
        logger.info(f"Found {len(search_results)} search results")
        for i, result in enumerate(search_results):
            logger.info(f"Result {i+1}: Score={result.score}, Text={result.text[:100]}...")
        
        # Step 4: Check if we can complete
        logger.info("Step 4: Checking completion capability...")
        completion_score = self.completion_checker.can_complete(query, search_results)
        logger.info(f"Completion score: {completion_score}")
        if completion_score < self.completion_threshold:
            logger.info(f"Query rejected due to low completion score: {completion_score} < {self.completion_threshold}")
            return None
            
        # Step 5: Generate answer
        logger.info("Step 5: Generating answer...")
        try:
            response = self.answer_generator.generate_answer(query, search_results)
            logger.info(f"Generated answer with confidence score: {response.confidence_score}")
            logger.info(f"Number of citations: {len(response.citations)}")
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise 