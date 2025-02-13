from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Filter, FieldCondition, MatchText
from openai import OpenAI
import numpy as np
from ..models import SearchResult, Document
from ..config import Settings
from .base_component import BaseComponent

class BaseRetriever(BaseComponent):
    """Base class for retrieving relevant context"""
    def __init__(self):
        super().__init__(name="retriever")
    
    def _execute(self, query: str, keywords: List[str]) -> List[SearchResult]:
        """Execute retrieval"""
        return self.retrieve(query, keywords)
    
    @abstractmethod
    def retrieve(self, query: str, keywords: List[str]) -> List[SearchResult]:
        """Retrieve relevant context based on query and keywords."""
        pass

class VectorRetriever(BaseRetriever):
    def __init__(
        self,
        collection_name: str,
        embedding_model: str = "text-embedding-3-small",
        url: Optional[str] = None
    ):
        super().__init__()
        settings = Settings()
        
        self.client = QdrantClient(url=url)
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Create collection if it doesn't exist
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=1536,  # OpenAI embedding dimension
                distance=rest.Distance.COSINE
            )
        )
    
    def retrieve(self, query: str, keywords: List[str]) -> List[SearchResult]:
        """Combine semantic and keyword search results."""
        semantic_results =  self.semantic_search(query)
        keyword_results =  self.keyword_search(keywords)
        return self.rerank(semantic_results, keyword_results)
    
    def add_documents(self, documents: List[Document]) -> None:
        # Get embeddings for all texts
        embeddings = [self._get_embedding(doc.text) for doc in documents]
        
        # Prepare points for Qdrant
        points = [
            rest.PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "text": doc.text,
                    **(doc.metadata or {})
                }
            )
            for i, (doc, embedding) in enumerate(zip(documents, embeddings))
        ]
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        query_vector = self._get_embedding(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        return [
            SearchResult(
                text=hit.payload["text"],
                vector=np.array(hit.vector) if hit.vector else np.array([]),
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                score=hit.score
            )
            for hit in results
        ]
    
    def keyword_search(self, keywords: List[str], top_k: int = 5) -> List[SearchResult]:
        keyword_conditions = [
            FieldCondition(
                key="text",
                match=MatchText(text=keyword)
            )
            for keyword in keywords
        ]
        
        results = (self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                should=keyword_conditions
            ),
            limit=top_k
        ))[0]
        
        return [
            SearchResult(
                text=point.payload["text"],
                vector=np.array(point.vector) if point.vector else np.array([]),
                metadata={k: v for k, v in point.payload.items() if k != "text"},
                score=1.0
            )
            for point in results
        ]
    
    def _get_embedding(self, text: str) -> np.ndarray:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def rerank(self, semantic_results: List[SearchResult], 
                      keyword_results: List[SearchResult]) -> List[SearchResult]:
        """Merge results using a simple score-based approach."""
        merged = {}
        
        for result in semantic_results:
            merged[result.text] = result
        
        for result in keyword_results:
            if result.text not in merged or result.score > merged[result.text].score:
                merged[result.text] = result
        
        return sorted(merged.values(), key=lambda x: x.score, reverse=True) 
