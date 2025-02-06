from abc import ABC, abstractmethod
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Filter, FieldCondition, MatchText
from openai import OpenAI
import numpy as np
from .models import SearchResult, Document
from .config import Settings

class BaseRetriever(ABC):
    @abstractmethod
    def semantic_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Perform semantic search using vector embeddings."""
        pass
    
    @abstractmethod
    def keyword_search(self, keywords: List[str], top_k: int = 5) -> List[SearchResult]:
        """Perform keyword-based search."""
        pass
    
    def hybrid_search(self, query: str, keywords: List[str], top_k: int = 5) -> List[SearchResult]:
        """Combine semantic and keyword search results."""
        semantic_results = self.semantic_search(query, top_k)
        keyword_results = self.keyword_search(keywords, top_k)
        return self._merge_results(semantic_results, keyword_results)
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the retrieval database."""
        pass
    
    def _merge_results(self, semantic_results: List[SearchResult], 
                      keyword_results: List[SearchResult]) -> List[SearchResult]:
        """Implement your result fusion strategy."""
        pass

class QdrantRetriever(BaseRetriever):
    def __init__(
        self,
        collection_name: str,
        embedding_model: str = "text-embedding-3-small",
        url: Optional[str] = None
    ):
        settings = Settings()  # This will load from .env
        
        self.client = QdrantClient(url=url)
        self.openai_client = OpenAI(api_key=settings.openai_api_key)  # Pass the API key here
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
    
    def _get_embedding(self, text: str) -> np.ndarray:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)
    
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
        # Simple keyword search using Qdrant's payload filtering
        keyword_conditions = [
            FieldCondition(
                key="text",
                match=MatchText(text=keyword)
            )
            for keyword in keywords
        ]
        
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                should=keyword_conditions  # This creates an OR condition
            ),
            limit=top_k
        )[0]  # scroll returns (points, next_page_offset)
        
        return [
            SearchResult(
                text=point.payload["text"],
                vector=np.array(point.vector) if point.vector else np.array([]),
                metadata={k: v for k, v in point.payload.items() if k != "text"},
                score=1.0  # For keyword search, we don't have a natural score
            )
            for point in results
        ]
    
    def _merge_results(self, semantic_results: List[SearchResult], 
                      keyword_results: List[SearchResult]) -> List[SearchResult]:
        """Merge results using a simple score-based approach."""
        # Create a dictionary to store unique results with their best scores
        merged = {}
        
        # Add semantic results with their original scores
        for result in semantic_results:
            merged[result.text] = result
        
        # Add keyword results, keeping the higher score if duplicate
        for result in keyword_results:
            if result.text not in merged or result.score > merged[result.text].score:
                merged[result.text] = result
        
        # Sort by score and return top results
        return sorted(merged.values(), key=lambda x: x.score, reverse=True) 
