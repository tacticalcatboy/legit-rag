from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from .config import Settings
from .router import LLMRequestRouter
from .reformulator import LLMQueryReformulator
from .retriever import QdrantRetriever
from .completion_checker import LLMCompletionChecker
from .answer_generator import LLMAnswerGenerator
from .rag_workflow import RAGWorkflow
from .models import RAGResponse, Document

app = FastAPI()
settings = Settings()

# Initialize vector store
retriever = QdrantRetriever(
    collection_name=settings.qdrant_collection_name,
    embedding_model=settings.embedding_model,
    url=settings.qdrant_url
)

# Initialize components
router = LLMRequestRouter(model=settings.router_model)
reformulator = LLMQueryReformulator(model=settings.reformulator_model)
completion_checker = LLMCompletionChecker(model=settings.completion_model)
answer_generator = LLMAnswerGenerator(model=settings.answer_model)

workflow = RAGWorkflow(
    router=router,
    reformulator=reformulator,
    retriever=retriever,
    completion_checker=completion_checker,
    answer_generator=answer_generator,
    completion_threshold=settings.completion_threshold
)

class QueryRequest(BaseModel):
    query: str

class DocumentRequest(BaseModel):
    documents: List[Document]

@app.post("/query")
async def process_query(request: QueryRequest) -> Optional[RAGResponse]:
    """Process a query through the RAG workflow"""
    response = workflow.process_query(request.query)
    if response is None:
        raise HTTPException(status_code=400, detail="Query cannot be answered with available context")
    return response

@app.post("/documents")
async def add_documents(request: DocumentRequest) -> dict:
    """Add documents to the retriever"""
    try:
        documents = [Document(text=doc.text, metadata=doc.metadata) for doc in request.documents]
        retriever.add_documents(documents)
        return {"status": "success", "message": f"Added {len(documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)