from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "documents"
    
    # LLM Settings
    router_model: str = "gpt-4-turbo-preview"
    reformulator_model: str = "gpt-4-turbo-preview"
    completion_model: str = "gpt-4-turbo-preview"
    answer_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-small"
    
    # RAG Settings
    completion_threshold: float = 0.7
    
    class Config:
        env_file = ".env" 