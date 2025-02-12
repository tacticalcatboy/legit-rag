from abc import ABC, abstractmethod
from typing import Any

class BaseComponent(ABC):
    """Base class for all RAG workflow components"""
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def _execute(self, *args, **kwargs) -> Any:
        """Internal execution method to be implemented by components"""
        pass
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the component"""
        return await self._execute(*args, **kwargs) 