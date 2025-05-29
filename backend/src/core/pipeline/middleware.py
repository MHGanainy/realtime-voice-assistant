from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, Any, Optional
from ..events.types import Event
import functools

class Middleware(ABC):
    """Base middleware class"""
    
    @abstractmethod
    async def process(
        self, 
        event: Event, 
        next_handler: Callable[[Event], AsyncIterator[Event]]
    ) -> AsyncIterator[Event]:
        """Process event and call next handler"""
        pass

class MiddlewarePipeline:
    """Pipeline for chaining middleware"""
    
    def __init__(self):
        self.middleware_stack = []
        
    def add(self, middleware: Middleware) -> None:
        """Add middleware to the pipeline"""
        self.middleware_stack.append(middleware)
        
    def remove(self, middleware: Middleware) -> None:
        """Remove middleware from the pipeline"""
        self.middleware_stack.remove(middleware)
        
    async def process(
        self, 
        event: Event, 
        final_handler: Callable[[Event], AsyncIterator[Event]]
    ) -> AsyncIterator[Event]:
        """Process event through middleware pipeline"""
        
        async def build_chain(index: int):
            if index >= len(self.middleware_stack):
                # End of middleware chain, call final handler
                async for result in final_handler(event):
                    yield result
            else:
                # Process through current middleware
                middleware = self.middleware_stack[index]
                next_handler = lambda e: build_chain(index + 1)
                async for result in middleware.process(event, next_handler):
                    yield result
                    
        async for result in build_chain(0):
            yield result