import functools
import asyncio
import logging
from typing import TypeVar, Callable, Any, List, Dict
from datetime import datetime
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')

def measure_performance(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure function performance"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        start_time = datetime.utcnow()
        try:
            result = await func(*args, **kwargs)
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {elapsed:.3f}s: {e}")
            raise
    return wrapper

def validate_config(required_fields: List[str]):
    """Decorator to validate configuration"""
    def decorator(cls):
        original_init = cls.__init__
        
        @functools.wraps(original_init)
        def new_init(self, config: Dict[str, Any], *args, **kwargs):
            # Validate required fields
            missing = [f for f in required_fields if f not in config]
            if missing:
                raise ValueError(f"Missing required config fields: {missing}")
            original_init(self, config, *args, **kwargs)
            
        cls.__init__ = new_init
        return cls
    return decorator

def provider(provider_type: str, name: str):
    """
    Decorator to auto-register providers with the registry.
    
    Usage:
        @provider("stt", "whisper")
        class WhisperSTT(STTProvider):
            ...
    """
    def decorator(cls):
        # Import here to avoid circular imports
        from ..core.factories.registry import registry
        
        # Register the provider
        if provider_type == "stt":
            registry.register_stt(name, cls)
        elif provider_type == "llm":
            registry.register_llm(name, cls)
        elif provider_type == "tts":
            registry.register_tts(name, cls)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
            
        logger.info(f"Registered {provider_type} provider: {name}")
        return cls
    return decorator

@asynccontextmanager
async def managed_resource(resource):
    """
    Context manager for resources with async initialization/cleanup.
    
    Usage:
        async with managed_resource(provider) as p:
            await p.process()
    """
    try:
        if hasattr(resource, '__aenter__'):
            yield await resource.__aenter__()
        else:
            if hasattr(resource, 'initialize'):
                await resource.initialize()
            yield resource
    finally:
        if hasattr(resource, '__aexit__'):
            await resource.__aexit__(None, None, None)
        elif hasattr(resource, 'cleanup'):
            await resource.cleanup()