import asyncio
import functools
import logging
from typing import TypeVar, Callable, Type, Tuple, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')

class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 30.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.exceptions = exceptions

def exponential_backoff_retry(config: RetryConfig = None):
    """
    Decorator for retrying async functions with exponential backoff.
    
    Usage:
        @exponential_backoff_retry(RetryConfig(max_attempts=5))
        async def api_call():
            ...
    """
    if config is None:
        config = RetryConfig()
        
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            attempt = 0
            delay = config.initial_delay
            last_exception = None
            
            while attempt < config.max_attempts:
                try:
                    result = await func(*args, **kwargs)
                    
                    # Success - log if we had retries
                    if attempt > 0:
                        logger.info(
                            f"{func.__name__} succeeded after {attempt + 1} attempts"
                        )
                    
                    return result
                    
                except config.exceptions as e:
                    last_exception = e
                    attempt += 1
                    
                    if attempt >= config.max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {config.max_attempts} attempts: {e}"
                        )
                        raise
                    
                    # Calculate next delay with backoff
                    delay = min(delay * config.backoff_factor, config.max_delay)
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{config.max_attempts} "
                        f"failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    
                    await asyncio.sleep(delay)
                    
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator

# Convenience decorator for network errors
def retry_on_network_error(func: Callable[..., T]) -> Callable[..., T]:
    """Retry on network-related errors"""
    # Import aiohttp only when decorator is used
    try:
        import aiohttp
        exceptions = (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ConnectionError,
        )
    except ImportError:
        # If aiohttp not available, use basic exceptions
        exceptions = (
            asyncio.TimeoutError,
            ConnectionError,
        )
    
    config = RetryConfig(
        max_attempts=5,
        initial_delay=1.0,
        backoff_factor=2.0,
        exceptions=exceptions
    )
    return exponential_backoff_retry(config)(func)

# Convenience decorator for API errors
retry_on_api_error = exponential_backoff_retry(
    RetryConfig(
        max_attempts=3,
        initial_delay=2.0,
        backoff_factor=3.0,
        max_delay=60.0
    )
)