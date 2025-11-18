"""
Retry utilities with exponential backoff for Azure API calls.

Provides decorators and functions for retrying operations that may fail
due to transient errors (network issues, rate limiting, etc.).
"""

import time
import logging
import functools
from typing import Callable, Type, Tuple, Any
import asyncio

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    operation_name: str = None
):
    """
    Decorator for retrying function calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_factor: Multiplier for delay after each retry (default: 2.0)
        exceptions: Tuple of exception types to catch and retry (default: all exceptions)
        operation_name: Name of the operation for logging (default: function name)

    Example:
        @retry_with_backoff(max_retries=3, initial_delay=2.0)
        def call_azure_api():
            # API call that might fail
            pass

        @retry_with_backoff(max_retries=3, exceptions=(ConnectionError, TimeoutError))
        async def async_api_call():
            # Async API call
            pass
    """
    def decorator(func: Callable) -> Callable:
        import inspect

        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            # Async wrapper
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                func_name = operation_name or func.__name__
                delay = initial_delay

                for attempt in range(max_retries + 1):
                    try:
                        result = await func(*args, **kwargs)
                        if attempt > 0:
                            logger.info(f"[{func_name}] Succeeded on retry {attempt}/{max_retries}")
                        return result

                    except exceptions as e:
                        if attempt == max_retries:
                            logger.error(
                                f"[{func_name}] Failed after {max_retries} retries: {e}"
                            )
                            raise

                        logger.warning(
                            f"[{func_name}] Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )

                        await asyncio.sleep(delay)
                        delay *= backoff_factor

            return async_wrapper

        else:
            # Sync wrapper
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                func_name = operation_name or func.__name__
                delay = initial_delay

                for attempt in range(max_retries + 1):
                    try:
                        result = func(*args, **kwargs)
                        if attempt > 0:
                            logger.info(f"[{func_name}] Succeeded on retry {attempt}/{max_retries}")
                        return result

                    except exceptions as e:
                        if attempt == max_retries:
                            logger.error(
                                f"[{func_name}] Failed after {max_retries} retries: {e}"
                            )
                            raise

                        logger.warning(
                            f"[{func_name}] Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )

                        time.sleep(delay)
                        delay *= backoff_factor

            return sync_wrapper

    return decorator


def retry_operation(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    *args,
    **kwargs
) -> Any:
    """
    Retry a function call with exponential backoff (non-decorator version).

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries fail

    Example:
        result = retry_operation(
            some_function,
            max_retries=3,
            initial_delay=2.0,
            arg1="value",
            arg2=123
        )
    """
    delay = initial_delay
    func_name = func.__name__

    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            if attempt > 0:
                logger.info(f"[{func_name}] Succeeded on retry {attempt}/{max_retries}")
            return result

        except exceptions as e:
            if attempt == max_retries:
                logger.error(
                    f"[{func_name}] Failed after {max_retries} retries: {e}"
                )
                raise

            logger.warning(
                f"[{func_name}] Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )

            time.sleep(delay)
            delay *= backoff_factor


# Common Azure exception types for retry
AZURE_TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
    # Add more Azure-specific exceptions as needed
)


if __name__ == "__main__":
    # Test retry decorator
    import random

    @retry_with_backoff(max_retries=3, initial_delay=0.5)
    def flaky_function():
        """Simulates a function that fails randomly."""
        if random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("Simulated network error")
        return "Success!"

    print("Testing retry decorator...")
    try:
        result = flaky_function()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")
