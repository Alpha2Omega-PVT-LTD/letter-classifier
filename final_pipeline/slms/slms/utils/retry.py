import time
from typing import Callable, TypeVar, Any
from utils.logger import logger

T = TypeVar('T')

def retry_with_backoff(
    fn: Callable[[], T],
    retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    check_none: bool = True
) -> Any:
    """
    Executes a function and retries it on exception or if returning None (when check_none is True).
    """
    delay = initial_delay
    for attempt in range(1, retries + 1):
        try:
            result = fn()
            if check_none and result is None:
                raise ValueError("Function returned None/Invalid output.")
            return result
        except Exception as e:
            logger.warning(
                f"Attempt {attempt}/{retries} failed with error: {e}. "
                f"Retrying in {delay} seconds..."
            )
            if attempt == retries:
                logger.error(f"Max retries reached. Execution failed.")
                return None
            time.sleep(delay)
            delay *= backoff_factor
    return None
