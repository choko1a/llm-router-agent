import logging
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from app.config import settings
from app.providers.base import ProviderConfigurationError, ProviderError

T = TypeVar("T")
logger = logging.getLogger("llm_router_agent.retry")


@dataclass(frozen=True)
class RetryResult(Generic[T]):
    value: T
    attempts: int


class ModelCallFailed(ProviderError):
    def __init__(self, label: str, attempts: int) -> None:
        self.label = label
        self.attempts = attempts
        super().__init__("%s model request failed after %s attempt(s)." % (label, attempts))


def call_with_retries(operation: Callable[[], T], label: str) -> RetryResult[T]:
    max_attempts = 1 + min(max(settings.model_max_retries, 0), 5)
    for attempt in range(1, max_attempts + 1):
        try:
            return RetryResult(value=operation(), attempts=attempt)
        except ProviderConfigurationError:
            raise
        except ProviderError as exc:
            if attempt == max_attempts:
                logger.error(
                    "model_request_exhausted route=%s attempts=%s error_type=%s",
                    label, attempt, type(exc).__name__,
                )
                raise ModelCallFailed(label, attempt) from exc
            logger.warning(
                "model_request_retry route=%s attempt=%s next_attempt=%s error_type=%s",
                label, attempt, attempt + 1, type(exc).__name__,
            )
            time.sleep(min(0.25 * (2 ** (attempt - 1)), 2.0))
    raise AssertionError("Retry loop ended without a result")
