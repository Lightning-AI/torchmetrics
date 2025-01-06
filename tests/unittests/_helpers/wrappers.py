import os
from functools import wraps
from typing import Any, Callable, Optional

import pytest

ALLOW_SKIP_IF_OUT_OF_MEMORY = os.getenv("ALLOW_SKIP_IF_OUT_OF_MEMORY", "0") == "1"
ALLOW_SKIP_IF_BAD_CONNECTION = os.getenv("ALLOW_SKIP_IF_BAD_CONNECTION", "0") == "1"
_ERROR_CONNECTION_MESSAGE_PATTERNS = (
    "We couldn't connect to",
    "Connection error",
    # "Can't load",  # fixme: this hid breaking change in transformers, so make it more specific
    # "`nltk` resource `punkt` is",  # todo: this is not intuitive ahy this is a connection issue
)


def skip_on_running_out_of_memory(reason: str = "Skipping test as it ran out of memory."):
    """Handle tests that sometimes runs out of memory, by simply skipping them."""

    def test_decorator(function: Callable, *args: Any, **kwargs: Any) -> Optional[Callable]:
        @wraps(function)
        def run_test(*args: Any, **kwargs: Any) -> Optional[Any]:
            try:
                return function(*args, **kwargs)
            except RuntimeError as ex:
                if "DefaultCPUAllocator: not enough memory:" not in str(ex):
                    raise ex
                if ALLOW_SKIP_IF_OUT_OF_MEMORY:
                    pytest.skip(reason)

        return run_test

    return test_decorator


def skip_on_connection_issues(reason: str = "Unable to load checkpoints from HuggingFace `transformers`."):
    """Handle download related tests if they fail due to connection issues.

    The tests run normally if no connection issue arises, and they're marked as skipped otherwise.

    """

    def test_decorator(function: Callable, *args: Any, **kwargs: Any) -> Optional[Callable]:
        @wraps(function)
        def run_test(*args: Any, **kwargs: Any) -> Optional[Any]:
            from urllib.error import URLError

            try:
                return function(*args, **kwargs)
            except URLError as ex:
                if "Error 403: Forbidden" not in str(ex) or not ALLOW_SKIP_IF_BAD_CONNECTION:
                    raise ex
                pytest.skip(reason)
            except (OSError, ValueError) as ex:
                if (
                    all(msg_start not in str(ex) for msg_start in _ERROR_CONNECTION_MESSAGE_PATTERNS)
                    or not ALLOW_SKIP_IF_BAD_CONNECTION
                ):
                    raise ex
                pytest.skip(reason)

        return run_test

    return test_decorator
