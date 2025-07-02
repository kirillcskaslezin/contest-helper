import io
from contextlib import redirect_stdout
from functools import wraps
from typing import TypeVar, Callable, Iterable

# Generic type variable for flexible function typing
T = TypeVar("T")


def printer(func: Callable[[T], None]) -> Callable[[T], Iterable[str]]:
    """
    A decorator that captures the standard output of a function and returns it as an iterable of strings.

    This wrapper is particularly useful for:
    - Testing functions that print to stdout
    - Capturing print output for logging purposes
    - Converting console output into processable data

    Args:
        func: A callable that takes a single argument of type T and returns None (prints to stdout)

    Returns:
        A wrapped function that returns the captured output lines as an iterable of strings

    Example:
        >>> @printer
        ... def greet(name):
        ...     print(f"Hello, {name}!")
        ...
        >>> list(greet("World"))
        ['Hello, World!']
    """

    @wraps(func)
    def wrapper(data: T) -> Iterable[str]:
        """
        Inner wrapper function that handles the output capture.

        Args:
            data: The input data of type T to be passed to the original function

        Returns:
            An iterable of stripped strings representing each line of output
        """
        # Create an in-memory text buffer
        buffer = io.StringIO()

        # Redirect stdout to our buffer
        with redirect_stdout(buffer):
            func(data)  # Execute the original function

        # Reset buffer position to beginning
        buffer.seek(0)

        # Return cleaned lines (stripped of whitespace)
        return map(str.strip, buffer)

    return wrapper
