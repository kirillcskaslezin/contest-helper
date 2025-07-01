class BadTestException(Exception):
    """Custom exception to indicate invalid test cases during automated test generation.

    This exception should be raised by the solution function when it detects
    an invalid test input that should be discarded and regenerated.

    Basic Usage:
        >>> def solution(input_data):
        ...     if is_invalid(input_data):
        ...         raise BadTestException("Invalid input parameters")
        ...     # Normal processing...

    Note:
        - The Generator class will catch and handle this exception automatically
        - Use descriptive messages to help debug test generation issues
        - Only use for truly invalid cases, not for normal error conditions
        - This exception should typically be raised before any heavy computation
    """

    pass
