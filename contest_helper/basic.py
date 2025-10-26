import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import mkdir
from os.path import isdir
from shutil import rmtree
from typing import Iterable, Any, Union, Callable, List, Dict, Tuple, TypeVar, Generic

from contest_helper import exceptions
from contest_helper.values import Value

# Type aliases for better code readability
Number = Union[int, float]  # Numeric type that can be either int or float
Input = TypeVar('Input')  # Generic type for input data
Output = TypeVar('Output')  # Generic type for output data


# Container for sample input data
@dataclass
class SampleData:
    """Container for sample inputs that preserves original text and parsed form.

    Attributes:
        raw_lines: The original lines read from the user's sample file (without trailing newlines).
        parsed: The parsed representation passed to the solution/validator.
        raw_bytes: (optional) The original bytes if loaded from a binary file.
    """
    raw_lines: List[str] = None
    parsed: Any = None
    raw_bytes: bytes = None


# --- Adapters for input/output handling ---

class InputAdapter(ABC):
    """Abstract adapter for handling input reading/parsing and writing."""

    @abstractmethod
    def parse_sample(self, path: str) -> SampleData:
        """Read a sample file and return SampleData with `parsed` filled in."""
        raise NotImplementedError

    @abstractmethod
    def write_input(self, file_path: str, data: Any) -> None:
        """Write generated input data to file_path."""
        raise NotImplementedError


class OutputAdapter(ABC):
    """Abstract adapter for handling output formatting and writing."""

    @abstractmethod
    def write_output(self, file_path: str, result: Any) -> None:
        raise NotImplementedError


class TextInputAdapter(InputAdapter):
    """Default text input adapter. Subclass to customize parsing/printing."""

    def parse_lines(self, lines: Iterable[str]) -> Any:
        """Convert list of lines to structured input. Default: return as-is."""
        return list(lines)

    def input_lines(self, data: Any) -> Iterable[str]:
        """Render structured input back to text lines. Default: str(data)."""
        return [str(data)]

    def parse_sample(self, path: str) -> SampleData:
        with open(path, 'r', encoding='utf-8') as f:
            raw_lines = [line.rstrip('\n') for line in f]
        parsed = self.parse_lines(raw_lines)
        return SampleData(raw_lines=raw_lines, parsed=parsed)

    def write_input(self, file_path: str, data: Any) -> None:
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            for line in self.input_lines(data):
                print(line, file=f)


class BinaryInputAdapter(InputAdapter):
    """Binary input adapter. Subclass to customize parse/serialization of bytes."""

    def parse_bytes(self, blob: bytes) -> Any:
        """Convert bytes to structured input. Default: return bytes as-is."""
        return blob

    def input_bytes(self, data: Any) -> Iterable[bytes]:
        """Render structured input back to bytes. Default: if bytes, write directly."""
        if isinstance(data, (bytes, bytearray)):
            return [bytes(data)]
        raise TypeError("BinaryInputAdapter.input_bytes expects bytes; override for custom types.")

    def parse_sample(self, path: str) -> SampleData:
        with open(path, 'rb') as f:
            raw_bytes = f.read()
        parsed = self.parse_bytes(raw_bytes)
        return SampleData(raw_bytes=raw_bytes, parsed=parsed)

    def write_input(self, file_path: str, data: Any) -> None:
        with open(file_path, 'wb') as f:
            for chunk in self.input_bytes(data):
                f.write(chunk)


class TextOutputAdapter(OutputAdapter):
    """Default text output adapter. Subclass to customize formatting."""

    def output_lines(self, result: Any) -> Iterable[str]:
        return [str(result)]

    def write_output(self, file_path: str, result: Any) -> None:
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            for line in self.output_lines(result):
                print(line, file=f)


class BinaryOutputAdapter(OutputAdapter):
    """Binary output adapter. Subclass to customize formatting to bytes."""

    def output_bytes(self, result: Any) -> Iterable[bytes]:
        if isinstance(result, (bytes, bytearray)):
            return [bytes(result)]
        raise TypeError("BinaryOutputAdapter.output_bytes expects bytes; override for custom types.")

    def write_output(self, file_path: str, result: Any) -> None:
        with open(file_path, 'wb') as f:
            for chunk in self.output_bytes(result):
                f.write(chunk)


class IOPipeline:
    """Handles input/output parsing and formatting operations.

    Provides configurable parsing and printing functions for test case data.
    """

    def __init__(
            self,
            input_parser: Callable[[Iterable[str]], Any] = None,
            input_printer: Callable[[Any], Iterable[str]] = None,
            output_printer: Callable[[Any], Iterable[str]] = None
    ):
        """Initialize the I/O pipeline with parsing/printing functions.

        Args:
            input_parser: Converts input lines to program input (default: identity)
            input_printer: Converts program input to text lines (default: str())
            output_printer: Formats program output as text lines (default: str())
        """
        self.input_parser = input_parser or (lambda x: x)
        self.input_printer = input_printer or (lambda x: [str(x)])
        self.output_printer = output_printer or (lambda x: [str(x)])


class TestCaseGenerator:
    """Generates and validates individual test cases."""

    def __init__(self, solution: Callable, logger: logging.Logger, validator: Callable[[Any, Any], bool] = None):
        """Initialize with solution function and logger.

        Args:
            solution: Reference implementation for validation
            logger: Configured logger for messaging
            validator: Callable that returns True for valid (data, result) pairs; defaults to accept all
        """
        self.solution = solution
        self.logger = logger
        self.validator = validator or (lambda _data, _result: True)

    def generate_case(self, generator: Callable, group_name: str = None) -> Tuple[Any, Any]:
        """Generate and validate a single test case.

        Args:
            generator: Callable that produces test input
            group_name: Optional group identifier for logging

        Returns:
            Tuple of (input_data, expected_output)

        Raises:
            Exception: For any non-BadTestException errors
        """
        retry_count = 0
        while True:
            try:
                retry_count += 1
                self.logger.debug(f"Generating test case (attempt {retry_count})" +
                                  (f" for group '{group_name}'" if group_name else ""))

                data = generator()
                result = self.solution(data)

                # Validate generated test case if validator is provided (defaults to accept all)
                if not self.validator(data, result):
                    raise exceptions.BadTestException("Validation failed for generated test case")

                self.logger.info(f"Successfully generated test case" +
                                 (f" for group '{group_name}'" if group_name else ""))
                return data, result

            except exceptions.BadTestException as e:
                self.logger.warning(
                    f"Invalid test case generated" +
                    (f" for group '{group_name}'" if group_name else "") +
                    f": {str(e)}. Retrying..."
                )
            except Exception as e:
                self.logger.error(f"Unexpected error generating test case: {str(e)}")
                raise


class TestGroupManager:
    """Manages configuration and generation of test groups."""

    def __init__(self, logger: logging.Logger):
        """Initialize with logger instance."""
        self.logger = logger

    def prepare_groups(
            self,
            generators: Union[Callable, List[Callable]],
            counts: Union[int, List[int]]
    ) -> Dict[str, Tuple[Callable, int]]:
        """Normalize group specifications to a consistent dictionary format.

        Args:
            generators: Single generator, list of generators, or dict mapping
            counts: Single count or list matching generators

        Returns:
            Dictionary mapping group names to (generator, count) tuples
        """
        if isinstance(generators, dict):
            return generators

        if not isinstance(generators, list):
            generators = [generators]
        if not isinstance(counts, list):
            counts = [counts]

        return {f"group_{i + 1}": (gen, cnt)
                for i, (gen, cnt) in enumerate(zip(generators, counts))}


class TestFileManager:
    """Manages filesystem operations for test cases."""

    def __init__(self, input_adapter: InputAdapter, output_adapter: OutputAdapter, logger: logging.Logger):
        """Initialize with adapters and logger.

        Args:
            input_adapter: Adapter for input read/write
            output_adapter: Adapter for output write
            logger: Configured logger for messaging
        """
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.logger = logger

    def prepare_directory(self):
        """Initialize a clean test directory.

        Removes existing directory if present and creates a new empty one.
        """
        if isdir('tests'):
            self.logger.info("Clearing existing test directory")
            rmtree('tests')
        mkdir('tests')
        self.logger.info("Created fresh test directory")

    def save_test_case(self, filename: str, data: Any, result: Any):
        """Save a test case to disk.

        Args:
            filename: Base filename (without extension)
            data: Input data to save (may be SampleData)
            result: Expected output to save
        """
        try:
            input_path = f'tests/{filename}'
            output_path = f'tests/{filename}.a'

            # Save input file
            if isinstance(data, SampleData):
                # Preserve original form for samples
                if data.raw_bytes is not None:
                    with open(input_path, 'wb') as f:
                        f.write(data.raw_bytes)
                else:
                    with open(input_path, 'w', encoding='utf-8', newline='\n') as f:
                        if data.raw_lines is not None:
                            for line in data.raw_lines:
                                print(line, file=f)
            else:
                # Delegate rendering to adapter
                self.input_adapter.write_input(input_path, data)

            # Save output file via adapter
            self.output_adapter.write_output(output_path, result)

            self.logger.debug(f"Saved test case {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save test case {filename}: {str(e)}")
            raise


class Generator(Generic[Input, Output]):
    """Orchestrates the complete test generation process.

    Coordinates sample processing, random test generation, validation,
    and file operations using decomposed components.
    """

    def __init__(
            self,
            solution: Callable[[Input], Output] = None,
            samples: List[str] = None,
            tests_generator: Union[Value, List[Value], Dict[str, Value]] = Value(None),
            tests_count: Union[int, List[int], Dict[str, int]] = 0,
            input_parser: Callable[[Any], Input] = None,
            input_printer: Callable[[Input], Iterable[str]] = None,
            output_printer: Callable[[Output], Iterable[str]] = None,
            validator: Callable[[Input, Output], bool] = None,
            input_adapter: InputAdapter = None,
            output_adapter: OutputAdapter = None,
    ):
        """Initialize the test generator with configuration.

        Args:
            solution: Reference implementation for validation
            samples: List of sample input file paths
            tests_generator: Generator(s) for random tests
            tests_count: Number of tests to generate per group
            input_parser: Converts input lines to program input
            input_printer: Converts program input to text lines
            output_printer: Formats program output as text lines
            validator: Function to validate generated (input, output); defaults to accepting all
            input_adapter: Adapter for input file reading/writing
            output_adapter: Adapter for output file writing
        """
        # Initialize components
        self.io = IOPipeline(input_parser, input_printer, output_printer)
        self.logger = self._setup_logger()
        self.case_generator = TestCaseGenerator(solution, self.logger, validator)
        self.group_manager = TestGroupManager(self.logger)

        # Initialize adapters (default to text-based behavior)
        self.input_adapter = input_adapter or TextInputAdapter()
        self.output_adapter = output_adapter or TextOutputAdapter()
        self.file_manager = TestFileManager(self.input_adapter, self.output_adapter, self.logger)

        # Store configuration
        self.solution = solution
        self.samples = samples or []
        self.tests_generator = tests_generator
        self.tests_count = tests_count

    def _setup_logger(self) -> logging.Logger:
        """Configure and return a logger instance.

        Returns:
            Configured logger with stream handler.
        """
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _process_samples(self) -> List[Tuple[str, Any, Any]]:
        """Process all sample test cases from files."""
        samples = []
        for index, sample_file in enumerate(self.samples, 1):
            try:
                sample = self.input_adapter.parse_sample(sample_file)
                result = self.solution(sample.parsed)
                samples.append((f'sample{index:02}', sample, result))
            except Exception as e:
                self.logger.error(f"Failed to process sample {sample_file}: {str(e)}")
                raise
        return samples

    def run(self):
        """Execute the complete test generation pipeline.

        Performs:
        1. Directory preparation
        2. Sample processing
        3. Random test generation
        4. File saving

        Raises:
            Exception: If any critical generation step fails
        """
        try:
            self.logger.info("Starting test generation")

            # Initialize directory
            self.file_manager.prepare_directory()

            all_tests = []
            test_counter = 1  # Continuous numbering across groups

            # Process samples
            if self.samples:
                self.logger.info(f"Processing {len(self.samples)} samples")
                all_tests.extend(self._process_samples())

            # Process generated tests
            if self.tests_count:
                groups = self.group_manager.prepare_groups(
                    self.tests_generator,
                    self.tests_count
                )

                for group_name, (generator, count) in groups.items():
                    if count <= 0:
                        continue

                    self.logger.info(f"Generating {count} tests for {group_name}")
                    for _ in range(count):
                        data, result = self.case_generator.generate_case(generator, group_name)
                        filename = f"{test_counter:02}"
                        all_tests.append((filename, data, result))
                        test_counter += 1

            # Save all tests
            self.logger.info(f"Saving {len(all_tests)} test cases")
            for filename, data, result in all_tests:
                self.file_manager.save_test_case(filename, data, result)

            self.logger.info("Test generation completed successfully")

        except Exception as e:
            self.logger.critical(f"Test generation failed: {str(e)}", exc_info=True)
            raise
