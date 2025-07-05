import logging
import random as rd
from os import mkdir
from os.path import isdir
from shutil import rmtree
from string import ascii_lowercase
from typing import Iterable, Any, Union, Callable, List, Set, Dict, Tuple, TypeVar, Generic

from contest_helper import exceptions

# Type aliases for better code readability
Number = Union[int, float]  # Numeric type that can be either int or float
Input = TypeVar('Input')    # Generic type for input data
Output = TypeVar('Output')  # Generic type for output data
K = TypeVar('K')           # Key type for dictionaries
T = TypeVar('T')           # Generic type for values


class Value(Generic[T]):
    """A callable wrapper for constant values that provides a consistent generator interface.

    This class allows constant values to be used interchangeably with random value generators
    in contexts that expect callable objects. When called, it returns the stored value.

    Example:
        >>> five = Value(5)
        >>> five()  # Returns the wrapped value 5
        5

    Args:
        value: The value to be wrapped. Can be of any type T.

    Note:
        Useful when you need to provide fixed values in contexts that expect callable generators.
    """

    def __init__(self, value: T):
        """Initialize the Value wrapper with the given value."""
        self._value_ = value

    def __call__(self) -> T:
        """Return the wrapped value when called as a function.

        Returns:
            The original value provided during initialization.
        """
        return self._value_


class Lambda(Value[T]):
    """A callable wrapper that adapts functions to the Value interface.

    This class wraps any nullary function (callable with no arguments) to make it
    compatible with the Value[T] protocol, allowing it to be used in generator chains.

    Examples:
        Basic usage:
        >>> rand_lambda = Lambda(lambda: random.randint(1, 10))
        >>> rand_lambda()  # Returns a random number between 1-10

        Wrapping existing functions:
        >>> def get_timestamp(): return time.time()
        >>> time_source = Lambda(get_timestamp)
        >>> time_source()  # Returns current timestamp

    Args:
        func: A callable that takes no arguments and returns a value of type T.

    Note:
        - The wrapped function should ideally be stateless/pure for predictable behavior
        - For functions requiring arguments, use functools.partial or lambda closures
    """

    def __init__(self, func: Callable[..., T]):
        """Initialize the Lambda wrapper with the given function.

        Args:
            func: A callable that requires no arguments when called.
        """
        super().__init__(None)
        self._func_ = func

    def __call__(self) -> T:
        """Execute the wrapped function and return its result.

        Returns:
            The result of the function call.

        Raises:
            Exception: Propagates any exceptions raised by the wrapped function.
        """
        return self._func_()


class RandomValue(Value[T]):
    """A generator that produces random values from a predefined sequence.

    Extends the Value wrapper to provide randomized output by selecting uniformly
    from the provided sequence. The sequence is stored as a list for efficient random access.

    Example:
        >>> colors = RandomValue(['red', 'green', 'blue'])
        >>> colors()  # Randomly returns one of the colors
        'green'

    Args:
        sequence: An iterable collection of possible output values. Converted to a list
                 internally to support multiple sampling.

    Note:
        Uses random.choice() internally for uniform sampling.
    """

    def __init__(self, sequence: Iterable[T]):
        """Initialize with the sequence of possible values."""
        super().__init__(None)
        self._sequence_ = list(sequence)

    def __call__(self) -> T:
        """Generate a new random selection from the sequence.

        Returns:
            A randomly chosen element from the sequence.

        Raises:
            IndexError: If the sequence was empty during initialization.
        """
        return rd.choice(self._sequence_)


class RandomNumber(Value[T]):
    """Generates random numbers within a specified range with a fixed step size.

    Creates random numbers following start/stop/step parameters without generating
    the entire sequence upfront. Supports both integer and float ranges.

    Example:
        >>> rand_int = RandomNumber(1, 10)  # Integers 1 through 9
        >>> rand_int()  # Returns a random integer in the range

        >>> rand_float = RandomNumber(0.5, 5.0, 0.5)  # 0.5, 1.0, 1.5...4.5
        >>> rand_float()  # Returns a random float from the sequence

    Args:
        start: Inclusive lower bound of the range
        stop: Exclusive upper bound of the range
        step: Interval between numbers (must be positive)

    Raises:
        ValueError: If parameters form an invalid range
    """

    def __init__(self, start: T, stop: T, step: T = 1):
        """Initialize and validate the number generator."""
        super().__init__(None)

        # Validate parameters
        if step <= 0:
            raise ValueError(f"Step must be positive, got {step}")
        if start >= stop:
            raise ValueError(f"Invalid range: start ({start}) >= stop ({stop})")

        self._start = start
        self._stop = stop
        self._step = step

        # Calculate number of possible values
        if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
            self._discrete = True
            self._num_values = (stop - start) // step
            if (stop - start) % step != 0:
                self._num_values += 1
        else:
            self._discrete = False
            self._num_values = int((stop - start) / step)
            # Handle floating point imprecision
            if (stop - start) % step != 0:
                self._num_values += 1

        if self._num_values <= 0:
            raise ValueError(f"Invalid range: no values between {start} and {stop} with step {step}")

    def __call__(self) -> T:
        """Generate a new random number within the configured range.

        Returns:
            A randomly chosen number from the range.
        """
        if self._discrete:
            # For integers, use randrange which is more efficient
            random_index = rd.randrange(0, self._num_values)
            return self._start + random_index * self._step
        else:
            # For floats, calculate a random value directly
            random_steps = rd.randint(0, self._num_values - 1)
            result = self._start + random_steps * self._step
            # Handle potential floating point rounding errors
            return min(result, self._stop - self._step / 2)


class RandomWord(RandomValue[str]):
    """Generates random words with customizable length and character set.

    Creates pronounceable random words by selecting characters from a given alphabet.
    Supports both fixed and dynamically-generated length parameters.

    Examples:
        Basic usage:
        >>> word_gen = RandomWord()  # Default: 3-10 lowercase letters
        >>> word_gen()  # Returns a random word

        Custom character set:
        >>> custom_gen = RandomWord(ascii_uppercase, 5, 5)  # Fixed 5-char uppercase
        >>> custom_gen()  # Returns a 5-character uppercase word

    Args:
        sequence: Characters to use for word construction (default: lowercase letters)
        min_length: Minimum word length (int or Value[int] generator)
        max_length: Maximum word length (int or Value[int] generator)
        register: Case transformation function (default: str.lower)

    Note:
        - Word length is determined by randint(min_length(), max_length())
        - Character selection allows repetitions by default
        - Register function applies after word generation
    """

    def __init__(
            self,
            sequence: Iterable[str] = ascii_lowercase,
            min_length: Union[int, Value[int]] = 3,
            max_length: Union[int, Value[int]] = 10,
            register: Callable[[str], str] = str.lower
    ):
        """Initialize the word generator with configuration."""
        # Convert int lengths to Value wrappers if needed
        self._min_length_ = min_length if isinstance(min_length, Value) else Value(min_length)
        self._max_length_ = max_length if isinstance(max_length, Value) else Value(max_length)

        super().__init__(sequence)
        self._register_ = register

    def __call__(self) -> str:
        """Generate a random word according to configured parameters.

        Returns:
            A random word meeting length and character requirements.

        Raises:
            ValueError: If invalid length parameters are provided.
        """
        min_len = self._min_length_()
        max_len = self._max_length_()
        if min_len > max_len:
            raise ValueError(f"min_length ({min_len}) > max_length ({max_len})")

        length = rd.randint(min_len, max_len)
        return self._register_(''.join(
            rd.choices(self._sequence_, k=length)
        ))


class RandomSentence(RandomWord):
    """Generates random sentences composed of random words with customizable structure.

    Creates natural-looking sentences by combining randomly generated words according
    to specified grammatical rules. Supports dynamic configuration of all components.

    Examples:
        Basic usage:
        >>> sentence_gen = RandomSentence()  # Default: 2-3 words, 3-10 letters each
        >>> sentence_gen()  # Returns a simple sentence

        Formal structure:
        >>> formal_gen = RandomSentence(register=str.capitalize, end=Value('.'))
        >>> formal_gen()  # Returns capitalized sentences ending with period

    Args:
        sequence: Character set for word generation (default: lowercase letters)
        min_length: Minimum words per sentence (int or Value[int])
        max_length: Maximum words per sentence (int or Value[int])
        min_word_length: Minimum letters per word (int or Value[int])
        max_word_length: Maximum letters per word (int or Value[int])
        register: Case transformation function (default: str.lower)
        sep: Word separator (str or Value[str], default: space)
        end: Sentence ending (str or Value[str], default: empty string)

    Note:
        - Word length parameters are passed to the underlying RandomWord generator
        - Register applies to the entire final sentence
    """

    def __init__(
            self,
            sequence: Iterable[str] = ascii_lowercase,
            min_length: Union[int, Value[int]] = 2,
            max_length: Union[int, Value[int]] = 3,
            min_word_length: Union[int, Value[int]] = 3,
            max_word_length: Union[int, Value[int]] = 10,
            register: Callable[[str], str] = str.lower,
            sep: Union[str, Value[str]] = ' ',
            end: Union[str, Value[str]] = ''
    ):
        """Initialize the sentence generator with configuration."""
        super().__init__(sequence)
        self._min_length_ = min_length if isinstance(min_length, Value) else Value(min_length)
        self._max_length_ = max_length if isinstance(max_length, Value) else Value(max_length)
        self._min_word_length_ = min_word_length if isinstance(min_word_length, Value) else Value(min_word_length)
        self._max_word_length_ = max_word_length if isinstance(max_word_length, Value) else Value(max_word_length)
        self._word_generator_ = RandomWord(
            sequence,
            self._min_word_length_,
            self._max_word_length_
        )
        self._register_ = register
        self._sep_ = sep if isinstance(sep, Value) else Value(sep)
        self._end_ = end if isinstance(end, Value) else Value(end)

    def __call__(self) -> str:
        """Generate a complete random sentence according to configuration.

        Returns:
            A properly formatted random sentence.

        Raises:
            ValueError: If any length parameters are invalid.
        """
        min_len = self._min_length_()
        max_len = self._max_length_()
        if min_len > max_len:
            raise ValueError(f"min_length ({min_len}) > max_length ({max_len})")

        count = rd.randint(min_len, max_len)
        words = [self._word_generator_() for _ in range(count)]
        sentence = self._sep_().join(words) + self._end_()
        return self._register_(sentence)


class RandomList(RandomValue[T]):
    """Generates random lists by repeatedly calling a value generator.

    Creates lists of specified length where each element is independently generated
    by the provided value generator. Supports both fixed and dynamic list lengths.

    Examples:
        Basic usage:
        >>> int_list = RandomList(RandomNumber(1, 11), 5)
        >>> int_list()  # Returns list of 5 random numbers

        Nested structures:
        >>> matrix_gen = RandomList(RandomList(RandomNumber(0, 10), 3), 2)
        >>> matrix_gen()  # Returns 2x3 matrix

    Args:
        value_generator: Callable that generates individual elements
        length: List length (int or Value[int] generator)

    Note:
        - Each element is generated independently
        - May contain duplicates unless constrained by value_generator
        - Length is evaluated on each call when using Value generators
    """

    def __init__(
            self,
            value_generator: Value[T],
            length: Union[int, Value[int]]
    ):
        """Initialize the list generator."""
        super().__init__([])
        self._value_generator_ = value_generator
        self._length_ = length if isinstance(length, Value) else Value(length)

    def __call__(self) -> List[T]:
        """Generate a new random list according to configuration.

        Returns:
            A new list with independently generated elements.

        Raises:
            ValueError: If length evaluates to negative number.
        """
        length = self._length_()
        if length < 0:
            raise ValueError(f"Invalid list length: {length}")

        return [self._value_generator_() for _ in range(length)]


class RandomSet(RandomValue[T]):
    """Generates random sets of unique elements using a value generator.

    Creates sets of specified size where each element is unique, generated by
    repeatedly calling the provided generator until enough distinct elements are obtained.

    Examples:
        Basic usage:
        >>> num_set = RandomSet(RandomNumber(1, 11), 5)
        >>> num_set()  # Returns set of 5 unique numbers

        With custom objects:
        >>> point_set = RandomSet(Lambda(lambda: Point(random(), random())), 3)
        >>> point_set()  # Returns 3 unique points

    Args:
        value_generator: Callable that generates candidate elements
        length: Target set size (int or Value[int] generator)

    Note:
        - Will retry until enough unique elements are obtained
        - May run indefinitely if generator can't produce enough unique values
        - Elements must be hashable (implement __hash__)
    """

    def __init__(
            self,
            value_generator: Value[T],
            length: Union[int, Value[int]]
    ):
        """Initialize the set generator."""
        super().__init__([])
        self._value_generator_ = value_generator
        self._length_ = length if isinstance(length, Value) else Value(length)

    def __call__(self) -> Set[T]:
        """Generate a new random set with unique elements.

        Returns:
            A new set with the specified number of unique elements.

        Raises:
            ValueError: If length is negative.
            TypeError: If generated elements aren't hashable.
        """
        length = self._length_()
        if length < 0:
            raise ValueError(f"Invalid set size: {length}")

        result = set()
        while len(result) < length:
            result.add(self._value_generator_())
        return result


class RandomDict(Generic[K, T], RandomValue[Tuple[K, T]]):
    """Generates random dictionaries with unique keys using separate generators.

    Creates dictionaries of specified size where each key-value pair is independently
    generated. Ensures key uniqueness by repeatedly generating keys when needed.

    Examples:
        Basic usage:
        >>> str_num_dict = RandomDict(RandomWord(3,5), RandomNumber(1,100), 3)
        >>> str_num_dict()  # Returns dict with 3 string-number pairs

        Nested structures:
        >>> nested_dict = RandomDict(RandomWord(), RandomList(RandomNumber(1,10),3), 2)
        >>> nested_dict()  # Returns dict with word keys and number lists

    Args:
        key_generator: Generator for dictionary keys (must produce hashable values)
        value_generator: Generator for dictionary values
        length: Dictionary size (int or Value[int] generator)

    Note:
        - Will retry until enough unique keys are obtained
        - Keys must be hashable (implement __hash__)
        - Values may be duplicated even when keys are unique
    """

    def __init__(
            self,
            key_generator: Value[K],
            value_generator: Value[T],
            length: Union[int, Value[int]]
    ):
        """Initialize the dictionary generator."""
        super().__init__([])
        self._key_generator_ = key_generator
        self._value_generator_ = value_generator
        self._length_ = length if isinstance(length, Value) else Value(length)

    def __call__(self) -> Dict[K, T]:
        """Generate a new random dictionary with unique keys.

        Returns:
            A new dictionary with the specified number of unique key-value pairs.

        Raises:
            ValueError: If length is negative.
            TypeError: If generated keys aren't hashable.
        """
        length = self._length_()
        if length < 0:
            raise ValueError(f"Invalid dictionary size: {length}")

        result = {}
        while len(result) < length:
            key = self._key_generator_()
            result[key] = self._value_generator_()
        return result


class CombineValues(RandomValue[Any]):
    """Combines multiple value generators into a single list output.

    Creates a generator that produces lists by collecting values from multiple
    source generators. Each call produces a new list with current outputs.

    Examples:
        Basic usage:
        >>> combined = CombineValues([RandomWord(), RandomNumber(1,100), Lambda(time.time)])
        >>> combined()  # Returns list with word, number, and timestamp

    Args:
        sequence: Iterable of value generators to combine

    Note:
        - Generators are called in sequence order on each invocation
        - Output list length matches number of input generators
        - Any generator implementing Value protocol can be used
    """

    def __init__(self, sequence: Iterable[Value[Any]]):
        """Initialize the value combiner."""
        super().__init__(sequence)
        self._sequence_ = list(sequence)

    def __call__(self) -> List[Any]:
        """Generate a new combined list of values.

        Returns:
            A new list containing current output from all generators.

        Raises:
            Exception: Propagates any exceptions from constituent generators.
        """
        return [generator() for generator in self._sequence_]


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

    def __init__(self, solution: Callable, logger: logging.Logger):
        """Initialize with solution function and logger.

        Args:
            solution: Reference implementation for validation
            logger: Configured logger for messaging
        """
        self.solution = solution
        self.logger = logger

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

                self.logger.info(f"Successfully generated test case" +
                               (f" for group '{group_name}'" if group_name else ""))
                return data, result

            except exceptions.BadTestException as e:
                self.logger.warning(f"Invalid test case generated: {str(e)}. Retrying...")
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

    def __init__(self, io_pipeline: IOPipeline, logger: logging.Logger):
        """Initialize with I/O pipeline and logger.

        Args:
            io_pipeline: Configured I/O handlers
            logger: Configured logger for messaging
        """
        self.io = io_pipeline
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
            data: Input data to save
            result: Expected output to save

        Raises:
            Exception: If file operations fail
        """
        try:
            # Save input file
            with open(f'tests/{filename}', 'w', encoding='utf-8', newline='\n') as f:
                for line in self.io.input_printer(data):
                    print(line, file=f)

            # Save output file
            with open(f'tests/{filename}.a', 'w', encoding='utf-8', newline='\n') as f:
                for line in self.io.output_printer(result):
                    print(line, file=f)

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
            input_parser: Callable[[Iterable[str]], Input] = None,
            input_printer: Callable[[Input], Iterable[str]] = None,
            output_printer: Callable[[Output], Iterable[str]] = None,
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
        """
        # Initialize components
        self.io = IOPipeline(input_parser, input_printer, output_printer)
        self.logger = self._setup_logger()
        self.case_generator = TestCaseGenerator(solution, self.logger)
        self.group_manager = TestGroupManager(self.logger)
        self.file_manager = TestFileManager(self.io, self.logger)

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
        """Process all sample test cases from files.

        Returns:
            List of (filename, input_data, expected_output) tuples

        Raises:
            Exception: If any sample processing fails
        """
        samples = []
        for index, sample_file in enumerate(self.samples, 1):
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    data = self.io.input_parser(f)
                    result = self.solution(data)
                    samples.append((f'sample{index:02}', data, result))
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