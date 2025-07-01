import random as rd
import subprocess
from itertools import chain
from os import rmdir, mkdir
from os.path import isdir
from shutil import rmtree
from string import ascii_lowercase
from typing import Iterable, Any, Union, Callable, List, Set, Dict, Tuple, Type, TypeVar, Generic, NoReturn

from contest_helper import exceptions

# Type aliases for better code readability
Number = Union[int, float]
Input = TypeVar('Input')
Output = TypeVar('Output')
K = TypeVar('K')  # Key type for dictionaries
T = TypeVar('T')  # Generic type for values


class Value(Generic[T]):
    """A wrapper for constant values that makes them callable.

    Provides a consistent callable interface for constant values, allowing them
    to be used interchangeably with random value generators. When called,
    returns the stored value.

    Example:
        >>> five = Value(5)
        >>> five()
        5

    Args:
        value (T): The value to be wrapped.

    Note:
        This is particularly useful when you need to provide fixed values in contexts
        that expect callable generators.
    """

    def __init__(self, value: T):
        self._value_ = value

    def __call__(self) -> T:
        """Returns the wrapped value.

        Returns:
            T: The original value provided during initialization.
        """
        return self._value_


class Lambda(Value[T]):
    """A callable wrapper for arbitrary functions that conforms to the Value interface.

    Adapts any nullary function (callable with no arguments) to the Value[T] protocol,
    allowing it to be used interchangeably with other Value types in generator chains.

    Basic Examples:
        >>> # Wrap a simple lambda
        >>> rand_lambda = Lambda(lambda: random.randint(1, 10))
        >>> rand_lambda()  # Returns random number 1-10
        7

        >>> # Wrap existing function
        >>> def get_timestamp(): return time.time()
        >>> time_source = Lambda(get_timestamp)
        >>> time_source()  # Returns current timestamp
        1634567890.123

    Advanced Examples:
        >>> # Use with configurable generators
        >>> configurable = Lambda(
        ...     lambda: f"ID-{RandomNumber(1, 1000)():04}"
        ... )
        >>> configurable()  # Returns formatted random ID
        'ID-4821'

    Args:
        func: A callable that takes no arguments and returns a value of type T

    Note:
        - The wrapped function should be stateless/pure for predictable behavior
        - For functions requiring arguments, use functools.partial or lambda closures
        - Provides integration point for custom logic in generator pipelines
    """

    def __init__(self, func: Callable[..., T]):
        """Initializes the Lambda wrapper.

        Args:
            func: Callable to wrap. Must require no arguments when called.
        """
        super().__init__(None)
        self._func_ = func

    def __call__(self) -> T:
        """Executes the wrapped function and returns its result.

        Returns:
            T: Result of the function call

        Raises:
            Exception: Any exceptions raised by the wrapped function will propagate
        """
        return self._func_()


class RandomValue(Value[T]):
    """A callable generator that produces random values from a predefined sequence.

    This extends the `Value` wrapper to provide randomized output, selecting uniformly
    from the provided sequence each time it's called. The sequence is converted to a list
    during initialization for consistent random access.

    Example:
        >>> colors = RandomValue(['red', 'green', 'blue'])
        >>> colors()  # Randomly returns 'red', 'green' or 'blue'
        'green'
        >>> colors()
        'red'

    Args:
        sequence: An iterable collection of possible output values. Will be converted
                 to a list internally to support multiple sampling.

    Note:
        Uses `random.choice()` internally, which performs uniform sampling.
    """

    def __init__(self, sequence: Iterable[T]):
        super().__init__(None)
        self._sequence_ = list(sequence)

    def __call__(self) -> T:
        """Generates a new random selection from the sequence.

        Returns:
            A randomly chosen element from the original sequence.
            The same element may be returned on consecutive calls.

        Raises:
            IndexError: If the sequence was empty during initialization.
        """
        return rd.choice(self._sequence_)


class RandomNumber(RandomValue[T]):
    """Generates random numbers within a specified numeric range with fixed step size.

    Creates a sequence of numbers following start/stop/step parameters and provides
    uniform random selection from this sequence. Supports both integer and floating-point
    ranges.

    Example:
        >>> rand_int = RandomNumber(1, 10)  # Integers 1 through 9
        >>> rand_int()  # Random integer between 1-9
        7
        >>> rand_float = RandomNumber(0.5, 5.0, 0.5)  # 0.5, 1.0, 1.5...4.5
        >>> rand_float()  # Random float from the sequence
        3.0

    Args:
        start: Inclusive lower bound of the number range
        stop: Exclusive upper bound of the number range
        step: Interval between numbers (default: 1)

    Note:
        - The stop value is never included in the results (similar to range())
        - For floating-point ranges, be aware of potential rounding errors in the sequence
        - The sequence is generated once during initialization for efficient sampling
    """

    def __init__(self, start: T, stop: T, step: T = 1):
        sequence = []
        value = start
        while value < stop:
            sequence.append(value)
            value += step
        super().__init__(sequence)


class RandomWord(RandomValue[str]):
    """Generates random words with customizable length and character set.

    Creates pronounceable random words by selecting characters from a given alphabet.
    Supports both fixed and dynamically-generated length parameters.

    Basic Examples:
        >>> word_gen = RandomWord()  # Default: 3-10 lowercase letters
        >>> word_gen()  # Returns random word like 'hello' or 'xyz'
        'banana'

        >>> custom_gen = RandomWord(ascii_uppercase, 5, 5)  # Fixed 5-char uppercase
        >>> custom_gen()  # Returns like 'WORLD' or 'HELLO'
        'ABCDE'

    Advanced Example (Dynamic Lengths):
        >>> dynamic_gen = RandomWord(
        ...     min_length=RandomNumber(1, 5),  # Random length 1-4
        ...     max_length=RandomNumber(5, 10)  # Random length 5-9
        ... )
        >>> dynamic_gen()  # Could return 'a' (1 char) or 'python' (6 chars)
        'code'

    Args:
        sequence: Characters to use for word construction (default: lowercase letters)
        min_length: Minimum word length (int or Value[int] generator)
        max_length: Maximum word length (int or Value[int] generator)
        register: Case transformation function (default: str.lower)

    Note:
        - When using random length generators, actual word length is:
          randint(min_length(), max_length())
        - Character selection allows repetitions by default
        - For unique characters, pre-filter the input sequence
        - Register function applies after word generation
    """

    def __init__(
            self,
            sequence: Iterable[str] = ascii_lowercase,
            min_length: Union[int, Value[int]] = 3,
            max_length: Union[int, Value[int]] = 10,
            register: Callable[[str], str] = str.lower
    ):
        # Convert int lengths to Value wrappers if needed
        self._min_length_ = min_length if isinstance(min_length, Value) else Value(min_length)
        self._max_length_ = max_length if isinstance(max_length, Value) else Value(max_length)

        super().__init__(sequence)
        self._register_ = register

    def __call__(self) -> str:
        """Generates a random word according to configured parameters.

        Returns:
            str: Random word meeting length and character requirements

        Raises:
            ValueError: If invalid length parameters provided
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

    Basic Examples:
        >>> sentence_gen = RandomSentence()  # Default: 2-3 words, 3-10 letters each
        >>> sentence_gen()  # Returns like "hello world" or "foo bar baz"
        'python is awesome'

    Advanced Examples:
        >>> # Custom sentence structure with fixed word count
        >>> fixed_gen = RandomSentence(min_length=3, max_length=3)
        >>> fixed_gen()  # Always returns 3-word sentences
        'the quick brown'

        >>> # Dynamic lengths with random generators
        >>> dynamic_gen = RandomSentence(
        ...     min_length=RandomNumber(1, 3),
        ...     max_length=RandomNumber(3, 5),
        ...     min_word_length=RandomNumber(2, 4),
        ...     max_word_length=RandomNumber(5, 8)
        ... )
        >>> dynamic_gen()  # Variable sentence and word lengths
        'code works'

        >>> # Formal sentence structure
        >>> formal_gen = RandomSentence(
        ...     register=str.capitalize,
        ...     end=Value('.')
        ... )
        >>> formal_gen()  # Returns like "Important data." or "Random text."
        'Python is great.'

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
        - For punctuation, include in either sep or end parameters
        - Register applies to the entire final sentence
        - Uses recursive composition of RandomWord for word generation
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
        """Generates a complete random sentence according to configuration.

        Returns:
            str: Properly formatted random sentence

        Raises:
            ValueError: If any length parameters are invalid
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

    Basic Examples:
        >>> # List of 5 random integers (1-10)
        >>> int_list = RandomList(RandomNumber(1, 11), 5)
        >>> int_list()  # Returns like [3, 7, 2, 8, 1]
        [5, 2, 9, 2, 7]

        >>> # List of 3 random words
        >>> word_list = RandomList(RandomWord(), 3)
        >>> word_list()  # Returns like ['cat', 'dog', 'bird']
        ['apple', 'banana', 'orange']

    Advanced Examples:
        >>> # Dynamic list length with random bounds
        >>> dynamic_list = RandomList(
        ...     value_generator=RandomNumber(100, 200),
        ...     length=RandomNumber(3, 6)
        ... )
        >>> dynamic_list()  # Returns 3-5 random numbers (100-199)
        [150, 125, 180, 130]

        >>> # Nested random structures
        >>> matrix_gen = RandomList(
        ...     value_generator=RandomList(RandomNumber(0, 10), 3),
        ...     length=2
        ... )
        >>> matrix_gen()  # Returns 2x3 matrix
        [[1, 2, 3], [4, 5, 6]]

    Args:
        value_generator: Callable that generates individual elements
        length: List length (int or Value[int] generator)

    Note:
        - Each list element is generated independently
        - May contain duplicate values unless constrained by value_generator
        - For unique elements, use RandomSet instead
        - Length is evaluated on each call when using Value generators
    """

    def __init__(
            self,
            value_generator: Value[T],
            length: Union[int, Value[int]]
    ):
        super().__init__([])
        self._value_generator_ = value_generator
        self._length_ = length if isinstance(length, Value) else Value(length)

    def __call__(self) -> List[T]:
        """Generates a new random list according to configuration.

        Returns:
            List[T]: New list with independently generated elements

        Raises:
            ValueError: If length evaluates to negative number
        """
        length = self._length_()
        if length < 0:
            raise ValueError(f"Invalid list length: {length}")

        return [self._value_generator_() for _ in range(length)]


class RandomSet(RandomValue[T]):
    """Generates random sets of unique elements using a value generator.

    Creates sets of specified size where each element is unique, generated by
    repeatedly calling the provided generator until the desired number of
    distinct elements is obtained.

    Basic Examples:
        >>> # Set of 5 unique random numbers (1-10)
        >>> num_set = RandomSet(RandomNumber(1, 11), 5)
        >>> num_set()  # Returns like {3, 7, 2, 8, 1} (always 5 elements)
        {5, 2, 9, 7, 3}

        >>> # Set of 3 unique words
        >>> word_set = RandomSet(RandomWord(), 3)
        >>> word_set()  # Returns like {'cat', 'dog', 'bird'}
        {'apple', 'banana', 'orange'}

    Advanced Examples:
        >>> # Dynamic set size with random bounds
        >>> dynamic_set = RandomSet(
        ...     value_generator=RandomNumber(100, 200),
        ...     length=RandomNumber(3, 6)
        ... )
        >>> dynamic_set()  # Returns set with 3-5 unique numbers (100-199)
        {150, 125, 180, 130}

        >>> # Using with custom objects
        >>> class Point:
        ...     def __init__(self, x, y): self.x, self.y = x, y
        >>> point_set = RandomSet(
        ...     value_generator=Lambda(lambda: Point(RandomNumber(0, 10)(), RandomNumber(0, 10)())),
        ...     length=3
        ... )
        >>> point_set()  # Returns 3 unique points
        {Point(1,2), Point(3,4), Point(5,6)}

    Args:
        value_generator: Callable that generates candidate elements
        length: Target set size (int or Value[int] generator)

    Note:
        - Will repeatedly call generator until enough unique elements are obtained
        - May run indefinitely if generator cannot produce enough unique values
        - Elements must be hashable (implement __hash__)
        - For ordered collections with possible duplicates, use RandomList instead
    """

    def __init__(
            self,
            value_generator: Value[T],
            length: Union[int, Value[int]]
    ):
        super().__init__([])
        self._value_generator_ = value_generator
        self._length_ = length if isinstance(length, Value) else Value(length)

    def __call__(self) -> Set[T]:
        """Generates a new random set with unique elements.

        Returns:
            Set[T]: New set with specified number of unique elements

        Raises:
            ValueError: If length is negative
            TypeError: If generated elements aren't hashable
        """
        length = self._length_()
        if length < 0:
            raise ValueError(f"Invalid set size: {length}")

        result = set()
        while len(result) < length:
            result.add(self._value_generator_())
        return result


class RandomDict(Generic[K, T], RandomValue[Tuple[K, T]]):
    """Generates random dictionaries with unique keys using separate key and value generators.

    Creates dictionaries of specified size where each key-value pair is independently
    generated. Ensures all keys are unique by repeatedly generating keys until uniqueness
    is achieved.

    Basic Examples:
        >>> # Dictionary with 3 random string keys and number values
        >>> str_num_dict = RandomDict(
        ...     key_generator=RandomWord(3, 5),
        ...     value_generator=RandomNumber(1, 100),
        ...     length=3
        ... )
        >>> str_num_dict()  # Returns like {'cat': 42, 'dog': 17, 'bird': 89}
        {'apple': 42, 'banana': 17, 'orange': 89}

        >>> # Dictionary with fixed keys and random values
        >>> fixed_keys = RandomDict(
        ...     key_generator=RandomValue(['id', 'count', 'flag']),
        ...     value_generator=RandomValue([True, False]),
        ...     length=3
        ... )
        >>> fixed_keys()  # Returns like {'id': True, 'count': False, 'flag': True}
        {'id': True, 'count': False, 'flag': False}

    Advanced Examples:
        >>> # Nested random structures
        >>> nested_dict = RandomDict(
        ...     key_generator=RandomWord(3, 5),
        ...     value_generator=RandomList(RandomNumber(1, 10), 3),
        ...     length=2
        ... )
        >>> nested_dict()  # Returns like {'data': [1, 2, 3], 'items': [4, 5, 6]}
        {'scores': [1, 2, 3], 'stats': [4, 5, 6]}

        >>> # Dynamic dictionary size
        >>> dynamic_dict = RandomDict(
        ...     key_generator=RandomNumber(1000, 9999),
        ...     value_generator=RandomWord(),
        ...     length=RandomNumber(2, 5)
        ... )
        >>> dynamic_dict()  # Returns 2-4 entries like {1001: 'a', 1002: 'bc'}
        {1234: 'apple', 5678: 'banana'}

    Args:
        key_generator: Generator for dictionary keys (must produce hashable values)
        value_generator: Generator for dictionary values
        length: Dictionary size (int or Value[int] generator)

    Note:
        - Will repeatedly call key_generator until enough unique keys are obtained
        - May run indefinitely if key_generator cannot produce enough unique values
        - Keys must be hashable (implement __hash__)
        - Values may be duplicated even when keys are unique
        - For ordered collections, use collections.OrderedDict on the result
    """

    def __init__(
            self,
            key_generator: Value[K],
            value_generator: Value[T],
            length: Union[int, Value[int]]
    ):
        """Initializes the random dictionary generator.

        Args:
            key_generator: Generator for keys (must produce hashable values)
            value_generator: Generator for values
            length: Target dictionary size
        """
        super().__init__([])
        self._key_generator_ = key_generator
        self._value_generator_ = value_generator
        self._length_ = length if isinstance(length, Value) else Value(length)

    def __call__(self) -> Dict[K, T]:
        """Generates a new random dictionary with unique keys.

        Returns:
            Dict[K, T]: New dictionary with specified number of unique key-value pairs

        Raises:
            ValueError: If length is negative
            TypeError: If generated keys aren't hashable
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

    Creates a generator that produces lists by collecting and combining values
    from multiple source generators. Each call produces a new list containing
    the current output of all constituent generators.

    Basic Examples:
        >>> # Combine different value types
        >>> combined = CombineValues([
        ...     RandomWord(),
        ...     RandomNumber(1, 100),
        ...     Lambda(lambda: datetime.now().hour)
        ... ])
        >>> combined()  # Returns like ['hello', 42, 15]
        ['apple', 42, 15]

    Advanced Examples:
        >>> # Dynamic combination with nested generators
        >>> dynamic_combine = CombineValues([
        ...     RandomWord(),
        ...     RandomList(RandomNumber(1, 6), 3),
        ...     RandomDict(
        ...         key_generator=RandomWord(3, 5),
        ...         value_generator=RandomNumber(10, 20),
        ...         length=2
        ...     )
        ... ])
        >>> dynamic_combine()  # Returns mixed-type list
        ['banana', [1, 4, 2], {'apple': 15, 'pear': 18}]

        >>> # Configuration-driven generation
        >>> config_generators = [
        ...     Lambda(lambda: f"ID-{uuid.uuid4().hex[:6]}"),
        ...     Value("constant_value"),
        ...     RandomValue([True, False])
        ... ]
        >>> config_combiner = CombineValues(config_generators)
        >>> config_combiner()  # Returns like ['ID-a3f5b2', 'constant_value', False]
        ['ID-a3f5b2', 'constant_value', False]

    Args:
        sequence: Iterable of value generators to combine

    Note:
        - Generators are called in sequence order each time CombineValues is called
        - The output list length matches the number of input generators
        - Any generator type implementing the Value protocol can be used
        - For parallel combination with different lengths, use zip() with RandomList
    """

    def __init__(self, sequence: Iterable[Value[Any]]):
        """Initializes the value combiner.

        Args:
            sequence: Collection of value generators to combine
        """
        super().__init__(sequence)
        self._sequence_ = list(sequence)

    def __call__(self) -> List[Any]:
        """Generates a new combined list of values.

        Returns:
            List[Any]: New list containing current output from all generators

        Raises:
            Exception: Propagates any exceptions from constituent generators
        """
        return [generator() for generator in self._sequence_]


class Generator(Generic[Input, Output]):
    """A complete test case generator with validation that produces input/output pairs for testing solutions.

    Orchestrates the generation and validation of test cases by combining:
    - Predefined sample inputs
    - Dynamically generated random tests
    - Custom parsing and formatting logic
    - Test case validation

    Basic Example with Validation:
        >>> # Testing a function that counts words in text with validation
        >>> def validate_word_count_input(text: str):
        ...     if len(text) > 1000:
        ...         raise BadTestException("Input too long")

        >>> def word_count(text: str) -> int:
        ...     validate_word_count_input(text)
        ...     return len(text.split())

        >>> generator = Generator(
        ...     solution=word_count,
        ...     samples=["sample1.txt"],
        ...     tests_generator=RandomSentence(
        ...         min_length=1,
        ...         max_length=100
        ...     ),
        ...     tests_count=10,
        ...     input_parser=lambda lines: ' '.join(lines),
        ...     input_printer=lambda text: [text],
        ...     output_printer=lambda count: [str(count)]
        ... )
        >>> generator.run()  # Will automatically retry if BadTestException occurs

    Advanced Validation Example:
        >>> # Testing graph algorithm with complex validation
        >>> def validate_graph(graph: Dict[str, List[str]]):
        ...     if not graph:
        ...         raise BadTestException("Empty graph")
        ...     if any(not edges for edges in graph.values()):
        ...         raise BadTestException("Disconnected node")

        >>> def shortest_path(graph: Dict[str, List[str]]) -> int:
        ...     validate_graph(graph)
        ...     # Implementation here
        ...     return path_length

        >>> generator = Generator(
        ...     solution=shortest_path,
        ...     tests_generator=graph_generator,
        ...     tests_count=20,
        ...     # ... other parameters ...
        ... )
        >>> generator.run()  # Will retry on invalid graphs

    Args:
        solution: The reference solution function (Input -> Output) that may raise BadTestException
        samples: List of sample input file paths (optional)
        tests_generator: Generator for creating random test inputs
        tests_count: Number of random tests to generate
        input_parser: Function to convert input lines to Input type
        input_printer: Function to convert Input to output lines
        output_printer: Function to convert Output to answer lines

    Directory Structure Produced:
        tests/
        ├── sample01    # Sample input
        ├── sample01.a  # Sample output
        ├── 01          # Generated test input
        ├── 01.a        # Generated test output
        └── ...

    Note:
        - Automatically retries test generation when BadTestException occurs
        - Maximum retries are not limited - ensure generators can eventually produce valid tests
        - Sample files are not validated (assumed to be correct)
        - Validation should be implemented in the solution function
    """

    def __init__(
            self,
            solution: Callable[[Input], Output] = None,
            samples: Union[List[str], None] = None,
            tests_generator: Value = Value(None),
            tests_count: int = 0,
            input_parser: Callable[[Iterable[str]], Input] = None,
            input_printer: Callable[[Input], Iterable[str]] = None,
            output_printer: Callable[[Output], Iterable[str]] = None,
    ):
        """Initializes the test case generator with validation support.

        Args:
            solution: Reference implementation that includes validation
            samples: Sample input files to include (not validated)
            tests_generator: Random input generator
            tests_count: Number of valid random tests to create
            input_parser: Converts text input to program input
            input_printer: Converts program input to text
            output_printer: Formats program output as text
        """
        self._solution_ = solution
        self._samples_ = samples
        self._tests_generator_ = tests_generator
        self._tests_count_ = tests_count
        self._input_parser_ = input_parser
        self._input_printer_ = input_printer
        self._output_printer_ = output_printer

    def run(self):
        """Executes the test generation pipeline with automatic validation.

        Performs:
        1. Clears existing test directory
        2. Processes all sample inputs (without validation)
        3. Generates and validates random test cases
           - Retries automatically when BadTestException occurs
        4. Saves all valid test cases with corresponding answers

        Raises:
            OSError: If test directory operations fail
            ValueError: If input cannot be parsed
            Exception: Any exceptions other than BadTestException from solution
        """
        if isdir('tests'):
            rmtree('tests')
        mkdir('tests')
        tests = []

        # Process sample files (no validation)
        for index, sample in enumerate(self._samples_ or tuple(), 1):
            with open(sample, 'r', encoding='utf-8') as file:
                data = self._input_parser_(file)
                result = self._solution_(data)
                tests.append((f'sample{index:02}', data, result))

        # Generate and validate random tests
        for index in range(1, self._tests_count_ + 1):
            bad_test = True
            while bad_test:
                try:
                    bad_test = False
                    data = self._tests_generator_()
                    result = self._solution_(data)
                    tests.append((f'{index:02}', data, result))
                except exceptions.BadTestException:
                    bad_test = True

        # Save all valid test cases
        for filename, data, result in tests:
            with open(f'tests/{filename}', 'w', encoding='utf-8', newline='\n') as file:
                for line in self._input_printer_(data):
                    print(line, file=file)
            with open(f'tests/{filename}.a', 'w', encoding='utf-8', newline='\n') as file:
                for line in self._output_printer_(result):
                    print(line, file=file)
