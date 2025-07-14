# Contest Helper Framework

A powerful Python framework for competitive programming problem creation, test generation, and package management.

## Features

- **Problem Initialization**: Quickly scaffold new problem directories with templates
- **Randomized Test Generation**: Flexible system for creating diverse test cases
- **Problem Packaging**: Combine all components into ready-to-use zip archives
- **Statement Management**: Automatically update problem statements with examples
- **Custom Postprocessors**: Configure scoring systems with various options

## Installation

```bash
pip install contest-helper
```

## Command Line Tools

### Start a New Problem
```bash
ch-start-problem problem_name [--language en|ru] [--checker]
```

### Create Custom Postprocessor
```bash
ch-make-postprocessor --max-value 100 [--groups "10,20,30"] [--different] [--by-groups]
```

### Package Problem for Import
```bash
ch-combine problem_directory [options]
```

### Update Statement with Examples
```bash
ch-statement-preview problem_directory [--lang en|ru]
```

## Core Components

### Test Generators
- Random numbers, strings, lists, dictionaries
- Configurable ranges and constraints
- Composition of generators for complex cases

### Problem Configuration
- Time/memory limits
- Input/output settings
- Checker configuration
- Solution validation

### File Management
- Automatic test case processing
- File categorization (compile/run/post)
- Sample test detection

## Usage Examples

1. **Create a new problem**:
```bash
ch-start-problem my_problem -l en -c
```

2. **Generate tests**:
Edit the generated `generator.py` to define your test cases

3. **Preview statement**:
```bash
ch-statement-preview my_problem --lang en
```

4. **Package for import**:
```bash
ch-combine my_problem --time-limit 2000 --memory-limit 256000000
```

## Test Generation Examples

### 1. Basic Number Generation
```python
from contest_helper import RandomNumber, Generator

# Generate 20 random numbers between 1-100
generator = Generator(
    tests_generator=RandomNumber(1, 101),
    tests_count=20,
    input_printer=lambda x: [str(x)],
    output_printer=lambda x: [str(x**2)]  # Output is input squared
)
generator.run()
```

### 2. String Manipulation Problem
```python
from contest_helper import RandomWord

# Generate 15 random words (3-10 chars) and reverse them
generator = Generator(
    tests_generator=RandomWord(),
    tests_count=15,
    input_printer=lambda x: [x],
    output_printer=lambda x: [x[::-1]]  # Reversed string
)
generator.run()
```

### 3. Matrix Generation
```python
from contest_helper import RandomNumber, RandomList

# Generate 10 random 3x3 matrices (values 0-9)
matrix_gen = RandomList(
    value_generator=RandomList(RandomNumber(0, 10), 3),
    length=3
)

generator = Generator(
    tests_generator=matrix_gen,
    tests_count=10,
    input_printer=lambda m: [f"{n}" for row in m for n in row],
    output_printer=lambda m: [str(sum(sum(row) for row in m))]  # Sum of all elements
)
generator.run()
```

### 4. Graph Problem (Edges List)
```python
from contest_helper import RandomNumber, RandomDict

# Generate graphs with 5-10 nodes and random edges
graph_gen = RandomDict(
    key_generator=RandomNumber(1, 11),  # Node IDs 1-10
    value_generator=RandomList(RandomNumber(1, 11), RandomNumber(1, 4)),  # Adjacent nodes
    length=RandomNumber(5, 11)  # Node count
)

def print_graph(g):
    edges = []
    for node, neighbors in g.items():
        edges.extend(f"{node} {n}" for n in neighbors)
    return edges

generator = Generator(
    tests_generator=graph_gen,
    tests_count=5,
    input_printer=print_graph,
    output_printer=lambda _: ["1"]  # Dummy output
)
generator.run()
```

### 5. Combined Generators
```python
from contest_helper import CombineValues, RandomWord, RandomNumber

# Generate tests with multiple values per case
combined_gen = CombineValues([
    RandomWord(min_length=5, max_length=10),  # String
    RandomNumber(1, 100),                     # Number
    RandomNumber(0, 2)                        # Boolean-like
])

generator = Generator(
    tests_generator=combined_gen,
    tests_count=8,
    input_printer=lambda vals: [str(v) for v in vals],
    output_printer=lambda vals: [str(len(vals[0]) * vals[1])]  # String length repeated N times
)
generator.run()
```

## Grouped Test Generation Examples

### 1. Basic Grouped Tests
```python
from contest_helper import RandomNumber, Generator

# Different groups with different number ranges
generator = Generator(
    tests_generator={
        'small': RandomNumber(1, 11),      # 1-10
        'medium': RandomNumber(10, 101),   # 10-100
        'large': RandomNumber(100, 1001)   # 100-1000
    },
    tests_count={
        'small': 5,    # 5 small tests
        'medium': 3,   # 3 medium tests
        'large': 2     # 2 large tests
    },
    input_printer=lambda x: [str(x)],
    output_printer=lambda x: [str(x % 10)]  # Last digit
)
generator.run()
```

### 2. String Problems with Difficulty Groups
```python
from contest_helper import RandomWord, RandomSentence

generator = Generator(
    tests_generator={
        'easy': RandomWord(min_length=3, max_length=5),
        'medium': RandomWord(min_length=10, max_length=20),
        'hard': RandomSentence(min_length=3, max_length=5)  # Multi-word sentences
    },
    tests_count={
        'easy': 3,
        'medium': 2,
        'hard': 1
    },
    input_printer=lambda x: [x],
    output_printer=lambda x: [x.upper()]  # Convert to uppercase
)
generator.run()
```

### 3. Graph Problems with Increasing Complexity
```python
from contest_helper import RandomDict, RandomNumber, RandomList

generator = Generator(
    tests_generator={
        'trees': RandomDict(
            key_generator=RandomNumber(1, 6),
            value_generator=RandomList(RandomNumber(1, 6), 1),  # Trees have 1 parent
            length=5
        ),
        'graphs': RandomDict(
            key_generator=RandomNumber(1, 11),
            value_generator=RandomList(RandomNumber(1, 11), RandomNumber(1, 3)),
            length=10
        )
    },
    tests_count={
        'trees': 3,
        'graphs': 2
    },
    input_printer=lambda g: [f"{k}:{','.join(map(str,v))}" for k,v in g.items()],
    output_printer=lambda _: ["1"]  # Dummy output
)
generator.run()
```

### 4. Combined Groups with Custom Logic
```python
from contest_helper import CombineValues, RandomWord, RandomNumber

generator = Generator(
    tests_generator={
        'arithmetic': RandomNumber(1, 100),
        'strings': RandomWord(min_length=5, max_length=10),
        'mixed': CombineValues([
            RandomWord(min_length=3, max_length=5),
            RandomNumber(10, 20)
        ])
    },
    tests_count={
        'arithmetic': 4,
        'strings': 3,
        'mixed': 2
    },
    input_printer=lambda x: [str(x) if isinstance(x, int) else x],
    output_printer=lambda x: (
        [str(x*2)] if isinstance(x, int) 
        else [x[::-1]] if isinstance(x, str)
        else [x[0]*x[1]]  # string repeated N times
    )
)
generator.run()
```

### 5. Scientific Formatting with Groups
```python
from contest_helper import RandomNumber, Generator
import math

generator = Generator(
    tests_generator={
        'precision_low': RandomNumber(1.0, 5.0, 0.1),   # 1 decimal
        'precision_high': RandomNumber(1.0, 5.0, 0.001) # 3 decimals
    },
    tests_count={
        'precision_low': 3,
        'precision_high': 2
    },
    input_printer=lambda x: [f"{x:.3f}"],
    output_printer=lambda x: [f"{math.sin(x):.5e}"]  # Scientific notation
)
generator.run()
```

## Compiler Search Utility

The `ch-compilers` command allows you to search through available compilers and development tools. It supports flexible name matching and displays results in an easy-to-read table format.

### Basic Usage

```bash
ch-compilers <query>
```

### Examples

1. Search for Python compilers:
   ```bash
   ch-compilers python
   ```

2. Search for Java compilers (case insensitive):
   ```bash
   ch-compilers java
   ```

3. Search with a custom CSV data file:
   ```bash
   ch-compilers "c++"
   ```

### Expected Output

```
Found 3 matches for 'python':

+----+----------------+------------------------------------+
| #  | ID             | Name                               |
+====+================+====================================+
| 1  | python3_13     | Python 3.13.2                      |
+----+----------------+------------------------------------+
| 2  | pypy3_7_1_0    | Python 3.11 (PyPy 7.3.19)          |
+----+----------------+------------------------------------+
| 3  | python2_6      | Python 2.7                         |
+----+----------------+------------------------------------+
```

## Automated Testing Utility

The `ch-test` command provides comprehensive automated testing capabilities for competitive programming solutions.

### Key Features

- **Multi-format support**: Tests binary executables and interpreted scripts
- **Flexible execution**: Handles C/C++/Rust/Go binaries and Python/Bash scripts
- **Smart test discovery**: Automatically pairs input files with expected outputs
- **Custom validation**: Supports both direct comparison and checker programs
- **Detailed reporting**: Provides test-by-test results and summary statistics

### Basic Usage

```bash
ch-test ./solution [options]
```

### Common Options

| Option        | Description                          | Example                      |
|---------------|--------------------------------------|------------------------------|
| `-c`/`--checker` | Path to custom checker program     | `ch-test ./sol -c ./checker` |
| `-i`/`--interpreter` | Specify script interpreter      | `ch-test sol.py -i python3`  |
| `-t`/`--timeout` | Set execution timeout (seconds)  | `ch-test ./sol -t 5`         |

### Test File Structure

Organize tests in a `tests` directory with pairs of:
- Input files (any name without `.a` extension)
- Output files (same name with `.a` extension)

Example:
```
tests/
├── 001       # Test input
├── 001.a     # Expected output
├── 002       # Another test
└── 002.a     # Its expected output
```

### Execution Examples

1. **Test a C++ binary**:
```bash
ch-test ./solution_cpp
```

2. **Test a Python script**:
```bash
ch-test solution.py -i python3
```

3. **With custom checker**:
```bash
ch-test ./solution -c ./checker -t 3
```

4. **Test a Bash script**:
```bash
ch-test script.sh -i bash
```

### Output Format

```
[timestamp] INFO - Executing: 001
[timestamp] INFO - Result: PASSED (exact match)
[timestamp] ERROR - Result: FAILED
Expected: 42
Actual: 43

=== Test Execution Summary ===
Total tests run:      5
Passed:               3
Failed:               1
Execution errors:     1
Success rate:    60.00%
```

## Advanced Features

- **Group-based test generation**: Organize tests into logical groups
- **Custom scoring**: Differential scoring by test difficulty
- **Multi-language support**: English and Russian statements
- **Flexible configuration**: Override defaults via command line

## Requirements

- Python 3.10+
- Standard library only (no external dependencies)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
