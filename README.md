# Easy Random Test Case Generator Framework

A flexible framework for generating randomized test cases with input/output validation, designed for competitive
programming and algorithmic testing.

## Features

- **Type-safe generators** for all basic data types and structures
- **Automatic test validation** with `BadTestException` handling
- **Customizable I/O formatting** for any problem format
- **Sample test integration** alongside generated cases
- **Extensible architecture** for custom generators

## Installation

```bash
pip install contest-helper
```

## Quick Start

1. Using the CLI Tool

Initialize a new problem directory:

```bash
ch-start-problem path/to/problem --language en --checker
```

This creates:

- Problem statement (in specified language)
- Test generator template
- Metadata file
- Optional checker script

2. Generating Test Cases

```python
from contest_helper.basic import *


# Define your solution with validation
def word_count(text: str) -> int:
    if len(text) > 1000:
        raise BadTestException("Input too long")
    return len(text.split())


# Configure and run generator
Generator(
    solution=word_count,
    tests_generator=RandomSentence(min_length=1, max_length=20),
    tests_count=10
).run()
```

## Core Components

### Value Generators

| Generator        | Description                | Example                                 |
|------------------|----------------------------|-----------------------------------------|
| `Value`          | Basic value wrapper        | `Value(5)`                              |
| `Lambda`         | Custom generator functions | `Lambda(lambda: datetime.now())`        |
| `RandomValue`    | Random value from sequence | `RandomValue(['red', 'green', 'blue'])` |
| `RandomNumber`   | Numeric ranges             | `RandomNumber(1, 100)`                  |
| `RandomWord`     | Random strings             | `RandomWord()`                          |
| `RandomSentence` | Natural language-like text | `RandomSentence()`                      |
| `RandomList`     | Random sequences           | `RandomList(gen, 5)`                    |
| `RandomSet`      | Random sequences           | `RandomSet(gen, 5)`                     |
| `RandomDict`     | Key-value pairs            | `RandomDict(kgen, vgen, 3)`             |

### Test Validation

```python
def solution(input_data):
    # Validate input before processing
    if is_invalid(input_data):
        raise BadTestException("Validation failed")

    # Normal processing...
```

### I/O Configuration

```python
Generator(
    input_parser=lambda lines: parse_custom_format(lines),
    input_printer=lambda obj: format_as_text(obj),
    output_printer=lambda result: [str(result)]
)
```

## Advanced Usage

### Dynamic Test Generation

```python
dynamic_gen = RandomDict(
    key_generator=RandomWord(),
    value_generator=RandomList(
        RandomNumber(1, 100),
        length=RandomNumber(2, 5)
    ),
    length=RandomNumber(3, 10)
)
```

### Custom Generators

```python
class RandomGraph(Value[Dict]):
    def __init__(self, node_count: Value[int]):
        self.node_count = node_count

    def __call__(self) -> Dict:
        nodes = [f"node{i}" for i in range(self.node_count())]
        return {
            n: random.sample(nodes, k=random.randint(1, len(nodes)))
            for n in nodes
        }
```

## Directory Structure

Generated tests follow this structure:

```
tests/
├── sample01    # Sample input
├── sample01.a  # Sample output
├── 01          # Generated test input
├── 01.a        # Generated test output
└── ...
```

## Best Practices

1. **Validation**: Always validate inputs in your solution function
2. **Descriptive Messages**: Provide clear `BadTestException` messages
3. **Generator Composition**: Build complex types from simple generators
4. **Test Diversity**: Configure generators to produce edge cases
5. **Performance**: Avoid expensive operations in validation

## License

MIT License - Free for commercial and personal use
