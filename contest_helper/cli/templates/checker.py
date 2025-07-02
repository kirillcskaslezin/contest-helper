"""
Problem Checker System (ejudge-compatible)
-----------------------------------------
A template for creating problem checkers that validate solutions against test cases.
Uses standard ejudge exit codes for compatibility with programming contest systems.
"""

import sys
import traceback

# ejudge-compatible exit codes
OK = 0  # Solution is correct
PE = 4  # Presentation Error (formatting issues)
WA = 5  # Wrong Answer (incorrect solution)
CRASH = 6  # Checker crashed (internal error)


def parse_input(file):
    """
    Parse input test case file (optional).
    """
    ...


def parse_output(file):
    """
    Parse solution's output file.
    """
    return list(map(str.strip, file))


def parse_pattern(file):
    """
    Parse expected output file.
    """
    return parse_input(file)


def check(input_data, output_data, pattern_data):
    """
    Core validation logic comparing solution against expected results.

    Raises:
        AssertionError: For Wrong Answer (WA) with descriptive message
        Exception: For other validation failures
    """

    assert output_data == pattern_data, 'Wrong answer'


def main():
    """
    Main checker workflow:
    1. Reads input files
    2. Parses data
    3. Validates solution
    4. Returns appropriate exit code

    Expected command-line arguments:
    [input_file] [output_file] [pattern_file]

    Exit codes:
        0 - OK (Accepted)
        4 - PE (Presentation Error)
        5 - WA (Wrong Answer)
        6 - CRASH (Checker failure)
    """
    if len(sys.argv) != 4:
        print("Usage: python checker.py [input] [output] [pattern]")
        sys.exit(CRASH)

    try:
        # Read and parse all input files
        data = []
        for filename, parser in zip(sys.argv[1:], (parse_input, parse_output, parse_pattern)):
            with open(filename) as file:
                data.append(parser(file))

        # Execute validation
        check(*data)
        sys.exit(OK)

    except IOError as error:
        print(f"File error: {error}")
        sys.exit(PE)
    except AssertionError as error:
        print(f"Wrong answer: {error}")
        sys.exit(WA)
    except Exception as error:
        print(f"Checker error: {error}")
        traceback.print_exc()
        sys.exit(CRASH)


if __name__ == "__main__":
    main()