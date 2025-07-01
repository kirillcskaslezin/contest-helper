#!/usr/bin/python3

import json
import sys
import traceback

# Global configuration constants
MAX_VALUE = 0  # Maximum possible score for the test suite
GROUPS = []  # Group sizes for group-based scoring
DIFFERENT = False  # Flag for differential scoring mode
BY_GROUPS = False  # Flag for group-based scoring mode


def main():
    """Main function that processes test results and calculates a score.

    Reads JSON test data from stdin, processes it according to configuration flags,
    and outputs the calculated score to stdout. Handles all exceptions gracefully.
    """
    try:
        # Load and parse input test data from stdin
        testData = json.load(sys.stdin)["tests"]

        # Extract only important fields from each test case
        important = ["testName", "verdict", "testsetName"]
        tests = [{j: i.get(j, None) for j in important} for i in testData]

        # Sort tests by name for consistent processing
        tests.sort(key=lambda x: x["testName"])
        tests_count = len(tests)

        # Filter out duplicate test names and sample tests, then collect verdicts
        tests_statuses = [
            tests[i]["verdict"] == "ok"
            for i in range(tests_count)
            if (
                    (i == 0 or tests[i]["testName"] != tests[i - 1]["testName"])
                    and tests[i]["testsetName"] != 'samples'
            )
        ]

        passed_tests_count = sum(tests_statuses)

        # Calculate final mark based on configuration
        if tests_count == 0:
            mark = 0  # Edge case: no tests
        elif DIFFERENT:
            if BY_GROUPS:
                # Group-based scoring logic
                mark = 0
                for n in GROUPS:
                    if all(tests[:n]):
                        mark = round(MAX_VALUE * n / tests_count)
                    else:
                        break
            else:
                # Differential scoring based on passed tests percentage
                mark = round(MAX_VALUE * passed_tests_count / tests_count)
        else:
            # All-or-nothing scoring
            mark = 0 if passed_tests_count < tests_count else MAX_VALUE

        # Output the final calculated mark
        sys.stdout.write(f'{mark}\n')

    except Exception:
        # Handle any processing errors gracefully
        sys.stdout.write('0\n')  # Default fail score
        traceback.print_exc(file=sys.stderr)  # Debug info to stderr
        sys.exit(5)  # Exit with error code


if __name__ == '__main__':
    main()
