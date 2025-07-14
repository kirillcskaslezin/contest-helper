"""
Enhanced Test Runner Script

A robust automated testing framework that supports:
1. Execution of binary executables and interpreted scripts
2. Multiple test case formats
3. Custom checker programs for advanced validation
4. Detailed test statistics and logging

Features:
- Supports binaries (C/C++/Rust/Go compiled)
- Supports interpreted scripts (Python, Bash, etc.)
- Flexible test case organization
- Timeout protection for test cases
- Comprehensive result reporting

Usage Examples:
  ./test_runner.py ./solution_binary
  ./test_runner.py solution.py -i python
  ./test_runner.py script.sh -i bash -c ./checker
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime


class TestRunner:
    """
    Core test execution engine.

    Handles all aspects of test execution including:
    - Test case discovery
    - Program execution with timeout protection
    - Result validation (direct or via checker)
    - Comprehensive reporting
    """

    def __init__(self, timeout=1):
        """Initialize test runner with default timeout (seconds)."""
        self.timeout = timeout
        self.logger = self._setup_logging()
        self.stats = {
            'total': 0,  # Total tests discovered
            'passed': 0,  # Tests that passed validation
            'failed': 0,  # Tests that failed validation
            'errors': 0  # Tests with execution errors
        }

    def _setup_logging(self):
        """Configure logging system with console output."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        return logging.getLogger(__name__)

    def _validate_executable(self, path, name):
        """
        Validate that a file exists and is executable.

        Args:
            path: Filesystem path to validate
            name: Human-readable name for error messages

        Raises:
            FileNotFoundError: If path doesn't exist
            PermissionError: If path isn't executable
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file '{path}' not found")
        if not os.access(path, os.X_OK):
            raise PermissionError(f"{name} file '{path}' is not executable")

    def _find_test_cases(self, tests_dir='tests'):
        """
        Discover test cases in the specified directory.

        Test format:
        - Input files: any file without .a extension
        - Output files: corresponding .a files

        Args:
            tests_dir: Directory containing test files

        Returns:
            List of absolute paths to test input files

        Raises:
            FileNotFoundError: If directory is empty or doesn't exist
        """
        if not os.path.exists(tests_dir):
            raise FileNotFoundError(f"Test directory '{tests_dir}' not found")

        test_files = sorted([
            os.path.join(tests_dir, f)
            for f in os.listdir(tests_dir)
            if os.path.isfile(os.path.join(tests_dir, f)) and not f.endswith('.a')
        ])

        if not test_files:
            raise FileNotFoundError(f"No test files found in '{tests_dir}'")

        return test_files

    def _run_checker(self, checker_path, test_input, program_output, expected_output):
        """
        Execute validation checker program.

        Args:
            checker_path: Path to checker executable
            test_input: Original test input file
            program_output: File containing program's output
            expected_output: File containing expected output

        Returns:
            Tuple: (bool status, str message)
        """
        checker_result = subprocess.run(
            [checker_path, test_input, program_output, expected_output],
            capture_output=True,
            text=True
        )

        if checker_result.returncode == 0:
            return (True, "PASSED (via checker)")

        # Build detailed failure message
        details = []
        if checker_result.stdout:
            details.append(f"Checker output:\n{checker_result.stdout}")
        if checker_result.stderr:
            details.append(f"Checker error:\n{checker_result.stderr}")
        return (False, "FAILED\n" + "\n".join(details))

    def _compare_outputs(self, program_output, expected_output):
        """
        Perform direct output comparison.

        Args:
            program_output: File containing program output
            expected_output: File containing expected output

        Returns:
            Tuple: (bool status, str message)
        """
        with open(program_output, 'r') as outfile:
            actual = outfile.read().strip()

        with open(expected_output, 'r') as outfile:
            expected = outfile.read().strip()

        if actual == expected:
            return (True, "PASSED (exact match)")
        return (False, f"FAILED\nExpected:\n{expected}\nActual:\n{actual}")

    def _execute_test(self, solution_path, test_input, test_output,
                      checker_path=None, interpreter_path=None):
        """
        Execute a single test case with full error handling.

        Args:
            solution_path: Path to solution program
            test_input: Path to test input file
            test_output: Path to expected output file
            checker_path: Optional path to checker program
            interpreter_path: Optional interpreter (e.g. 'python')

        Returns:
            Tuple: (bool status, str message)
        """
        # Unique output filename using timestamp
        program_output = f"program_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tmp"

        try:
            # Build execution command
            command = []
            if interpreter_path:
                command.append(interpreter_path)
            command.append(solution_path)

            # Execute with input/output redirection
            with open(test_input, 'r') as infile, open(program_output, 'w') as outfile:
                process = subprocess.run(
                    command,
                    stdin=infile,
                    stdout=outfile,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout
                )

            # Log any errors from the program
            if process.stderr:
                self.logger.debug("Program stderr:\n%s", process.stderr)

            # Validate results
            if checker_path:
                return self._run_checker(checker_path, test_input, program_output, test_output)
            return self._compare_outputs(program_output, test_output)

        except subprocess.TimeoutExpired:
            return (False, "TIMEOUT (program exceeded time limit)")
        except Exception as e:
            return (False, f"EXECUTION ERROR: {str(e)}")
        finally:
            # Clean up temporary files
            if os.path.exists(program_output):
                os.remove(program_output)

    def run_tests(self, solution_path, checker_path=None, interpreter_path=None):
        """
        Execute complete test suite.

        Args:
            solution_path: Path to solution program
            checker_path: Optional path to checker program
            interpreter_path: Optional interpreter command
        """
        try:
            # Validate inputs
            self._validate_executable(solution_path, "Solution")
            if checker_path:
                self._validate_executable(checker_path, "Checker")

            # Discover and process test cases
            test_files = self._find_test_cases()
            for test_input in test_files:
                test_output = test_input + '.a'
                self.stats['total'] += 1

                if not os.path.exists(test_output):
                    self.logger.warning("Missing expected output for '%s'", test_input)
                    continue

                self.logger.info("Executing: %s", os.path.basename(test_input))
                status, message = self._execute_test(
                    solution_path,
                    test_input,
                    test_output,
                    checker_path,
                    interpreter_path
                )

                # Update statistics
                if status:
                    self.stats['passed'] += 1
                    self.logger.info("Result: %s", message)
                else:
                    self.stats['failed'] += 1
                    self.logger.error("Result: %s", message)

            # Generate final report
            self._report_stats()

        except Exception as e:
            self.logger.error("Fatal error: %s", str(e))
            sys.exit(1)

    def _report_stats(self):
        """Generate comprehensive test execution report."""
        self.logger.info("\n=== Test Execution Summary ===")
        self.logger.info(f"Total tests run: {self.stats['total']:>6}")
        self.logger.info(f"Passed: {self.stats['passed']:>15}")
        self.logger.info(f"Failed: {self.stats['failed']:>15}")
        self.logger.info(f"Execution errors: {self.stats['errors']:>5}")

        if self.stats['total'] > 0:
            success_rate = (self.stats['passed'] / self.stats['total']) * 100
            self.logger.info(f"Success rate: {success_rate:>8.2f}%")


def main():
    """Command line interface for the test runner."""
    parser = argparse.ArgumentParser(
        description="Advanced Automated Test Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example usages:\n"
               "  ./test_runner.py ./solution\n"
               "  ./test_runner.py script.py -i python3\n"
               "  ./test_runner.py program -t 5 -c ./checker"
    )

    # Required arguments
    parser.add_argument(
        "solution",
        help="Path to solution program (binary or script)"
    )

    # Optional arguments
    parser.add_argument(
        "-c", "--checker",
        help="Path to custom checker program"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=1,
        help="Maximum execution time per test (seconds)"
    )
    parser.add_argument(
        "-i", "--interpreter",
        help="Interpreter for script solutions (e.g. 'python3', 'bash')"
    )

    args = parser.parse_args()

    # Initialize and run test suite
    runner = TestRunner(timeout=args.timeout)
    runner.run_tests(
        solution_path=args.solution,
        checker_path=args.checker,
        interpreter_path=args.interpreter
    )


if __name__ == '__main__':
    main()
