"""
Problem Package Generator

This script creates a zip archive of a programming problem directory and updates
the meta.json configuration file with specified settings. It handles:
- Problem statements
- Test cases
- Checker configuration
- Solution limits
- Input/output settings
- File categorization
- Section visibility controls

Usage:
    python combine.py <directory> [options]
"""

import argparse
import zipfile
import os
import json
import logging
from glob import glob
from datetime import datetime

# Configure logging
def setup_logging():
    """Configure logging with both file and console output."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"combine_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Main function that parses arguments and coordinates the packaging process."""
    setup_logging()
    logging.info("Starting problem package generation")

    parser = argparse.ArgumentParser(description="Combine problem in zip-file for import")

    # Basic arguments
    parser.add_argument('directory', help='Path to problem directory')

    # File categorization arguments
    parser.add_argument('--checker-files', nargs='*', default=[],
                      help='Files for checkerSettings/checkerFiles')
    parser.add_argument('--compile-files', nargs='*', default=[],
                      help='Files for includeForCompileFiles')
    parser.add_argument('--run-files', nargs='*', default=[],
                      help='Files for includeForRunFiles')
    parser.add_argument('--post-files', nargs='*', default=[],
                      help='Files for postProcessFiles')
    parser.add_argument('--solutions', nargs='*', default=[],
                      help='Author solutions in format "compiler_id:file_path"')

    # Solution limits arguments
    parser.add_argument('--time-limit', type=int,
                      help='Time limit in milliseconds')
    parser.add_argument('--memory-limit', type=int,
                      help='Memory limit in bytes')
    parser.add_argument('--output-limit', type=int,
                      help='Output limit in bytes')
    parser.add_argument('--idleness-limit', type=int,
                      help='Idleness limit in milliseconds')

    # I/O configuration arguments
    parser.add_argument('--input-file',
                      help='Input file name')
    parser.add_argument('--output-file',
                      help='Output file name')
    parser.add_argument('--redirect-stdin', action='store_true',
                      help='Enable stdin redirection')
    parser.add_argument('--redirect-stdout', action='store_true',
                      help='Enable stdout redirection')

    # Section visibility arguments
    parser.add_argument('--hide-limits', action='store_false', dest='show_limits',
                      help='Hide limits section (default: show)')
    parser.add_argument('--hide-io', action='store_false', dest='show_io',
                      help='Hide input/output section (default: show)')
    parser.add_argument('--hide-samples', action='store_false', dest='show_samples',
                      help='Hide samples section (default: show)')

    args = parser.parse_args()

    logging.info(f"Processing directory: {args.directory}")
    logging.debug(f"Command line arguments: {vars(args)}")

    try:
        combine(
            directory=args.directory,
            checker_files=args.checker_files,
            compile_files=args.compile_files,
            run_files=args.run_files,
            post_files=args.post_files,
            solutions=args.solutions,
            time_limit=args.time_limit,
            memory_limit=args.memory_limit,
            output_limit=args.output_limit,
            idleness_limit=args.idleness_limit,
            input_file=args.input_file,
            output_file=args.output_file,
            redirect_stdin=args.redirect_stdin,
            redirect_stdout=args.redirect_stdout,
            show_limits=args.show_limits,
            show_io=args.show_io,
            show_samples=args.show_samples
        )
        logging.info("Problem package created successfully")
    except Exception as e:
        logging.error(f"Failed to create problem package: {str(e)}", exc_info=True)
        raise


def combine(directory, **kwargs):
    """
    Main function to package the problem directory into a zip archive.

    Args:
        directory (str): Path to the problem directory
        **kwargs: Various configuration options including:
            - checker_files: List of checker files
            - compile_files: List of compilation files
            - solutions: List of author solutions
            - time_limit: Time limit in ms
            - memory_limit: Memory limit in bytes
            - Various I/O and visibility settings
    """
    zip_filename = directory + ".zip"
    logging.info(f"Creating archive: {zip_filename}")

    # Remove existing zip file if present
    if os.path.isfile(zip_filename):
        logging.warning(f"Removing existing zip file: {zip_filename}")
        os.unlink(zip_filename)

    meta_path = os.path.join(directory, "meta.json")
    if not os.path.exists(meta_path):
        error_msg = f"meta.json not found in {directory}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Load existing meta.json
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        logging.debug("Loaded meta.json successfully")
    except Exception as e:
        logging.error(f"Failed to load meta.json: {str(e)}")
        raise

    # Initialize problemMetadata if not present
    if 'problemMetadata' not in meta:
        logging.debug("Initializing problemMetadata in meta.json")
        meta['problemMetadata'] = {}

    # Process all components
    logging.info("Processing problem components")
    process_statement(directory, meta)
    update_limits(meta, kwargs)
    update_io_settings(meta, kwargs)
    update_section_visibility(meta, kwargs)

    if kwargs.get('checker_files'):
        logging.info(f"Updating checker settings with files: {kwargs['checker_files']}")
        update_checker_settings(meta, kwargs['checker_files'])

    if kwargs.get('solutions'):
        logging.info(f"Adding solutions: {kwargs['solutions']}")
        update_solutions(meta, kwargs['solutions'])

    update_file_categories(meta, kwargs)
    process_tests(directory, meta)

    # Save updated meta.json
    try:
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logging.debug("Saved updated meta.json")
    except Exception as e:
        logging.error(f"Failed to save meta.json: {str(e)}")
        raise

    # Create zip archive
    try:
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add all files recursively
            file_count = 0
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=directory)
                    zipf.write(file_path, arcname)
                    file_count += 1

            logging.info(f"Added {file_count} files to archive")
    except Exception as e:
        logging.error(f"Failed to create zip archive: {str(e)}")
        raise


def update_section_visibility(meta, kwargs):
    """
    Update section visibility settings in meta.json.

    Args:
        meta (dict): The meta.json content
        kwargs (dict): Contains visibility flags:
            - show_limits: Whether to show limits section
            - show_io: Whether to show I/O section
            - show_samples: Whether to show samples section
    """
    if 'sectionVisibility' not in meta['problemMetadata']:
        meta['problemMetadata']['sectionVisibility'] = {
            "showLimits": True,
            "showIo": True,
            "showSamples": True
        }
        logging.debug("Initialized sectionVisibility with default values")

    # Only update values that were explicitly passed
    visibility_updates = []
    if 'show_limits' in kwargs:
        meta['problemMetadata']['sectionVisibility']['showLimits'] = kwargs['show_limits']
        visibility_updates.append(f"showLimits={kwargs['show_limits']}")
    if 'show_io' in kwargs:
        meta['problemMetadata']['sectionVisibility']['showIo'] = kwargs['show_io']
        visibility_updates.append(f"showIo={kwargs['show_io']}")
    if 'show_samples' in kwargs:
        meta['problemMetadata']['sectionVisibility']['showSamples'] = kwargs['show_samples']
        visibility_updates.append(f"showSamples={kwargs['show_samples']}")

    if visibility_updates:
        logging.debug(f"Updated section visibility: {', '.join(visibility_updates)}")


def process_statement(directory, meta):
    """
    Process problem statement from statement.md and update meta.json.

    Args:
        directory (str): Problem directory path
        meta (dict): meta.json content to update
    """
    statement_path = os.path.join(directory, "statement.md")
    if os.path.exists(statement_path):
        logging.debug(f"Processing statement file: {statement_path}")
        try:
            with open(statement_path, 'r', encoding='utf-8') as f:
                statement_lines = f.read().split('\n')

            # Extract problem name from first line (header)
            if statement_lines and statement_lines[0].startswith('#'):
                problem_name = statement_lines[0][1:].strip()
                if 'names' not in meta['problemMetadata']:
                    meta['problemMetadata']['names'] = {}
                meta['problemMetadata']['names']['ru'] = problem_name
                logging.debug(f"Set problem name: {problem_name}")

            # The rest of the content becomes the legend
            legend = '\n'.join(statement_lines[1:]).strip()
            if 'statements' not in meta['problemMetadata']:
                meta['problemMetadata']['statements'] = [{}]
            if len(meta['problemMetadata']['statements']) == 0:
                meta['problemMetadata']['statements'].append({})
            if 'rawStatement' not in meta['problemMetadata']['statements'][0]:
                meta['problemMetadata']['statements'][0]['rawStatement'] = {}
            meta['problemMetadata']['statements'][0]['rawStatement']['legend'] = legend
            logging.debug("Updated problem statement legend")
        except Exception as e:
            logging.warning(f"Failed to process statement.md: {str(e)}")
    else:
        logging.warning("No statement.md file found in directory")


def update_limits(meta, kwargs):
    """
    Update solution limits in meta.json.

    Args:
        meta (dict): meta.json content to update
        kwargs (dict): Contains limit values:
            - time_limit: Time limit in ms
            - memory_limit: Memory limit in bytes
            - output_limit: Output limit in bytes
            - idleness_limit: Idleness limit in ms
    """
    limits = {}
    if kwargs.get('time_limit'):
        limits['timeLimitMillis'] = kwargs['time_limit']
    if kwargs.get('memory_limit'):
        limits['memoryLimit'] = kwargs['memory_limit']
    if kwargs.get('output_limit'):
        limits['outputLimit'] = kwargs['output_limit']
    if kwargs.get('idleness_limit'):
        limits['idlenessLimitMillis'] = kwargs['idleness_limit']

    if limits:
        if 'solutionLimits' not in meta['problemMetadata']:
            meta['problemMetadata']['solutionLimits'] = {}
        meta['problemMetadata']['solutionLimits'].update(limits)
        logging.info(f"Updated solution limits: {limits}")


def update_io_settings(meta, kwargs):
    """
    Update input/output settings in meta.json.

    Args:
        meta (dict): meta.json content to update
        kwargs (dict): Contains I/O settings:
            - input_file: Input file name
            - output_file: Output file name
            - redirect_stdin: Whether to redirect stdin
            - redirect_stdout: Whether to redirect stdout
    """
    io_settings = {}
    if kwargs.get('input_file'):
        io_settings['inputFile'] = kwargs['input_file']
    if kwargs.get('output_file'):
        io_settings['outputFile'] = kwargs['output_file']
    if kwargs.get('redirect_stdin') is not None:
        io_settings['redirectStdin'] = kwargs['redirect_stdin']
    if kwargs.get('redirect_stdout') is not None:
        io_settings['redirectStdout'] = kwargs['redirect_stdout']

    if io_settings:
        if 'fileSet' not in meta['problemMetadata']:
            meta['problemMetadata']['fileSet'] = {}
        meta['problemMetadata']['fileSet'].update(io_settings)
        logging.info(f"Updated I/O settings: {io_settings}")


def update_checker_settings(meta, checker_files):
    """
    Update checker configuration in meta.json.

    Args:
        meta (dict): meta.json content to update
        checker_files (list): List of checker files
    """
    if 'checkerSettings' not in meta['problemMetadata']:
        meta['problemMetadata']['checkerSettings'] = {}

    meta['problemMetadata']['checkerSettings']['checkerType'] = 'EJUDGE_EXITCODE'
    meta['problemMetadata']['checkerSettings']['checkerFiles'] = checker_files
    logging.info(f"Configured checker with type EJUDGE_EXITCODE and files: {checker_files}")


def update_solutions(meta, solutions):
    """
    Update author solutions in meta.json.

    Args:
        meta (dict): meta.json content to update
        solutions (list): List of solutions in "compiler_id:file_path" format
    """
    solution_list = []
    for solution in solutions:
        if ':' in solution:
            compiler_id, file_path = solution.split(':', 1)
            solution_list.append({
                "compilerId": compiler_id,
                "sourcePath": file_path,
                "comment": "",
                "verdict": "OK",
                "testNumber": 0
            })
            logging.debug(f"Added solution: {compiler_id}:{file_path}")

    if solution_list:
        meta['problemMetadata']['solutions'] = solution_list
        logging.info(f"Added {len(solution_list)} author solutions")


def update_file_categories(meta, kwargs):
    """
    Update file categorization in meta.json.

    Args:
        meta (dict): meta.json content to update
        kwargs (dict): Contains file lists:
            - compile_files: Files needed for compilation
            - run_files: Files needed for execution
            - post_files: Post-processing files
    """
    if kwargs.get('compile_files'):
        meta['problemMetadata']['includeForCompileFiles'] = kwargs['compile_files']
        logging.debug(f"Added compile files: {kwargs['compile_files']}")

    if kwargs.get('run_files'):
        meta['problemMetadata']['includeForRunFiles'] = kwargs['run_files']
        logging.debug(f"Added run files: {kwargs['run_files']}")

    if kwargs.get('post_files'):
        if 'postProcessFiles' not in meta['problemMetadata']:
            meta['problemMetadata']['postProcessFiles'] = []
        meta['problemMetadata']['postProcessFiles'].extend(kwargs['post_files'])
        logging.debug(f"Added post-processing files: {kwargs['post_files']}")


def process_tests(directory, meta):
    """
    Process test cases and update meta.json.

    Args:
        directory (str): Problem directory path
        meta (dict): meta.json content to update
    """
    tests_dir = os.path.join(directory, "tests")
    if os.path.exists(tests_dir) and os.path.isdir(tests_dir):
        logging.info(f"Processing tests from directory: {tests_dir}")
        # Find all test files (excluding .a answer files)
        test_files = sorted(glob(os.path.join(tests_dir, '*[!.a]')))
        matched_tests = []

        # Match each test with its answer file
        for test_file in test_files:
            if test_file.endswith('.a'):
                continue

            answer_file = test_file + '.a'
            if os.path.exists(answer_file):
                matched_tests.append({
                    "inputPath": os.path.relpath(test_file, start=directory),
                    "answerPath": os.path.relpath(answer_file, start=directory),
                    "inputExists": True,
                    "answerExists": True
                })

        if matched_tests:
            logging.info(f"Found {len(matched_tests)} test cases")
            if 'testSets' not in meta['problemMetadata']:
                meta['problemMetadata']['testSets'] = [{}, {}]

            # Separate sample tests from main tests
            sample_tests = [t for t in matched_tests if 'sample' in t['inputPath']]
            if sample_tests:
                logging.info(f"Found {len(sample_tests)} sample tests")
                if len(meta['problemMetadata']['testSets']) > 0:
                    if meta['problemMetadata']['testSets'][0] is None:
                        meta['problemMetadata']['testSets'][0] = {}
                    meta['problemMetadata']['testSets'][0]['matchedTests'] = sample_tests

            main_tests = [t for t in matched_tests if 'sample' not in t['inputPath']]
            if main_tests:
                logging.info(f"Found {len(main_tests)} main tests")
                if len(meta['problemMetadata']['testSets']) > 1:
                    if meta['problemMetadata']['testSets'][1] is None:
                        meta['problemMetadata']['testSets'][1] = {}
                    meta['problemMetadata']['testSets'][1]['matchedTests'] = main_tests
    else:
        logging.warning("No tests directory found")


if __name__ == '__main__':
    main()