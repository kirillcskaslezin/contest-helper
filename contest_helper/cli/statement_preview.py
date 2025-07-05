"""
Statement Preview Generator
==========================

A utility for automatically updating problem statements with test case examples.
Uses Python's logging module for output reporting.
"""

import os
import re
import logging
from contest_helper.cli.utils import load_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def find_sample_files(directory):
    """Finds all sample input/output files in the tests directory."""
    samples = []
    tests_dir = os.path.join(directory, "tests")

    if not os.path.exists(tests_dir):
        logger.warning(f"Tests directory not found at {tests_dir}")
        return samples

    for file in os.listdir(tests_dir):
        match = re.match(r'sample(\d+)', file)
        if match and not file.endswith('.a'):
            sample_num = match.group(1)
            input_path = os.path.join(tests_dir, file)
            output_path = os.path.join(tests_dir, f"{file}.a")

            if os.path.exists(output_path):
                samples.append((sample_num, input_path, output_path))
            else:
                logger.warning(f"Missing output file for sample {sample_num}")

    samples.sort(key=lambda x: int(x[0]))
    logger.debug(f"Found {len(samples)} valid samples")
    return samples


def read_file_contents(file_path):
    """Reads and sanitizes file content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except IOError as e:
        logger.error(f"Failed to read file {file_path}: {str(e)}")
        raise


def render_template(template, input_content, output_content, sample_num):
    """Formats test case into markdown using the template."""
    try:
        return template.replace('{{input}}', input_content.replace('\n', '<br />')) \
            .replace('{{output}}', output_content.replace('\n', '<br />')) \
            .replace('{{num}}', sample_num)
    except AttributeError as e:
        logger.error(f"Template rendering failed: {str(e)}")
        raise


def process_samples(samples, template):
    """Processes all samples into formatted markdown."""
    result = ''
    for num, input_path, output_path in samples:
        try:
            input_content = read_file_contents(input_path)
            output_content = read_file_contents(output_path)
            result += '\n\n' + render_template(template, input_content, output_content, num)
        except Exception as e:
            logger.error(f"Failed to process sample {num}: {str(e)}")
            continue
    return result


def update_statements_file(directory, template, samples, output_path=None):
    """Updates the problem statement with formatted examples."""
    if output_path is None:
        output_path = os.path.join(directory, "statement.md")

    try:
        # Read existing content
        with open(os.path.join(directory, "statement.md"), 'r', encoding='utf-8') as f:
            existing_content = f.read()

        # Generate examples section
        examples_section = process_samples(samples, template)

        # Combine with existing content
        new_content = existing_content + examples_section

        # Write updated file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        logger.info(f"Successfully updated statement file at {output_path}")
    except IOError as e:
        logger.error(f"Failed to update statement file: {str(e)}")
        raise


def main():
    """Command-line interface for the statement preview generator."""
    import argparse

    parser = argparse.ArgumentParser(description='Adds examples to statements.md')
    parser.add_argument('directory', help='Path to problem directory')
    parser.add_argument('--lang', choices=['en', 'ru'], default='ru',
                      help='Template language (en/ru)')
    parser.add_argument('-o', '--output',
                      help='Custom output file path')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Enable verbose logging')
    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Load template
    try:
        template = load_template(os.path.join('samples', f'{args.lang}.md'))
        logger.debug("Template loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Template loading failed: {e}")
        return

    # Find samples
    samples = find_sample_files(args.directory)
    if not samples:
        logger.error("No valid samples found - aborting")
        return

    logger.info(f"Processing {len(samples)} samples")

    # Update statement file
    try:
        update_statements_file(args.directory, template, samples, args.output)
        logger.info("Operation completed successfully")
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        return


if __name__ == "__main__":
    main()