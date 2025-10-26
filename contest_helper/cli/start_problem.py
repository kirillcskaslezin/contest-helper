"""
Problem Starter Utility
======================

A command-line tool for initializing programming problem directories with template files.
Automates creation of standard problem components including statements, generators, and checkers.
"""

import argparse
import re
from os import mkdir, path
from contest_helper.cli.utils import load_template


def main():
    """
    Main entry point for the problem starter utility.

    Sets up command-line argument parsing and initiates problem directory creation.
    """
    parser = argparse.ArgumentParser(
        description='Initialize a new programming problem directory with template files',
        epilog='Example: python start_problem.py my_problem -l ru -c'
    )
    parser.add_argument(
        'directory',
        help='name/path of directory to create for the problem'
    )
    parser.add_argument(
        '--language', '-l',
        default='en',
        choices=['en', 'ru'],
        help='language for problem statement (default: en)'
    )
    parser.add_argument(
        '--checker', '-c',
        action="store_true",
        help='include a checker.py file for solution validation'
    )
    parser.add_argument(
        '--input-type', '-i',
        default='text', choices=['text', 'binary'],
        help='input data type in the generated generator template (text|binary)'
    )
    parser.add_argument(
        '--output-type', '-o',
        default='text', choices=['text', 'binary'],
        help='output data type in the generated generator template (text|binary)'
    )
    args = parser.parse_args()

    process(args.directory, args.language, args.checker, args.input_type, args.output_type)


def process(directory, language, need_checker=False, input_type='text', output_type='text'):
    """
    Creates problem directory structure with template files.

    Args:
        directory (str): Path to the problem directory
        language (str): Language code for statement ('en' or 'ru')
        need_checker (bool): Whether to include checker template
        input_type (str): Input adapter type ('text' or 'binary')
        output_type (str): Output adapter type ('text' or 'binary')

    Raises:
        FileExistsError: If target directory already exists
        IOError: If template files cannot be accessed
    """
    try:
        # Create root problem directory
        mkdir(directory)

        # Problem statement (language-specific)
        with open(path.join(directory, 'statement.md'), 'w') as file:
            file.write(load_template(path.join('statements', f'{language}.md')))

        # Test case generator template with adapter substitution
        gen_tpl = load_template('generator.py')

        # Decide adapter class instantiations
        input_adapter_str = 'MyInputAdapter()' if input_type == 'text' else 'MyBinInputAdapter()'
        output_adapter_str = 'MyOutputAdapter()' if output_type == 'text' else 'MyBinOutputAdapter()'

        gen_tpl = gen_tpl.replace('__INPUT_ADAPTER__', input_adapter_str)
        gen_tpl = gen_tpl.replace('__OUTPUT_ADAPTER__', output_adapter_str)

        if input_type == 'text':
            # Remove MyBinInputAdapter class block
            gen_tpl = re.sub(r'class MyBinInputAdapter[\s\S]+?(?=class )', '', gen_tpl)
        else:
            # Remove MyInputAdapter class block
            gen_tpl = re.sub(r'class MyInputAdapter[\s\S]+?(?=class )', '', gen_tpl)

        if output_type == 'text':
            # Remove MyBinOutputAdapter class block, keep 'def validator' and subsequent code intact
            gen_tpl = re.sub(r'class MyBinOutputAdapter[\s\S]+?(?=def validator)', '', gen_tpl)
        else:
            # Remove MyOutputAdapter class block
            gen_tpl = re.sub(r'class MyOutputAdapter[\s\S]+?(?=class )', '', gen_tpl)

        with open(path.join(directory, 'generator.py'), 'w') as file:
            file.write(gen_tpl)

        # Problem metadata
        with open(path.join(directory, 'meta.json'), 'w') as file:
            file.write(load_template('meta.json'))

        # Optional checker file
        if need_checker:
            with open(path.join(directory, 'checker.py'), 'w') as file:
                file.write(load_template('checker.py'))

        print(f"Problem directory '{directory}' created successfully!")

    except FileExistsError:
        print(f"Error: Directory '{directory}' already exists")
        raise
    except Exception as e:
        print(f"Error creating problem: {str(e)}")
        raise


if __name__ == "__main__":
    main()
