#!/usr/bin/python3

import argparse
from contest_helper.cli.utils import load_template


def create_checker():
    """Creates a checker.py file from the template."""
    content = load_template('checker.py')

    with open('checker.py', 'w') as f:
        f.write(content)

    print("Created checker at checker.py")


def main():
    parser = argparse.ArgumentParser(description="Create checker.py from template")
    parser.parse_args()
    create_checker()


if __name__ == "__main__":
    main()
