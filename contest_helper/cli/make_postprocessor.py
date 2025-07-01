#!/usr/bin/python3

import argparse
from contest_helper.cli.utils import load_template


def create_custom_postprocessor(max_value, groups=None, different=False, by_groups=False):
    """Creates a customized postprocessor.py with specified parameters.

    Args:
        max_value: Maximum score value
        groups: List of group sizes (comma-separated)
        different: Enable differential scoring
        by_groups: Enable group-based scoring
    """

    content = load_template('postprocessor.py')

    # Prepare groups list
    groups_list = []
    if groups:
        groups_list = [int(g.strip()) for g in groups.split(',')]

    # Replace the global variables
    new_content = content.replace(
        "MAX_VALUE = 0",
        f"MAX_VALUE = {max_value}"
    ).replace(
        "GROUPS = []",
        f"GROUPS = {groups_list}"
    ).replace(
        "DIFFERENT = False",
        f"DIFFERENT = {different}"
    ).replace(
        "BY_GROUPS = False",
        f"BY_GROUPS = {by_groups}"
    )

    # Write the customized file
    with open('postprocessor.py', 'w') as f:
        f.write(new_content)

    print(f"Created customized postprocessor at postprocessor.py")


def main():
    parser = argparse.ArgumentParser(
        description='Configure postprocessor.py with custom parameters'
    )
    parser.add_argument(
        '-m', '--max-value',
        type=int,
        default=100,
        help='Maximum possible score value'
    )
    parser.add_argument(
        '-g', '--groups',
        help='Comma-separated group sizes for group scoring (e.g. "3,5,2")'
    )
    parser.add_argument(
        '-d', '--different',
        action='store_true',
        help='Enable differential scoring mode'
    )
    parser.add_argument(
        '-b', '--by-groups',
        action='store_true',
        help='Enable group-based scoring mode'
    )

    args = parser.parse_args()

    create_custom_postprocessor(
        max_value=args.max_value,
        groups=args.groups,
        different=args.different,
        by_groups=args.by_groups
    )


if __name__ == "__main__":
    main()
