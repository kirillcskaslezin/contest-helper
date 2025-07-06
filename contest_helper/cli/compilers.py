"""
Compiler Search Tool

This script allows searching through a list of compilers and development tools
loaded from a CSV file. It provides a command-line interface to search by name
and displays results in a formatted table.
"""

import argparse
import csv
import io
from tabulate import tabulate
from .utils import load_template


def load_compilers_data():
    """
    Load compiler data from a CSV file or template.

    Args:
        csv_file (str): Path to the CSV file containing compiler data.
                       Defaults to 'compilers.csv'.

    Returns:
        list: A list of dictionaries representing compiler entries.
    """
    # Using StringIO to simulate file reading from template
    buffer = io.StringIO(load_template('compilers.csv'))
    reader = csv.DictReader(buffer)
    return list(reader)


def search_compilers(data, search_query):
    """
    Search compilers by name matching the query (case-insensitive).

    Args:
        data (list): List of compiler dictionaries to search through.
        search_query (str): Search term to match against compiler names.

    Returns:
        list: Filtered list of compilers matching the search query.
    """
    search_query = search_query.lower()
    results = []
    for row in data:
        if search_query in row['Name'].lower():
            results.append(row)
    return results


def main():
    """Main function that handles command-line interface and program flow."""
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description='Search compilers and development tools by name'
    )
    parser.add_argument(
        'search',
        help='Search query (part of compiler name)'
    )
    args = parser.parse_args()

    try:
        # Load compiler data
        data = load_compilers_data()

        # Perform search
        results = search_compilers(data, args.search)

        # Display results
        if not results:
            print(f"No matches found for '{args.search}'")
        else:
            print(f"Found {len(results)} matches for '{args.search}':\n")
            # Format results as a pretty table
            print(tabulate(
                [(i + 1, row['ID'], row['Name']) for i, row in enumerate(results)],
                headers=['#', 'ID', 'Name'],
                tablefmt='grid',
                stralign='left',
                numalign='left'
            ))
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
