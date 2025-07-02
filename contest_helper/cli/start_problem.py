import argparse
from os import mkdir, path
from contest_helper.cli.utils import load_template


def main():
    parser = argparse.ArgumentParser(
        description='Configure problem with custom parameters'
    )
    parser.add_argument('directory', help='directory to start from')
    parser.add_argument('--language', '-l', default='en', help='language of statement')
    parser.add_argument('--checker', '-c', action="store_true", help='create checker file')
    args = parser.parse_args()

    process(args.directory, args.language, args.checker)


def process(directory, language, need_checker=False):
    mkdir(directory)

    with open(path.join(directory, 'statement.md'), 'w') as file:
        file.write(load_template(path.join('statements', f'{language}.md')))

    with open(path.join(directory, 'generator.py'), 'w') as file:
        file.write(load_template('generator.py'))

    with open(path.join(directory, 'meta.json'), 'w') as file:
        file.write(load_template('meta.json'))

    if need_checker:
        with open(path.join(directory, 'checker.py'), 'w') as file:
            file.write(load_template('checker.py'))


if __name__ == "__main__":
    main()
