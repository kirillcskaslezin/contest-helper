from typing import Iterable, NoReturn

from contest_helper.basic import Generator, Value
from contest_helper.utils import printer

Input = None
Output = None


def solution(data: Input) -> Output:
    ...


def input_parser(data: Iterable[str]) -> Input:
    ...


@printer
def input_printer(data: Input) -> NoReturn:
    ...


@printer
def output_printer(data: Output) -> NoReturn:
    ...


generator = Generator(
    solution=solution,
    samples=[],
    tests_generator=Value(None),
    tests_count=0,
    input_parser=input_parser,
    input_printer=input_printer,
    output_printer=output_printer,
)

generator.run()
