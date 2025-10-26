from typing import Iterable

from contest_helper.basic import (
    Generator,
    Value,
    TextInputAdapter,
    TextOutputAdapter,
    BinaryInputAdapter,
    BinaryOutputAdapter,
)

Input = None
Output = None


def solution(data: Input) -> Output:
    ...


class MyInputAdapter(TextInputAdapter):
    def parse_lines(self, lines: Iterable[str]):
        # TODO: convert text lines to your structured Input
        return list(lines)

    def input_lines(self, data: Input):
        # TODO: render your structured Input back to lines
        return [str(data)]


class MyBinInputAdapter(BinaryInputAdapter):
    def parse_bytes(self, blob: bytes):
        # TODO: convert bytes to your structured Input
        return blob

    def input_bytes(self, data: Input):
        # TODO: render your structured Input back to bytes
        return [bytes(data)] if isinstance(data, (bytes, bytearray)) else []


class MyOutputAdapter(TextOutputAdapter):
    def output_lines(self, result: Output):
        # TODO: render your Output as lines
        return [str(result)]


class MyBinOutputAdapter(BinaryOutputAdapter):
    def output_bytes(self, result: Output):
        # TODO: render your Output to bytes
        return [bytes(result)] if isinstance(result, (bytes, bytearray)) else []


def validator(input_data: Input, output_data: Output) -> bool:
    return True


generator = Generator(
    solution=solution,
    samples=[],
    tests_generator=Value(None),
    tests_count=0,
    validator=validator,
    input_adapter=__INPUT_ADAPTER__,
    output_adapter=__OUTPUT_ADAPTER__,
)

generator.run()
