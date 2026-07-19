#!/usr/bin/env python3

"""Apply Art. Lebedev Studio's Typograf service to a text file."""

from __future__ import annotations

import argparse
import html
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path


TYPOGRAF_URL = "http://typograf.artlebedev.ru/webservices/typograf.asmx"
SOAP_ACTION = "http://typograf.artlebedev.ru/webservices/ProcessText"
PLACEHOLDER_PATTERN = re.compile(r"\$\$[\s\S]+?\$\$|\$[^$\n]+\$|`[^`\n]*`")
RESULT_PATTERN = re.compile(r"<ProcessTextResult>([\s\S]*?)</ProcessTextResult>")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Protect formulas and inline code, send the rest to Typograf, "
            "and write the result back to the same file."
        )
    )
    parser.add_argument("file", help="Path to the file to typograph")
    return parser.parse_args(argv)


def protect_segments(text: str) -> tuple[str, list[tuple[str, str]]]:
    segments: list[tuple[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        placeholder = f"__CH_TYPOGRAF_PLACEHOLDER_{len(segments)}__"
        segments.append((placeholder, match.group(0)))
        return placeholder

    return PLACEHOLDER_PATTERN.sub(replace, text), segments


def restore_segments(text: str, segments: list[tuple[str, str]]) -> str:
    result = text
    for placeholder, value in segments:
        result = result.replace(placeholder, value)
    return result


def build_soap_envelope(text: str) -> bytes:
    escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    envelope = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
        'xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
        "<soap:Body>"
        '<ProcessText xmlns="http://typograf.artlebedev.ru/webservices/">'
        f"<text>{escaped_text}</text>"
        "<entityType>3</entityType>"
        "<useBr>false</useBr>"
        "<useP>false</useP>"
        "<maxNobr>3</maxNobr>"
        "</ProcessText>"
        "</soap:Body>"
        "</soap:Envelope>"
    )
    return envelope.encode("utf-8")


def normalize_result(input_text: str, output_text: str) -> str:
    normalized = output_text.replace("\r\n", "\n")
    if not input_text.endswith("\n"):
        normalized = re.sub(r"\n+$", "", normalized)
    return normalized


def typograf_text(text: str) -> str:
    request = urllib.request.Request(
        TYPOGRAF_URL,
        data=build_soap_envelope(text),
        headers={
            "Content-Type": "text/xml; charset=utf-8",
            "SOAPAction": f'"{SOAP_ACTION}"',
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            xml = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as error:
        raise RuntimeError(f"Typograf request failed: {error}") from error

    match = RESULT_PATTERN.search(xml)
    if not match:
        raise RuntimeError("ProcessTextResult not found in response")

    return normalize_result(text, html.unescape(match.group(1)))


def process_file(
    target_file: Path,
    processor: Callable[[str], str] = typograf_text,
) -> None:
    original_text = target_file.read_text(encoding="utf-8")
    protected_text, segments = protect_segments(original_text)
    typographed_text = processor(protected_text)
    target_file.write_text(restore_segments(typographed_text, segments), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target_file = Path(args.file)

    if not target_file.is_file():
        print(f"File not found: {target_file}", file=sys.stderr)
        return 1

    try:
        process_file(target_file)
    except (OSError, UnicodeError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
