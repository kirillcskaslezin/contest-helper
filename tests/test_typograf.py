import tempfile
import unittest
from pathlib import Path

from contest_helper.cli.typograf import (
    build_soap_envelope,
    normalize_result,
    process_file,
    protect_segments,
    restore_segments,
)


class TypografTest(unittest.TestCase):
    def test_protect_and_restore_formulas_and_inline_code(self):
        source = "Text $a < b$ and $$x = 1$$ with `value < 2`."

        protected, segments = protect_segments(source)

        self.assertNotIn("$a < b$", protected)
        self.assertNotIn("`value < 2`", protected)
        self.assertEqual(restore_segments(protected, segments), source)

    def test_protect_and_restore_blank_lines(self):
        source = "First paragraph.\n\nSecond paragraph.\n\n\nFourth paragraph."

        protected, segments = protect_segments(source)

        self.assertNotIn("\n\n", protected)
        self.assertEqual(restore_segments(protected, segments), source)

    def test_protect_and_restore_leading_and_indented_blank_lines(self):
        source = "\n\nFirst.\n \n\t\nSecond.\n"

        protected, segments = protect_segments(source)

        self.assertEqual(restore_segments(protected, segments), source)

    def test_protect_and_restore_fenced_code_blocks(self):
        source = (
            "Before.\n\n"
            "```python\n"
            "value = \"do not typograph\"\n\n"
            "print(value)\n"
            "```\n\n"
            "Between.\n\n"
            "~~~sql\n"
            "SELECT * FROM table_name;\n"
            "~~~\n\n"
            "After."
        )

        protected, segments = protect_segments(source)

        self.assertNotIn("do not typograph", protected)
        self.assertNotIn("SELECT * FROM", protected)
        self.assertEqual(restore_segments(protected, segments), source)

    def test_protects_unclosed_fenced_code_block_to_end_of_file(self):
        source = "Before.\n\n```python\nprint(\"unfinished\")\n"

        protected, segments = protect_segments(source)

        self.assertNotIn("unfinished", protected)
        self.assertEqual(restore_segments(protected, segments), source)

    def test_protect_and_restore_markdown_list_markers(self):
        source = (
            "- First item\n"
            "* Second item\n"
            "+ Third item\n"
            "  - Nested item\n"
            "1. Ordered item\n"
            "2) Another ordered item\n"
            "- [ ] Open task\n"
            "- [x] Completed task\n"
        )

        protected, segments = protect_segments(source)

        self.assertNotIn("- First", protected)
        self.assertNotIn("1. Ordered", protected)
        self.assertNotIn("- [ ] Open", protected)
        self.assertEqual(restore_segments(protected, segments), source)

    def test_build_soap_envelope_escapes_xml(self):
        envelope = build_soap_envelope("a < b & c > d").decode("utf-8")

        self.assertIn("<text>a &lt; b &amp; c &gt; d</text>", envelope)

    def test_normalize_result_preserves_newline_policy(self):
        self.assertEqual(normalize_result("text", "result\r\n\n"), "result")
        self.assertEqual(normalize_result("text\n", "result\r\n"), "result\n")

    def test_process_file_protects_segments_and_writes_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "statement.md"
            target.write_text('Some "text" and $a < b$ with `x = 1`.', encoding="utf-8")

            def processor(text: str) -> str:
                self.assertNotIn("$a < b$", text)
                self.assertNotIn("`x = 1`", text)
                return text.replace('"text"', "«text»")

            process_file(target, processor=processor)

            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "Some «text» and $a < b$ with `x = 1`.",
            )

    def test_process_file_preserves_blank_lines_removed_by_processor(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "statement.md"
            source = "# Title\n\nFirst paragraph.\n\n\nSecond paragraph.\n"
            target.write_text(source, encoding="utf-8")

            process_file(target, processor=lambda text: text.replace("\n\n", "\n"))

            self.assertEqual(target.read_text(encoding="utf-8"), source)

    def test_process_file_does_not_send_code_blocks_to_processor(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "statement.md"
            source = 'Text with "quotes".\n\n```python\nprint("code")\n```\n'
            target.write_text(source, encoding="utf-8")

            def processor(text: str) -> str:
                self.assertNotIn('print("code")', text)
                return text.replace('"quotes"', "«quotes»")

            process_file(target, processor=processor)

            self.assertEqual(
                target.read_text(encoding="utf-8"),
                'Text with «quotes».\n\n```python\nprint("code")\n```\n',
            )

    def test_process_file_preserves_list_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "statement.md"
            source = '- "First" item\n  - "Nested" item\n1. "Ordered" item\n'
            target.write_text(source, encoding="utf-8")

            def processor(text: str) -> str:
                self.assertNotIn("- ", text)
                self.assertNotIn("1. ", text)
                return text.replace('"First"', "«First»")

            process_file(target, processor=processor)

            self.assertEqual(
                target.read_text(encoding="utf-8"),
                '- «First» item\n  - "Nested" item\n1. "Ordered" item\n',
            )


if __name__ == "__main__":
    unittest.main()
