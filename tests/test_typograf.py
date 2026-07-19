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


if __name__ == "__main__":
    unittest.main()
