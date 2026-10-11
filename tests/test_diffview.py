# -*- coding: utf-8 -*-
"""差异视图测试。"""

from __future__ import annotations

import unittest

from phase_shifting.diffview import render_diff, render_inline_markers, render_side_by_side


class RenderDiffTest(unittest.TestCase):
    def test_no_diff(self):
        self.assertEqual(render_diff("abc", "abc"), "")

    def test_diff_shows_marker(self):
        before = "Hello world"
        after = "​​Hello world"
        diff = render_diff(before, after)
        self.assertIn("ZWSP", diff)
        self.assertIn("before", diff)
        self.assertIn("after", diff)

    def test_diff_multiline(self):
        before = "第一段\n\n第二段"
        after = "​第一段\n\n第二段"
        diff = render_diff(before, after)
        self.assertIn("ZWSP", diff)

    def test_empty_both(self):
        self.assertEqual(render_diff("", ""), "")


class SideBySideTest(unittest.TestCase):
    def test_basic(self):
        view = render_side_by_side("abc", "abcd")
        self.assertIn("BEFORE", view)
        self.assertIn("AFTER", view)
        self.assertIn("abc", view)

    def test_empty(self):
        view = render_side_by_side("", "")
        self.assertIn("BEFORE", view)


class InlineMarkersTest(unittest.TestCase):
    def test_zwsp_marked(self):
        out = render_inline_markers("a​b")
        self.assertIn("ZWSP", out)

    def test_bom_marked(self):
        out = render_inline_markers("a﻿b")
        self.assertIn("BOM", out)

    def test_plain_text_unchanged(self):
        self.assertEqual(render_inline_markers("abc"), "abc")

    def test_empty(self):
        self.assertEqual(render_inline_markers(""), "")


if __name__ == "__main__":
    unittest.main()
