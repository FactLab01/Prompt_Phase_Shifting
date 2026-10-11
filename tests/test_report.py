# -*- coding: utf-8 -*-
"""报告渲染测试。"""

from __future__ import annotations

import json
import unittest

from phase_shifting.aligner import STATUS_ALIGNED, STATUS_NO_KEY, AlignConfig, Aligner
from phase_shifting.fillers import FillerRegistry
from phase_shifting.keytokens import KEY_MODE_KEYWORDS, KeyTokenSelector
from phase_shifting.report import render_index_table_only, render_json, render_text
from phase_shifting.tokenizer_api import get_tokenizer


class ReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tk = get_tokenizer()

    def _align(self, text, **kw):
        sel_kw = kw.pop("selector", {})
        reg = FillerRegistry(self.tk)
        filler_chars = [reg.get(n).char for n in reg.names()]
        al = Aligner(
            self.tk,
            KeyTokenSelector(tokenizer=self.tk, filler_chars=filler_chars, **sel_kw),
            reg,
            AlignConfig(**kw),
        )
        return al.align(text)

    def test_render_text_contains_sections(self):
        out = self._align("Hello world 关键信息")
        text = render_text(out, self.tk)
        self.assertIn("Token Index", text)
        self.assertIn("占位符详情", text)
        self.assertIn("结论", text)
        self.assertIn("ALIGNED", text)
        self.assertIn("U+200B", text)

    def test_render_text_verbose(self):
        out = self._align("Hello world 关键信息")
        text = render_text(out, self.tk, verbose=True)
        self.assertIn("字符偏移", text)

    def test_render_text_no_key(self):
        out = self._align("Hello", selector={"mode": KEY_MODE_KEYWORDS, "keywords": ["zzz"]})
        self.assertEqual(out.status, STATUS_NO_KEY)
        text = render_text(out, self.tk)
        self.assertIn("NO_KEY_TOKENS", text)

    def test_render_text_zero_intervention(self):
        out = self._align("Hello")
        text = render_text(out, self.tk)
        self.assertIn("零干预", text)

    def test_render_json_valid(self):
        out = self._align("Hello world 关键信息")
        payload = json.loads(render_json(out, self.tk))
        self.assertEqual(payload["status"], STATUS_ALIGNED)
        self.assertEqual(payload["period"], 4)
        self.assertTrue(payload["insertions"])
        for ins in payload["insertions"]:
            self.assertIn("filler_codepoint", ins)
            self.assertIn("token_cost", ins)
            self.assertIn("char_count", ins)

    def test_render_json_has_aligned_text(self):
        out = self._align("Hello world 关键信息")
        payload = json.loads(render_json(out, self.tk))
        self.assertIn("aligned_text", payload)
        self.assertIn("aligned_text_visible", payload)
        self.assertIn("<ZWSP>", payload["aligned_text_visible"])

    def test_render_table_only(self):
        out = self._align("Hello world 关键信息")
        table = render_index_table_only(out, self.tk)
        self.assertIn("before", table)
        self.assertIn("after", table)
        self.assertIn("YES", table)

    def test_render_table_no_key(self):
        out = self._align("Hello", selector={"mode": KEY_MODE_KEYWORDS, "keywords": ["zzz"]})
        self.assertIn("无关键 token", render_index_table_only(out, self.tk))


if __name__ == "__main__":
    unittest.main()
