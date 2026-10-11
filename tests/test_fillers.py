# -*- coding: utf-8 -*-
"""占位符注册表测试。"""

from __future__ import annotations

import unittest

from phase_shifting.fillers import FillerRegistry
from phase_shifting.tokenizer_api import get_tokenizer


class FillerRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tk = get_tokenizer()
        cls.reg = FillerRegistry(cls.tk)

    def test_candidate_order(self):
        self.assertEqual(self.reg.names()[:3], ["zwsp", "bom", "mix"])

    def test_primary_is_zwsp(self):
        primary = self.reg.primary()
        self.assertEqual(primary.name, "zwsp")
        self.assertTrue(primary.invisible)
        self.assertFalse(primary.structural)
        self.assertEqual(primary.declared_cost, 1)

    def test_declared_cost_matches_reality(self):
        """注册表声明的每字符 token 数必须与实测一致。"""
        for name in self.reg.names():
            spec = self.reg.get(name)
            self.assertEqual(
                spec.declared_cost,
                self.tk.probe_cost(spec.char),
                "%s 声明的成本与实测不符" % name,
            )

    def test_render_linear_for_zwsp(self):
        """ZWSP 每字符必须贡献恰好 1 个 token（线性、无合并）。"""
        for n in range(1, 6):
            payload = self.reg.render("zwsp", n)
            self.assertEqual(self.tk.count_tokens(payload), n)
            self.assertEqual(self.reg.cost_of("zwsp", n), n)

    def test_render_zero_and_negative(self):
        self.assertEqual(self.reg.render("zwsp", 0), "")
        self.assertEqual(self.reg.render("zwsp", -3), "")

    def test_mix_alternates(self):
        payload = self.reg.render("mix", 4)
        zwsp = self.reg.get("zwsp").char
        bom = self.reg.get("bom").char
        self.assertEqual(payload, zwsp + bom + zwsp + bom)
        self.assertEqual(self.tk.count_tokens(payload), 4)

    def test_verify_isolated_zwsp(self):
        self.assertTrue(self.reg.verify_isolated("zwsp", 3))

    def test_verify_isolated_detects_merge(self):
        """LF 会被合并，隔离性校验必须失败。"""
        self.assertFalse(self.reg.verify_isolated("lf", 3))

    def test_unknown_name_raises(self):
        with self.assertRaises(KeyError):
            self.reg.get("nope")

    def test_describe_all_non_empty(self):
        lines = self.reg.describe_all()
        self.assertEqual(len(lines), len(self.reg.names()))
        for line in lines:
            self.assertIn("U+", line)


if __name__ == "__main__":
    unittest.main()
