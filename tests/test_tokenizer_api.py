# -*- coding: utf-8 -*-
"""tokenizer_api 封装层测试。"""

from __future__ import annotations

import unittest

from phase_shifting.tokenizer_api import (
    TokenizerHandle,
    codepoint_str,
    describe_char,
    get_tokenizer,
    visible,
)


class TokenizerApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tk = get_tokenizer()

    def test_encode_decode_roundtrip(self):
        sample = "phase shifting 相位偏移 123 abc"
        ids = self.tk.encode(sample)
        self.assertTrue(ids)
        self.assertEqual(self.tk.decode(ids), sample)

    def test_encode_empty(self):
        self.assertEqual(self.tk.encode(""), [])
        self.assertEqual(self.tk.decode([]), "")
        self.assertEqual(self.tk.decode_clean([]), "")

    def test_encode_with_tokens(self):
        enc = self.tk.encode_with_tokens("Hello world")
        self.assertEqual(len(enc.ids), len(enc.tokens))
        self.assertEqual(enc.tokens[0], "Hello")

    def test_token_str_and_id(self):
        ids = self.tk.encode("Hello")
        tok = self.tk.token_str(ids[0])
        self.assertEqual(tok, "Hello")
        self.assertEqual(self.tk.token_id("Hello"), ids[0])
        self.assertIsNone(self.tk.token_id("不存在的token字符串"))

    def test_decode_clean_restores_space(self):
        ids = self.tk.encode("Hello world")
        self.assertEqual(self.tk.decode_clean(ids), "Hello world")

    def test_probe_cost_zwsp_is_one(self):
        """ZWSP 必须是 1 字符 = 1 token（本 Demo 的核心假设）。"""
        self.assertEqual(self.tk.probe_cost("​"), 1)
        self.assertEqual(self.tk.probe_cost("﻿"), 1)

    def test_probe_cost_lf_merges(self):
        """连续 LF 会被合并成 1 个 token → 不能用作推进单位。"""
        self.assertEqual(self.tk.count_tokens("\n\n\n"), 1)

    def test_probe_cost_zwnj_splits(self):
        """ZWNJ 1 字符 = 2 token → 无法表达奇数位移。"""
        self.assertEqual(self.tk.probe_cost("‌"), 2)

    def test_health_check(self):
        ok, msg = self.tk.health_check()
        self.assertTrue(ok, msg)

    def test_locate_index_nth(self):
        # 注意：'alpha' 与 'Ġalpha' 是**不同**的 token id，真正重复出现的是后者
        text = "alpha beta alpha gamma alpha"
        ids = self.tk.encode(text)
        repeated = [i for i, v in enumerate(ids) if v == ids[2]]
        self.assertEqual(repeated, [2, 4])
        first = self.tk.locate_index_nth(text, ids[2], 0)
        second = self.tk.locate_index_nth(text, ids[2], 1)
        self.assertEqual((first, second), (2, 4))

    def test_locate_index_missing_token(self):
        text = "alpha beta"
        # 用一个不存在于文本中的 id：退化为 hint
        self.assertEqual(self.tk.locate_index_nth(text, 1, 0, hint=7), 7)

    def test_visible_and_codepoint(self):
        self.assertEqual(codepoint_str("​"), "U+200B")
        self.assertEqual(visible("a​b"), "a<ZWSP>b")
        self.assertIn("U+200B", describe_char("​"))


class TokenizerHandleInjectionTest(unittest.TestCase):
    def test_custom_tokenizer_injection(self):
        """支持注入自定义 tokenizer（测试替身）。"""

        class Fake:
            vocab_size = 10

            def encode(self, text, add_special_tokens=True):
                return list(range(len(text)))

            def decode(self, ids, **kw):
                return "x" * len(ids)

            def convert_ids_to_tokens(self, ids):
                return ["t%d" % i for i in ids]

            def convert_tokens_to_ids(self, tok):
                return None

        handle = TokenizerHandle(Fake())
        self.assertEqual(handle.vocab_size, 10)
        self.assertEqual(handle.encode("abc"), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
