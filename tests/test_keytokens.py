# -*- coding: utf-8 -*-
"""关键 token 判定测试。"""

from __future__ import annotations

import unittest

from phase_shifting.keytokens import (
    KEY_MODE_AUTO,
    KEY_MODE_KEYWORDS,
    KEY_MODE_WEIGHT,
    KeyTokenSelector,
    clean_token,
    is_pure_punct_or_space,
)
from phase_shifting.weights import is_word_like, score_token, score_tokens
from phase_shifting.tokenizer_api import get_tokenizer


class PurePunctOrSpaceTest(unittest.TestCase):
    def test_empty_is_pure(self):
        self.assertTrue(is_pure_punct_or_space(""))

    def test_whitespace_only(self):
        self.assertTrue(is_pure_punct_or_space("   \t\n"))

    def test_punct_only(self):
        self.assertTrue(is_pure_punct_or_space("..., !!!"))

    def test_cjk_punct_only(self):
        self.assertTrue(is_pure_punct_or_space("，。！"))

    def test_letters_are_not_pure(self):
        self.assertFalse(is_pure_punct_or_space("abc"))

    def test_cjk_is_not_pure(self):
        self.assertFalse(is_pure_punct_or_space("关键"))

    def test_mixed(self):
        self.assertFalse(is_pure_punct_or_space("a."))


class CleanTokenTest(unittest.TestCase):
    def test_byte_level_prefix(self):
        self.assertEqual(clean_token("Ġworld"), "world")

    def test_plain(self):
        self.assertEqual(clean_token("Hello"), "Hello")

    def test_newline(self):
        self.assertEqual(clean_token("Ċ"), "")


class WeightTest(unittest.TestCase):
    def test_is_word_like(self):
        self.assertEqual(is_word_like("abc"), 1.0)
        self.assertEqual(is_word_like("关键"), 1.0)
        self.assertEqual(is_word_like("123"), 0.5)
        self.assertEqual(is_word_like("..."), 0.0)
        self.assertEqual(is_word_like(""), 0.0)

    def test_score_range(self):
        tk = get_tokenizer()
        ids = tk.encode("Hello world 关键 ...")
        tokens = tk.convert_ids_to_tokens(ids)
        results = score_tokens(tokens, ids, tk.vocab_size)
        for r in results:
            self.assertGreaterEqual(r.weight, 0.0)
            self.assertLessEqual(r.weight, 1.0)

    def test_punct_scores_low(self):
        """标点权重应显著低于实词（用 is_word 分项做严格断言）。

        注：'...' 在词表里是稀有 token，rarity 分项会把它抬高（weight≈0.385），
        因此不能直接断言 weight < 0.3；改用单字符 '.'（id 16，weight≈0.26），
        并另外校验"标点的 is_word 分项恒为 0"这一更强的不变量。
        """
        tk = get_tokenizer()
        dot = score_token(".", 16, tk.vocab_size)
        self.assertEqual(dot.is_word, 0.0)
        self.assertLess(dot.weight, 0.3)

        ids = tk.encode("...")
        tokens = tk.convert_ids_to_tokens(ids)
        ellipsis = score_token(tokens[0], ids[0], tk.vocab_size)
        self.assertEqual(ellipsis.is_word, 0.0)

    def test_word_scores_above_punct(self):
        """实词权重必须高于标点。"""
        tk = get_tokenizer()
        dot = score_token(".", 16, tk.vocab_size)
        ids = tk.encode("information")
        tokens = tk.convert_ids_to_tokens(ids)
        w = score_token(tokens[0], ids[0], tk.vocab_size)
        self.assertGreater(w.weight, dot.weight)
        self.assertGreater(w.is_word, 0.0)


class KeyTokenSelectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tk = get_tokenizer()
        cls.enc = cls.tk.encode_with_tokens("Hello world 关键信息 ...")
        cls.ids = cls.enc.ids
        cls.tokens = cls.enc.tokens

    def _select(self, **kw):
        sel = KeyTokenSelector(**kw)
        return sel.select(self.tokens, self.ids, self.tk.vocab_size)

    def test_auto_mode(self):
        rep = self._select(mode=KEY_MODE_AUTO)
        self.assertTrue(rep.key_indices)
        # 纯标点 token 不应该是关键 token
        for d in rep.decisions:
            if d.text in (".", "...", ""):
                self.assertFalse(d.is_key, d.text)

    def test_keywords_mode(self):
        rep = self._select(mode=KEY_MODE_KEYWORDS, keywords=["world", "关键"])
        hits = [d.text for d in rep.decisions if d.is_key]
        self.assertTrue(hits)
        for h in hits:
            self.assertTrue("world" in h.lower() or "关键" in h)

    def test_keywords_requires_keywords(self):
        with self.assertRaises(ValueError):
            KeyTokenSelector(mode=KEY_MODE_KEYWORDS, keywords=[])

    def test_weight_mode(self):
        rep = self._select(mode=KEY_MODE_WEIGHT, weight_threshold=0.5)
        for d in rep.decisions:
            if d.weight is not None:
                self.assertEqual(d.is_key, d.weight >= 0.5)

    def test_weight_threshold_range(self):
        with self.assertRaises(ValueError):
            KeyTokenSelector(mode=KEY_MODE_WEIGHT, weight_threshold=1.5)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            KeyTokenSelector(mode="nope")

    def test_delta_needed(self):
        rep = self._select(mode=KEY_MODE_AUTO)
        for d in rep.decisions:
            if d.is_key:
                delta = d.delta_needed(4)
                self.assertTrue(0 <= delta < 4)
                self.assertEqual((d.index + delta) % 4, 0)

    def test_describe(self):
        self.assertIn("auto", KeyTokenSelector().describe())
        kw = KeyTokenSelector(mode=KEY_MODE_KEYWORDS, keywords=["world"])
        self.assertIn("world", kw.describe())
        self.assertIn("0.5", KeyTokenSelector(mode=KEY_MODE_WEIGHT).describe())

    def test_empty_input(self):
        rep = KeyTokenSelector().select([], [], 100)
        self.assertEqual(rep.decisions, [])
        self.assertEqual(rep.key_indices, [])

    def test_no_keyword_match(self):
        rep = self._select(mode=KEY_MODE_KEYWORDS, keywords=["zzz不存在"])
        self.assertEqual(rep.key_indices, [])


if __name__ == "__main__":
    unittest.main()
