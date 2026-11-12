# -*- coding: utf-8 -*-
"""宏观窗口（Peak Window）判定测试。

覆盖三层：

1. 纯函数层：:func:`window_scores` / :func:`peak_window` 的滑动累加、
   聚合方式、同分 tie-breaking、边界输入；
2. 选择器层：``weight`` 模式开启 ``window_size`` 后，只有 Peak Window 内的
   token 会被判为关键 token；
3. 对齐集成层：窗口化配置经 SDK 门面生效，仍满足幂等与还原不变量。
"""

from __future__ import annotations

import unittest

from phase_shifting import (
    PhaseShiftConfig,
    PhaseShifter,
    align_text,
    plan_text,
    restore_text,
)
from phase_shifting.keytokens import KEY_MODE_WEIGHT, KeyTokenSelector
from phase_shifting.weights import peak_window, window_scores
from phase_shifting.tokenizer_api import get_tokenizer

#: 关键信息集中在后段的样本：前段是无关背景，后段才是核心结论。
FILLER = (
    "这里是一段与任务无关的背景描述，用于把全文的 token 数撑过一个窗口。"
    "其中提到一些常见的名词与动词，但都不承载本次任务需要的核心信息。"
    "本段继续补充若干普通语句，让窗口滑动时有多个候选区间可以比较。"
)
CORE = (
    "下面给出核心结论：系统的主访问密钥是 KEY-7734，"
    "备用密钥是 KEY-9021，主密钥失效时请改用备用密钥完成鉴权。"
)


class WindowScoresTest(unittest.TestCase):
    """纯函数层：滑动窗口分数序列。"""

    def test_sum_aggregation_matches_naive(self):
        """sum 聚合必须与朴素逐窗求和完全一致（含浮点累进）。"""
        w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        size = 3
        got = window_scores(w, size)
        expected = [sum(w[i : i + size]) for i in range(len(w) - size + 1)]
        self.assertEqual(len(got), len(expected))
        for a, b in zip(got, expected):
            self.assertAlmostEqual(a, b, places=12)

    def test_count_aggregation_counts_above_threshold(self):
        """count 聚合统计窗口内不低于阈值的 token 个数。"""
        w = [0.1, 0.9, 0.8, 0.1, 0.7]
        self.assertEqual(window_scores(w, 2, agg="count", threshold=0.5), [1, 2, 1, 1])

    def test_length_is_n_minus_window_plus_one(self):
        w = [0.5] * 10
        for size in (1, 3, 10):
            self.assertEqual(len(window_scores(w, size)), 10 - size + 1)

    def test_empty_and_zero_window_return_empty(self):
        self.assertEqual(window_scores([], 4), [])
        self.assertEqual(window_scores([0.1, 0.2], 0), [])

    def test_window_larger_than_input_returns_empty(self):
        self.assertEqual(window_scores([0.1, 0.2], 8), [])

    def test_invalid_agg_rejected(self):
        with self.assertRaises(ValueError):
            window_scores([0.1, 0.2], 2, agg="bogus")

    def test_negative_window_rejected(self):
        with self.assertRaises(ValueError):
            window_scores([0.1, 0.2], -1)


class PeakWindowTest(unittest.TestCase):
    """纯函数层：峰值窗口定位。"""

    def test_peak_covers_densest_block(self):
        """峰窗应覆盖权重最高的那一块。"""
        w = [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]
        pk = peak_window(w, 4)
        self.assertIsNotNone(pk)
        self.assertEqual(pk.start, 4)
        self.assertEqual(pk.end, 7)
        self.assertEqual(pk.size, 4)
        self.assertAlmostEqual(pk.score, 3.6, places=12)

    def test_tie_break_takes_earliest(self):
        """多个同分窗口必须取最靠前的一个（确定性）。"""
        w = [0.5, 0.5, 0.1, 0.1, 0.5, 0.5]
        for size in (1, 2):
            pk = peak_window(w, size)
            self.assertEqual(pk.start, 0)
            self.assertEqual(pk.end, size - 1)

    def test_returns_none_when_no_window_fits(self):
        self.assertIsNone(peak_window([0.1, 0.2], 8))
        self.assertIsNone(peak_window([], 4))

    def test_count_agg_peak(self):
        w = [0.1, 0.9, 0.1, 0.1, 0.9, 0.9]
        pk = peak_window(w, 2, agg="count", threshold=0.5)
        self.assertEqual(pk.start, 4)
        self.assertEqual(pk.score, 2)


class WindowedSelectorTest(unittest.TestCase):
    """选择器层：窗口化 weight 判定。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tk = get_tokenizer()

    def _select(self, text, **kw):
        kw.setdefault("mode", KEY_MODE_WEIGHT)
        kw.setdefault("weight_threshold", 0.5)
        enc = self.tk.encode_with_tokens(text)
        sel = KeyTokenSelector(tokenizer=self.tk, **kw)
        return sel.select(enc.tokens, enc.ids, self.tk.vocab_size)

    def test_window_narrows_key_set(self):
        """开启窗口后关键 token 数量必须显著少于逐 token 判定。"""
        text = FILLER + CORE
        plain = self._select(text)
        windowed = self._select(text, window_size=16)
        self.assertLess(len(windowed.key_indices), len(plain.key_indices))
        self.assertTrue(windowed.key_indices)

    def test_key_tokens_all_inside_peak(self):
        """窗口化判定的关键 token 必须全部落在峰窗区间内。"""
        text = FILLER + CORE
        rep = self._select(text, window_size=16)
        peak = rep.peak
        self.assertIsNotNone(peak)
        for d in rep.decisions:
            if d.is_key:
                self.assertTrue(
                    peak.start <= d.index <= peak.end,
                    "token#%d %r 在峰窗外" % (d.index, d.text),
                )

    def test_peak_window_first_token_is_key(self):
        """峰窗首个 token 必须被判为关键 token（对齐锚点）。"""
        text = FILLER + CORE
        rep = self._select(text, window_size=16)
        self.assertIsNotNone(rep.peak)
        first = rep.by_index(rep.peak.start)
        self.assertTrue(first.is_key, "峰窗首 token 未判为关键：%r" % first.reason)

    def test_off_window_reason_mentions_peak(self):
        """峰窗外的 token 必须带上"窗口外"的理由，便于排查。"""
        rep = self._select(FILLER + CORE, window_size=16)
        outside = [d for d in rep.decisions if not d.is_key]
        self.assertTrue(any("窗口外" in d.reason for d in outside))

    def test_window_scores_sequence_present(self):
        """报告应带上窗口分数序列，长度 = N - window_size + 1。"""
        text = FILLER + CORE
        enc = self.tk.encode_with_tokens(text)
        rep = self._select(text, window_size=16)
        self.assertIsNotNone(rep.window_scores)
        self.assertEqual(len(rep.window_scores), len(enc.ids) - 16 + 1)

    def test_short_text_falls_back_to_whole_stream(self):
        """全文短于一个窗口时退化为全流判定（峰窗为 None）。"""
        rep = self._select("KEY-7734", window_size=64)
        self.assertIsNone(rep.peak)
        self.assertEqual(rep.window_scores, [])
        self.assertTrue(rep.key_indices)

    def test_window_size_zero_disables_windowing(self):
        """``window_size=0`` 必须完全等价于逐 token 判定。"""
        text = FILLER + CORE
        a = self._select(text)
        b = self._select(text, window_size=0)
        self.assertEqual(
            [d.is_key for d in a.decisions], [d.is_key for d in b.decisions]
        )
        self.assertIsNone(b.peak)

    def test_window_rejected_for_non_weight_mode(self):
        with self.assertRaises(ValueError):
            KeyTokenSelector(mode="auto", window_size=16)

    def test_negative_window_rejected(self):
        with self.assertRaises(ValueError):
            KeyTokenSelector(mode=KEY_MODE_WEIGHT, window_size=-1)

    def test_bad_agg_rejected(self):
        with self.assertRaises(ValueError):
            KeyTokenSelector(mode=KEY_MODE_WEIGHT, window_agg="bogus")

    def test_describe_mentions_window(self):
        sel = KeyTokenSelector(mode=KEY_MODE_WEIGHT, window_size=16, window_agg="count")
        self.assertIn("16", sel.describe())
        self.assertIn("count", sel.describe())

    def test_filler_never_key(self):
        """占位符 token 在任何模式下都不得被判为关键 token。"""
        text = FILLER + CORE
        enc = self.tk.encode_with_tokens(text)
        aligned = align_text(
            text, config=PhaseShiftConfig(key_mode="weight", window_size=16)
        ).aligned_text
        enc2 = self.tk.encode_with_tokens(aligned)
        sel = KeyTokenSelector(
            mode=KEY_MODE_WEIGHT,
            window_size=16,
            tokenizer=self.tk,
            filler_chars=["\u200b", "\ufeff"],
        )
        rep = sel.select(enc2.tokens, enc2.ids, self.tk.vocab_size)
        for d in rep.decisions:
            if d.text in ("\u200b", "\ufeff"):
                self.assertFalse(d.is_key, d.text)


class WindowedAlignIntegrationTest(unittest.TestCase):
    """对齐集成层：SDK 门面 + 引擎。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tk = get_tokenizer()
        cls.text = FILLER + CORE

    def _cfg(self, **kw):
        kw.setdefault("key_mode", "weight")
        kw.setdefault("weight_threshold", 0.5)
        kw.setdefault("window_size", 16)
        return PhaseShiftConfig(**kw)

    def test_plan_reports_peak_window(self):
        plan = plan_text(self.text, config=self._cfg())
        self.assertIsNotNone(plan.peak_window)
        self.assertEqual(plan.peak_window["size"], 16)
        self.assertIn("start", plan.peak_window)
        self.assertIn("score", plan.peak_window)
        import json

        json.dumps(plan.to_dict(), ensure_ascii=False)

    def test_plan_peak_window_none_when_disabled(self):
        plan = plan_text(
            self.text,
            config=PhaseShiftConfig(key_mode="weight", weight_threshold=0.5),
        )
        self.assertIsNone(plan.peak_window)

    def test_align_converges(self):
        result = align_text(self.text, config=self._cfg())
        self.assertEqual(result.status, "ALIGNED")
        self.assertGreater(result.token_delta, 0)
        self.assertEqual(result.token_delta, sum(i["char_count"] for i in result.insertions))

    def test_align_is_idempotent(self):
        cfg = self._cfg()
        first = align_text(self.text, config=cfg)
        second = align_text(first.aligned_text, config=cfg)
        self.assertEqual(first.aligned_text, second.aligned_text)
        self.assertEqual(second.insertions, [])

    def test_restore_roundtrip(self):
        cfg = self._cfg()
        aligned = align_text(self.text, config=cfg).aligned_text
        self.assertEqual(restore_text(aligned), self.text)

    def test_window_size_variants_all_align(self):
        for size in (8, 16, 32, 64):
            result = align_text(self.text, config=self._cfg(window_size=size))
            self.assertEqual(result.status, "ALIGNED", "window_size=%d" % size)

    def test_count_agg_variant_aligns(self):
        result = align_text(
            self.text, config=self._cfg(window_size=16, window_agg="count")
        )
        self.assertEqual(result.status, "ALIGNED")

    def test_shifter_object_path_matches_function_path(self):
        cfg = self._cfg()
        a = PhaseShifter(cfg).align(self.text)
        b = align_text(self.text, config=cfg)
        self.assertEqual(a.aligned_text, b.aligned_text)

    def test_config_roundtrip_preserves_window(self):
        cfg = self._cfg(window_size=16, window_agg="count")
        clone = PhaseShiftConfig.from_dict(cfg.to_dict())
        self.assertEqual(clone.window_size, 16)
        self.assertEqual(clone.window_agg, "count")

    def test_invalid_window_config_rejected(self):
        from phase_shifting.errors import ConfigurationError

        for bad in ({"window_size": -1}, {"window_agg": "bogus"}, {"window_size": 1.5}):
            with self.assertRaises(ConfigurationError):
                PhaseShiftConfig(key_mode="weight", **bad)

    def test_window_only_applies_to_weight_mode(self):
        from phase_shifting.errors import ConfigurationError

        with self.assertRaises(ConfigurationError):
            PhaseShiftConfig(key_mode="auto", window_size=16)


if __name__ == "__main__":
    unittest.main()
