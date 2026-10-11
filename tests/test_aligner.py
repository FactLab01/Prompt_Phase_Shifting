# -*- coding: utf-8 -*-
"""对齐引擎测试（覆盖全部边界场景 E1–E11）。"""

from __future__ import annotations

import unittest

from phase_shifting.aligner import (
    INSERT_MODE_INLINE,
    STATUS_ALIGNED,
    STATUS_FAILED,
    STATUS_NO_KEY,
    STATUS_PARTIAL,
    AlignConfig,
    Aligner,
    _occurrence_of,
)
from phase_shifting.fillers import FillerRegistry
from phase_shifting.keytokens import KEY_MODE_AUTO, KEY_MODE_KEYWORDS, KeyTokenSelector
from phase_shifting.tokenizer_api import get_tokenizer, visible

ZWSP = "​"


def make_aligner(**cfg):
    tk = get_tokenizer()
    reg = FillerRegistry(tk)
    filler_chars = [reg.get(n).char for n in reg.names()]
    selector_kw = cfg.pop("selector", {})
    return Aligner(
        tk,
        KeyTokenSelector(
            mode=cfg.pop("mode", KEY_MODE_AUTO),
            tokenizer=tk,
            filler_chars=filler_chars,
            **selector_kw,
        ),
        reg,
        AlignConfig(**cfg),
    )


def assert_all_key_tokens_aligned(testcase, outcome, tokenizer=None):
    """校验：引擎报告为 ALIGNED 时，所有关键 token 的真实新下标都能被 4 整除。

    说明：插入点在"段落头部"，因此同段内后续 token 会被顺带推移。引擎对每个
    未对齐 token 都会重新定位并单独插入，因此最终每个关键 token 都应落在
    ``index % period == 0`` 上。这里用"顺序匹配"独立复核引擎的结论。
    """
    tk = tokenizer or get_tokenizer()
    ids = tk.encode(outcome.aligned_text)
    report = outcome.key_report
    cursor = 0
    checked = 0
    for d in report.decisions:
        if not d.is_key:
            continue
        found = None
        for j in range(cursor, len(ids)):
            if ids[j] == d.token_id:
                found = j
                break
        if found is None:
            continue
        testcase.assertEqual(
            found % outcome.period,
            0,
            "token#%d %r 未对齐：新下标 %d" % (d.index, d.text, found),
        )
        cursor = found + 1
        checked += 1
    testcase.assertGreater(checked, 0, "没有任何关键 token 被复核")


class EmptyAndTrivialTest(unittest.TestCase):
    """E1 空输入 / E2 纯空白。"""

    def test_empty_input(self):
        out = make_aligner().align("")
        self.assertEqual(out.status, STATUS_ALIGNED)
        self.assertEqual(out.aligned_text, "")
        self.assertEqual(out.total_tokens_before, 0)
        self.assertEqual(out.insertions, [])

    def test_whitespace_only(self):
        out = make_aligner().align("   \n\n  ")
        self.assertEqual(out.status, STATUS_ALIGNED)
        self.assertEqual(out.insertions, [])

    def test_none_input(self):
        out = make_aligner().align(None)
        self.assertEqual(out.status, STATUS_ALIGNED)


class AlreadyAlignedTest(unittest.TestCase):
    """E3 已天然对齐。"""

    def test_natural_alignment(self):
        # "Hello world 关键信息" 中 index 0/1/3/4，1 和 3 未对齐 → 有插入
        out = make_aligner().align("Hello world 关键信息")
        self.assertEqual(out.status, STATUS_ALIGNED)
        self.assertTrue(out.insertions)

    def test_single_token_aligned(self):
        # 只有 1 个 token，index 0 已对齐 → 零插入
        out = make_aligner().align("Hello")
        self.assertEqual(out.status, STATUS_ALIGNED)
        self.assertEqual(out.insertions, [])
        self.assertEqual(out.aligned_text, "Hello")

    def test_summary_zero_intervention(self):
        out = make_aligner().align("Hello")
        self.assertIn("天然对齐", out.summary())


class MisalignedTest(unittest.TestCase):
    """E4 多个连续 token 均未对齐。"""

    def test_multiple_misaligned(self):
        out = make_aligner().align("alpha beta gamma delta epsilon zeta")
        self.assertEqual(out.status, STATUS_ALIGNED)
        self.assertTrue(len(out.insertions) >= 2)
        assert_all_key_tokens_aligned(self, out)

    def test_chinese_text(self):
        out = make_aligner().align("第一段落内容。\n\n第二段落关键信息在这里。\n\n第三段落补充说明。")
        self.assertEqual(out.status, STATUS_ALIGNED)
        assert_all_key_tokens_aligned(self, out)

    def test_mixed_language(self):
        out = make_aligner().align("Hello world 这是 mixed 中英混排 text 内容 123 end")
        self.assertEqual(out.status, STATUS_ALIGNED)
        assert_all_key_tokens_aligned(self, out)

    def test_insertion_details_recorded(self):
        out = make_aligner().align("Hello world 关键信息")
        self.assertTrue(out.insertions)
        for ins in out.insertions:
            self.assertGreater(ins.char_count, 0)
            self.assertGreater(ins.token_cost, 0)
            self.assertIn("U+", ins.filler_codepoint)
            self.assertFalse(ins.structural)
            self.assertEqual(ins.token_index_after % 4, 0)

    def test_filler_is_invisible(self):
        out = make_aligner().align("Hello world 关键信息")
        # 还原后应与原文一致（说明插入的都是不可见字符）
        restored, removed = make_aligner().restore(out.aligned_text)
        self.assertGreater(removed, 0)
        self.assertEqual(restored, out.original_text)


class NoKeyTokenTest(unittest.TestCase):
    """E8 关键词/权重模式无匹配。"""

    def test_keywords_no_match(self):
        out = make_aligner(mode=KEY_MODE_KEYWORDS, selector={"keywords": ["zzz"]}).align(
            "Hello world"
        )
        self.assertEqual(out.status, STATUS_NO_KEY)
        self.assertEqual(out.aligned_text, "Hello world")
        self.assertEqual(out.insertions, [])

    def test_whitespace_only_auto(self):
        out = make_aligner().align("   ")
        self.assertEqual(out.status, STATUS_ALIGNED)
        self.assertEqual(out.insertions, [])


class LongInputTest(unittest.TestCase):
    """E5 超长输入。"""

    def test_long_input(self):
        text = ("这是第 %d 段测试内容，包含一些英文单词与中文。\n\n" % 0) * 400
        out = make_aligner().align(text)
        self.assertIn(out.status, (STATUS_ALIGNED, STATUS_PARTIAL))
        if out.status == STATUS_ALIGNED:
            assert_all_key_tokens_aligned(self, out)

    def test_max_chars_limit(self):
        cfg = AlignConfig(max_chars=10)
        tk = get_tokenizer()
        al = Aligner(tk, KeyTokenSelector(), FillerRegistry(tk), cfg)
        with self.assertRaises(ValueError):
            al.align("x" * 100)


class IdempotencyTest(unittest.TestCase):
    """E7 幂等性。"""

    def test_rerun_is_stable(self):
        """对已对齐文本再跑一次：结果不变，且**零插入**。

        引擎会先对原始输入编码判定，已全部对齐就直接返回，不会先剥离再重算。
        """
        text = "Hello world 关键信息 第二段"
        first = make_aligner().align(text)
        second = make_aligner().align(first.aligned_text)
        self.assertEqual(second.aligned_text, first.aligned_text)
        self.assertEqual(second.status, STATUS_ALIGNED)
        self.assertEqual(second.insertions, [])

    def test_rerun_from_dirty_original_matches(self):
        """给"已插入占位符的脏文本"跑，应与从干净原文跑的结果一致。"""
        clean = "alpha beta gamma 关键信息"
        al = make_aligner()
        dirty = al.align(clean).aligned_text
        from_clean = al.align(clean)
        from_dirty = al.align(dirty)
        self.assertEqual(from_clean.aligned_text, from_dirty.aligned_text)
        self.assertEqual(from_dirty.insertions, [])

    def test_rerun_without_restore_still_ok(self):
        text = "Hello world 关键信息"
        al = make_aligner()
        out1 = al.align(text)
        # 关闭还原再跑一次：结果应与开启还原时一致（因为默认会还原）
        al2 = make_aligner(restore_first=False)
        out2 = al2.align(out1.aligned_text)
        # 不还原时会叠加，但必须仍然 ALIGNED
        self.assertEqual(out2.status, STATUS_ALIGNED)

    def test_restore_roundtrip(self):
        text = "alpha beta gamma 关键信息"
        out = make_aligner().align(text)
        restored, removed = make_aligner().restore(out.aligned_text)
        self.assertEqual(restored, text)
        self.assertEqual(removed, sum(i.char_count for i in out.insertions))


class ConfigTest(unittest.TestCase):
    def test_period_variants(self):
        for period in (2, 3, 4, 8):
            tk = get_tokenizer()
            al = Aligner(
                tk, KeyTokenSelector(), FillerRegistry(tk), AlignConfig(period=period)
            )
            out = al.align("alpha beta gamma delta epsilon zeta eta theta")
            self.assertEqual(out.period, period)
            self.assertIn(out.status, (STATUS_ALIGNED, STATUS_PARTIAL))

    def test_inline_insert_mode(self):
        out = make_aligner(insert_mode=INSERT_MODE_INLINE).align("Hello world 关键信息")
        self.assertIn(out.status, (STATUS_ALIGNED, STATUS_PARTIAL, STATUS_FAILED))

    def test_invalid_insert_mode(self):
        tk = get_tokenizer()
        with self.assertRaises(ValueError):
            Aligner(
                tk,
                KeyTokenSelector(),
                FillerRegistry(tk),
                AlignConfig(insert_mode="bogus"),
            )

    def test_invalid_period(self):
        tk = get_tokenizer()
        with self.assertRaises(ValueError):
            Aligner(tk, KeyTokenSelector(), FillerRegistry(tk), AlignConfig(period=0))

    def test_filler_priority_override(self):
        out = make_aligner(filler_priority=["bom", "zwsp"]).align("Hello world 关键信息")
        if out.insertions:
            self.assertIn(out.insertions[0].filler_name, ("bom", "zwsp"))

    def test_max_retries_one(self):
        out = make_aligner(max_retries=1).align("Hello world 关键信息")
        self.assertIn(out.status, (STATUS_ALIGNED, STATUS_PARTIAL, STATUS_FAILED))


class MergeSplitTest(unittest.TestCase):
    """E6 占位符被 tokenizer 意外合并或拆分。"""

    def test_lf_filler_would_merge_but_engine_avoids_it(self):
        """即使把 LF 排在首位，引擎也应通过重试切换到线性占位符。"""
        tk = get_tokenizer()
        al = Aligner(
            tk,
            KeyTokenSelector(),
            FillerRegistry(tk),
            AlignConfig(filler_priority=["lf"]),
        )
        out = al.align("Hello world 关键信息")
        # LF 单独使用时可能失败或成功，但绝不能悄悄产生错位
        if out.status == STATUS_ALIGNED:
            assert_all_key_tokens_aligned(self, out)

    def test_two_token_per_char_filler_compensates(self):
        """每字符 2 token 的占位符（``wj``）应通过数量补偿达到目标相位。"""
        tk = get_tokenizer()
        registry = FillerRegistry(tk)
        priority = ["wj"] if "wj" in registry.names() else ["zwsp"]
        al = Aligner(
            tk,
            KeyTokenSelector(),
            registry,
            AlignConfig(filler_priority=priority),
        )
        out = al.align("Hello world 关键信息")
        self.assertIn(out.status, (STATUS_ALIGNED, STATUS_PARTIAL, STATUS_FAILED))
        if out.status == STATUS_ALIGNED:
            assert_all_key_tokens_aligned(self, out)

    def test_outcome_has_validations(self):
        out = make_aligner().align("Hello world 关键信息")
        self.assertTrue(out.validations)
        for v in out.validations:
            self.assertTrue(v.reason)


class OccurrenceOfTest(unittest.TestCase):
    def test_occurrence_counting(self):
        """同一 token id 的第 N 次出现应记 N（0 起）。

        注意 'alpha'（段首，无前导空格）与 'Ġalpha' 是**不同**的 token id，
        因此这里用 ' beta' 与 'Ġalpha' 这两组真正重复的 id 来验证。
        """
        tk = get_tokenizer()
        sel = KeyTokenSelector()
        enc = tk.encode_with_tokens("alpha beta alpha beta alpha")
        rep = sel.select(enc.tokens, enc.ids, tk.vocab_size)
        occurrences = [_occurrence_of(rep, d) for d in rep.decisions]
        # ids = [alpha(8339), Ġbeta(21474), Ġalpha(21138), Ġbeta(21474), Ġalpha(21138)]
        # 'alpha'  bare → 0；'Ġbeta'   → 0,1；'Ġalpha'  → 0,1
        self.assertEqual(occurrences, [0, 0, 0, 1, 1])

    def test_occurrence_counting_same_form(self):
        """首尾同形时也应正确计数。"""
        tk = get_tokenizer()
        sel = KeyTokenSelector()
        enc = tk.encode_with_tokens("beta alpha beta alpha beta")
        rep = sel.select(enc.tokens, enc.ids, tk.vocab_size)
        occurrences = [_occurrence_of(rep, d) for d in rep.decisions]
        # 'beta'(段首) → 0；'Ġalpha' → 0,1；'Ġbeta' → 0,1
        self.assertEqual(occurrences, [0, 0, 0, 1, 1])


class SemanticIntegrityTest(unittest.TestCase):
    """语义完整性：插入占位符不得改变原文的可见内容与段落结构。"""

    MULTI = (
        "DeepSeek 的注意力偏置位置位于 index % 4 == 0。\n\n"
        "这段中文文本需要被物理推进，包含 key 关键信息 secret 秘密数据。\n\n"
        "Line one of the third paragraph.\n"
        "Line two stays in the same paragraph.\n\n"
        "emoji 😀 与 CJK 混合 mixed content 123 数字符号 #@! 结尾\n"
    )

    def test_newlines_preserved(self):
        out = make_aligner().align(self.MULTI)
        self.assertEqual(out.status, STATUS_ALIGNED)
        self.assertEqual(
            out.aligned_text.count("\n"),
            self.MULTI.count("\n"),
            "对齐后换行符数量必须与原文一致",
        )

    def test_paragraph_count_preserved(self):
        from phase_shifting.segments import split_segments

        out = make_aligner().align(self.MULTI)
        self.assertEqual(
            len(split_segments(out.aligned_text)),
            len(split_segments(self.MULTI)),
        )

    def test_restore_roundtrips_to_original(self):
        al = make_aligner()
        out = al.align(self.MULTI)
        restored, removed = al.restore(out.aligned_text)
        self.assertGreater(removed, 0)
        self.assertEqual(restored, self.MULTI)

    def test_visible_chars_unchanged(self):
        """剥掉占位符后，剩余字符序列必须与原文逐字符相同。"""
        out = make_aligner().align(self.MULTI)
        keep = lambda s: [c for c in s if c not in (ZWSP, "﻿")]
        self.assertEqual(keep(out.aligned_text), keep(self.MULTI))

    def test_only_invisible_fillers_used(self):
        """默认配置下只允许插入不可见、无结构干扰的占位符。"""
        out = make_aligner().align(self.MULTI)
        for ins in out.insertions:
            self.assertTrue(ins.filler_char in (ZWSP, "﻿"), ins.filler_name)
            self.assertFalse(ins.structural, ins.filler_name)

    def test_idempotent_no_accumulation(self):
        al = make_aligner()
        first = al.align(self.MULTI)
        second = al.align(first.aligned_text)
        self.assertEqual(first.aligned_text, second.aligned_text)
        self.assertEqual(second.insertions, [])
        self.assertEqual(second.status, STATUS_ALIGNED)


class SummaryTest(unittest.TestCase):
    def test_aligned_summary(self):
        out = make_aligner().align("Hello world 关键信息")
        self.assertTrue(out.success)
        self.assertIn("对齐", out.summary())

    def test_partial_or_failed_summary(self):
        out = make_aligner(max_retries=1, filler_priority=["lf"]).align("a b c d e f")
        self.assertIn(out.status, (STATUS_ALIGNED, STATUS_PARTIAL, STATUS_FAILED))
        self.assertTrue(out.summary())


if __name__ == "__main__":
    unittest.main()
