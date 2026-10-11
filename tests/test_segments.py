# -*- coding: utf-8 -*-
"""段落切分与插入点测试。"""

from __future__ import annotations

import unittest

from phase_shifting.segments import (
    byte_to_char_map,
    map_tokens_to_segments,
    segment_of_offset,
    split_segments,
    strip_fillers,
    token_bytes,
    token_offsets,
    token_start_offsets,
)
from phase_shifting.tokenizer_api import get_tokenizer


class SplitSegmentsTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(split_segments(""), [])

    def test_single_paragraph(self):
        segs = split_segments("hello world")
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].text, "hello world")

    def test_two_paragraphs(self):
        text = "第一段。\n\n第二段。"
        segs = split_segments(text)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0].text, "第一段。")
        self.assertEqual(segs[1].text, "第二段。")

    def test_crlf_paragraphs(self):
        text = "a\r\n\r\nb"
        segs = split_segments(text)
        self.assertEqual(len(segs), 2)

    def test_single_newline_is_not_a_break(self):
        segs = split_segments("line1\nline2")
        self.assertEqual(len(segs), 1)

    def test_three_paragraphs(self):
        text = "a\n\nb\n\nc"
        segs = split_segments(text)
        self.assertEqual(len(segs), 3)
        self.assertEqual([s.index for s in segs], [0, 1, 2])


class InsertOffsetTest(unittest.TestCase):
    def test_leading_newline_skipped(self):
        text = "\n\n正文内容"
        segs = split_segments(text)
        # "\n\n" 是段落分隔，因此分成 2 段："" 和 "正文内容"
        self.assertEqual(len(segs), 2)
        offset = segs[1].insert_offset()
        self.assertEqual(text[offset:], "正文内容")

    def test_leading_newline_within_segment(self):
        text = "标题\n正文内容"
        segs = split_segments(text)
        self.assertEqual(len(segs), 1)
        # 单个 \n 不是分隔符，插入点落在段首（"标"之前）
        self.assertEqual(segs[0].insert_offset(), 0)

    def test_leading_spaces_skipped(self):
        seg = split_segments("   abc")[0]
        self.assertEqual(seg.insert_offset(), 3)


class SegmentOfOffsetTest(unittest.TestCase):
    def test_lookup(self):
        text = "aaa\n\nbbb"
        segs = split_segments(text)
        self.assertIs(segment_of_offset(segs, 0), segs[0])
        self.assertIs(segment_of_offset(segs, 5), segs[1])

    def test_offset_at_end(self):
        segs = split_segments("abc")
        self.assertIs(segment_of_offset(segs, 3), segs[0])

    def test_empty_segments(self):
        self.assertIsNone(segment_of_offset([], 0))


class TokenOffsetsTest(unittest.TestCase):
    def test_offsets_monotonic(self):
        from phase_shifting.tokenizer_api import get_tokenizer

        tk = get_tokenizer()
        text = "Hello world 关键信息"
        enc = tk.encode_with_tokens(text)
        offsets = token_offsets(text, enc.tokens)
        self.assertEqual(len(offsets), len(enc.tokens))
        for i in range(1, len(offsets)):
            self.assertGreaterEqual(offsets[i][0], offsets[i - 1][0])

    def test_empty(self):
        self.assertEqual(token_offsets("", []), [])

    def test_token_start_offsets_basic(self):
        from phase_shifting.segments import token_start_offsets

        tk = get_tokenizer()
        text = "alpha beta gamma"
        enc = tk.encode_with_tokens(text)
        offs = token_start_offsets(text, enc.tokens)
        self.assertEqual(len(offs), len(enc.tokens))
        # 'alpha' 从 0 开始，' beta' 的前导空格在 5，'gamma' 的空格在 10
        self.assertEqual(offs, [0, 5, 10])

    def test_token_start_offsets_prefix_property(self):
        """核心不变量：在 offs[k] 处切分，前缀编码必须等于 ids[:k]。

        例外（已在 docs/SPEC.md 记录）：当某个 token 从一个多字节字符**内部**
        开始（例如 emoji '😀' 被 BPE 切成 3+1 字节，第 4 字节单独成 token），
        插入点只能回退到该字符起点，此时前缀不保证逐字节相等。这类 token 由
        对齐引擎的 ``_OFFSET_CANDIDATES`` 重试机制吸收，不影响收敛性。
        """
        from phase_shifting.segments import token_start_offsets

        tk = get_tokenizer()
        for text in (
            "alpha beta gamma",
            "hello world, 这是一段中文测试！",
            "line one\n\nline two\nline three",
            "  leading and\ttrailing  ",
            "单段落无空格中文文本测试相位移动效果",
            "abc123def 456ghi",
        ):
            enc = tk.encode_with_tokens(text)
            offs = token_start_offsets(text, enc.tokens)
            self.assertIsNotNone(offs, text)
            for k in range(len(enc.ids) + 1):
                off = offs[k] if k < len(offs) else len(text)
                self.assertEqual(tk.encode(text[:off]), enc.ids[:k], "%r k=%d" % (text, k))

    def test_token_start_offsets_never_splits_char(self):
        """插入点永远不落在多字节字符内部（这保证文本不会被劈坏）。"""
        from phase_shifting.segments import token_start_offsets

        tk = get_tokenizer()
        for text in ("emoji 😀 test 🚀 done", "a😀b😀c", "🎉🎊🎈 party"):
            enc = tk.encode_with_tokens(text)
            offs = token_start_offsets(text, enc.tokens)
            self.assertIsNotNone(offs, text)
            for off in offs:
                self.assertTrue(0 <= off <= len(text), text)
                # 从 off 处切分后再拼接，必须无损还原
                self.assertEqual(text[:off] + text[off:], text, text)

    def test_token_start_offsets_empty(self):
        from phase_shifting.segments import token_start_offsets

        self.assertEqual(token_start_offsets("", []), [])

    def test_token_bytes(self):
        from phase_shifting.segments import token_bytes

        self.assertEqual(token_bytes("alpha"), b"alpha")
        # ByteLevel 的 Ġ 在源码里是一个独立字符，等价于 ASCII 空格
        self.assertEqual(token_bytes("Ġbeta"), b" beta")
        # ZWSP 的 ByteLevel 串 → UTF-8 字节 E2 80 8B
        self.assertEqual(len(token_bytes("​")), 3)

    def test_byte_to_char_map(self):
        from phase_shifting.segments import byte_to_char_map

        m = byte_to_char_map("a😀b")
        # 'a'=1 字节，'😀'=4 字节，'b'=1 字节 → 共 6 字节 + 1
        self.assertEqual(len(m), 7)
        self.assertEqual(m[0], 0)
        # emoji 的 4 个字节都映射回字符 1
        self.assertEqual([m[1], m[2], m[3], m[4]], [1, 1, 1, 1])
        self.assertEqual(m[5], 2)
        self.assertEqual(m[6], 3)


class MapTokensToSegmentsTest(unittest.TestCase):
    def test_all_in_one_segment(self):
        from phase_shifting.tokenizer_api import get_tokenizer

        tk = get_tokenizer()
        text = "Hello world"
        enc = tk.encode_with_tokens(text)
        segs = split_segments(text)
        mapping = map_tokens_to_segments(text, enc.tokens, segs)
        self.assertEqual(len(mapping), len(enc.tokens))
        for seg in mapping:
            self.assertIsNotNone(seg)
            self.assertEqual(seg.index, 0)

    def test_no_segments(self):
        mapping = map_tokens_to_segments("x", ["x"], [])
        self.assertEqual(mapping, [None])


class StripFillersTest(unittest.TestCase):
    def test_strip_zwsp(self):
        text, removed = strip_fillers("a​b​c", ["​"])
        self.assertEqual(text, "abc")
        self.assertEqual(removed, 2)

    def test_strip_multiple_kinds(self):
        text, removed = strip_fillers("a​b﻿c", ["​", "﻿"])
        self.assertEqual(text, "abc")
        self.assertEqual(removed, 2)

    def test_strip_nothing(self):
        text, removed = strip_fillers("abc", ["​"])
        self.assertEqual(text, "abc")
        self.assertEqual(removed, 0)

    def test_strip_empty(self):
        self.assertEqual(strip_fillers("", ["​"]), ("", 0))
        self.assertEqual(strip_fillers("abc", []), ("abc", 0))

    def test_idempotent_strip(self):
        once, n1 = strip_fillers("a​​b", ["​"])
        twice, n2 = strip_fillers(once, ["​"])
        self.assertEqual(once, twice)
        self.assertEqual(n2, 0)


if __name__ == "__main__":
    unittest.main()
