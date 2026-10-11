# -*- coding: utf-8 -*-
"""动态校验器测试。"""

from __future__ import annotations

import unittest

from phase_shifting.tokenizer_api import get_tokenizer
from phase_shifting.validator import DynamicValidator


class LocateAnchorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tk = get_tokenizer()
        cls.val = DynamicValidator(cls.tk, period=4)

    def test_locate_simple(self):
        text = "Hello world 关键信息"
        ids = self.tk.encode(text)
        anchor = [ids[3]]
        res = self.val.locate_anchor(anchor, ids, hint=3)
        self.assertTrue(res.found)
        self.assertEqual(res.index, 3)

    def test_locate_with_hint_window(self):
        text = "alpha beta gamma delta epsilon"
        ids = self.tk.encode(text)
        anchor = [ids[4]]
        res = self.val.locate_anchor(anchor, ids, hint=4)
        self.assertTrue(res.found)
        self.assertEqual(res.index, 4)

    def test_locate_missing_anchor(self):
        ids = [1, 2, 3]
        res = self.val.locate_anchor([999], ids, hint=1)
        self.assertFalse(res.found)
        self.assertIn("丢失", res.detail)

    def test_locate_empty_anchor(self):
        res = self.val.locate_anchor([], [1, 2, 3])
        self.assertFalse(res.found)

    def test_locate_anchor_longer_than_stream(self):
        res = self.val.locate_anchor([1, 2, 3, 4], [1, 2])
        self.assertFalse(res.found)

    def test_locate_without_hint(self):
        ids = [5, 6, 7, 5]
        res = self.val.locate_anchor([5], ids, hint=None)
        self.assertTrue(res.found)
        self.assertEqual(res.index, 0)

    def test_locate_nearest_to_hint(self):
        ids = [9, 9, 9, 9]
        res = self.val.locate_anchor([9], ids, hint=2)
        self.assertTrue(res.found)
        self.assertEqual(res.index, 2)


class ValidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tk = get_tokenizer()
        cls.val = DynamicValidator(cls.tk, period=4)

    def test_validate_aligned(self):
        # 插入 4 个 ZWSP = 4 token，Hello 从 0 → 4，4 % 4 == 0 → 对齐
        before = "Hello world"
        after = "​​​​Hello world"
        ids = self.tk.encode(before)
        anchor = [ids[0]]
        res = self.val.validate(before, after, anchor, expected_index=4, expected_inserted_tokens=4)
        self.assertTrue(res.ok, res.reason)
        self.assertTrue(res.aligned)
        self.assertEqual(res.new_index, 4)
        self.assertEqual(res.inserted_tokens, 4)

    def test_validate_not_aligned(self):
        before = "Hello world"
        after = "​Hello world"  # 1 个 ZWSP → Hello 落在 index 1
        ids = self.tk.encode(before)
        anchor = [ids[0]]
        res = self.val.validate(before, after, anchor, expected_index=1, expected_inserted_tokens=1)
        self.assertFalse(res.ok)
        self.assertFalse(res.aligned)
        self.assertEqual(res.new_index, 1)
        self.assertIn("未对齐", res.reason)

    def test_validate_token_count_mismatch_noted(self):
        before = "Hello world"
        after = "\n\nHello world"  # 2 个 LF 被合并成 1 token，与预期 2 不符
        ids = self.tk.encode(before)
        anchor = [ids[0]]
        res = self.val.validate(before, after, anchor, expected_index=2, expected_inserted_tokens=2)
        self.assertTrue(res.notes)
        self.assertTrue(any("不符" in n for n in res.notes))
        self.assertEqual(res.inserted_tokens, 1)

    def test_validate_anchor_lost(self):
        before = "关键信息"
        after = "完全不同的内容"
        ids = self.tk.encode(before)
        anchor = [ids[0]]
        res = self.val.validate(before, after, anchor, expected_index=1, expected_inserted_tokens=1)
        self.assertFalse(res.ok)
        self.assertIn("定位失败", res.reason)

    def test_validate_empty_texts(self):
        res = self.val.validate("", "", [], 0, 0)
        self.assertFalse(res.ok)

    def test_period_must_be_positive(self):
        with self.assertRaises(ValueError):
            DynamicValidator(self.tk, period=0)

    def test_is_aligned_and_phase(self):
        self.assertTrue(self.val.is_aligned(0))
        self.assertTrue(self.val.is_aligned(4))
        self.assertFalse(self.val.is_aligned(1))
        self.assertEqual(self.val.phase(5), 1)


if __name__ == "__main__":
    unittest.main()
