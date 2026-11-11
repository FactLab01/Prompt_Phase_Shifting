# -*- coding: utf-8 -*-
"""C-ABI 原语层测试：签名约定、基础类型与切片长度参数化。"""

from __future__ import annotations

import json
import unittest

from phase_shifting import primitives
from phase_shifting.constants import DEFAULT_SLICE_LENGTH

SAMPLE = "alpha beta gamma delta epsilon zeta eta theta"


class SignatureContractTest(unittest.TestCase):
    """原语层必须保持最窄签名约定，供未来绑定生成器读取。"""

    def test_manifest_covers_all_primitives(self):
        for name in (
            "encode",
            "decode",
            "token_strings",
            "token_count",
            "align_text",
            "align_text_detailed",
            "restore_text",
        ):
            self.assertIn(name, primitives.PRIMITIVE_SIGNATURES, name)

    def test_manifest_is_json_serializable(self):
        text = json.dumps(primitives.PRIMITIVE_SIGNATURES, ensure_ascii=False)
        self.assertIn("align_text", text)

    def test_align_manifest_mentions_slice_length(self):
        self.assertIn("slice_length", primitives.PRIMITIVE_SIGNATURES["align_text"])
        self.assertIn(
            "slice_length", primitives.PRIMITIVE_SIGNATURES["align_text_detailed"]
        )

    def test_no_varargs_or_kwargs(self):
        # 位置参数化是硬约定：不允许 *args / **kwargs 穿越 ABI 边界
        import inspect

        for name in primitives.PRIMITIVE_SIGNATURES:
            func = getattr(primitives, name)
            signature = inspect.signature(func)
            for param in signature.parameters.values():
                self.assertEqual(
                    param.kind,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    "%s 的参数 %s 不是位置参数" % (name, param.name),
                )

    def test_all_parameters_have_annotations(self):
        import inspect

        for name in primitives.PRIMITIVE_SIGNATURES:
            func = getattr(primitives, name)
            hints = inspect.signature(func).parameters
            for param_name, param in hints.items():
                self.assertIsNot(
                    param.annotation,
                    inspect.Parameter.empty,
                    "%s.%s 缺少类型标注" % (name, param_name),
                )


class BasicTypePrimitivesTest(unittest.TestCase):
    """基础编解码原语。"""

    def test_encode_returns_ints(self):
        ids = primitives.encode(SAMPLE)
        self.assertIsInstance(ids, list)
        self.assertTrue(ids)
        for i in ids:
            self.assertIsInstance(i, int)

    def test_encode_empty(self):
        self.assertEqual(primitives.encode(""), [])

    def test_decode_roundtrip(self):
        ids = primitives.encode(SAMPLE)
        self.assertEqual(primitives.decode(ids), SAMPLE)

    def test_decode_empty(self):
        self.assertEqual(primitives.decode([]), "")

    def test_token_strings_length_matches(self):
        ids = primitives.encode(SAMPLE)
        tokens = primitives.token_strings(ids)
        self.assertEqual(len(tokens), len(ids))
        for t in tokens:
            self.assertIsInstance(t, str)

    def test_token_count(self):
        self.assertEqual(primitives.token_count(SAMPLE), len(primitives.encode(SAMPLE)))
        self.assertEqual(primitives.token_count(""), 0)

    def test_accepts_tuple_ids(self):
        ids = tuple(primitives.encode(SAMPLE))
        self.assertEqual(primitives.decode(ids), SAMPLE)


class AlignPrimitivesTest(unittest.TestCase):
    """对齐原语：切片长度必须可显式传入。"""

    def test_align_text_returns_str(self):
        aligned = primitives.align_text(SAMPLE)
        self.assertIsInstance(aligned, str)

    def test_default_slice_length(self):
        payload = primitives.align_text_detailed(SAMPLE)
        self.assertEqual(payload["slice_length"], DEFAULT_SLICE_LENGTH)
        self.assertEqual(payload["slice_length"], 4)

    def test_explicit_slice_length(self):
        for value in (1, 2, 3, 8, 16):
            payload = primitives.align_text_detailed(SAMPLE, value)
            self.assertEqual(payload["slice_length"], value)

    def test_keyword_slice_length(self):
        payload = primitives.align_text_detailed(SAMPLE, slice_length=8)
        self.assertEqual(payload["slice_length"], 8)

    def test_detailed_payload_keys(self):
        payload = primitives.align_text_detailed(SAMPLE)
        for key in (
            "status",
            "slice_length",
            "original_text",
            "aligned_text",
            "tokens_before",
            "tokens_after",
            "insertions",
            "failures",
            "summary",
        ):
            self.assertIn(key, payload)

    def test_detailed_payload_is_json_serializable(self):
        payload = primitives.align_text_detailed(SAMPLE)
        json.dumps(payload, ensure_ascii=False)

    def test_restore_text(self):
        aligned = primitives.align_text(SAMPLE)
        self.assertEqual(primitives.restore_text(aligned), SAMPLE)

    def test_restore_text_noop_on_clean(self):
        clean = "plain text without placeholders"
        self.assertEqual(primitives.restore_text(clean), clean)

    def test_align_restore_roundtrip_various_slice_lengths(self):
        for value in (1, 2, 4, 8):
            aligned = primitives.align_text(SAMPLE, value)
            self.assertEqual(primitives.restore_text(aligned), SAMPLE)


if __name__ == "__main__":
    unittest.main()
