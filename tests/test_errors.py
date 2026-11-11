# -*- coding: utf-8 -*-
"""结构化异常体系测试：层级、内置兼容与序列化摘要。"""

from __future__ import annotations

import unittest

from phase_shifting.errors import (
    AlignmentError,
    AnchorLostError,
    ConfigurationError,
    InputError,
    PhaseShiftingError,
    RetryExhaustedError,
    TokenizerError,
    UnknownFillerError,
    describe,
)


class HierarchyTest(unittest.TestCase):
    """所有异常都必须是 PhaseShiftingError 的子类。"""

    def test_all_subclasses(self):
        for cls in (
            ConfigurationError,
            UnknownFillerError,
            InputError,
            TokenizerError,
            AlignmentError,
            AnchorLostError,
            RetryExhaustedError,
        ):
            self.assertTrue(issubclass(cls, PhaseShiftingError), cls.__name__)

    def test_alignment_subclasses(self):
        self.assertTrue(issubclass(AnchorLostError, AlignmentError))
        self.assertTrue(issubclass(RetryExhaustedError, AlignmentError))

    def test_filler_error_is_configuration_error(self):
        self.assertTrue(issubclass(UnknownFillerError, ConfigurationError))

    def test_codes_are_stable(self):
        self.assertEqual(PhaseShiftingError.code, "phase_shifting_error")
        self.assertEqual(ConfigurationError.code, "configuration_error")
        self.assertEqual(UnknownFillerError.code, "unknown_filler")
        self.assertEqual(InputError.code, "input_error")
        self.assertEqual(TokenizerError.code, "tokenizer_error")
        self.assertEqual(AlignmentError.code, "alignment_error")
        self.assertEqual(AnchorLostError.code, "anchor_lost")
        self.assertEqual(RetryExhaustedError.code, "retry_exhausted")


class BuiltinCompatibilityTest(unittest.TestCase):
    """异常必须同时兼容既有基于内置异常的 except 分支。"""

    def test_configuration_error_is_value_error(self):
        with self.assertRaises(ValueError):
            raise ConfigurationError("bad value", {"k": 1})

    def test_input_error_is_value_error(self):
        with self.assertRaises(ValueError):
            raise InputError("too long")

    def test_tokenizer_error_is_runtime_error(self):
        with self.assertRaises(RuntimeError):
            raise TokenizerError("tokenizer unavailable")

    def test_alignment_error_is_runtime_error(self):
        with self.assertRaises(RuntimeError):
            raise AlignmentError("cannot align")

    def test_unknown_filler_is_key_error(self):
        with self.assertRaises(KeyError):
            raise UnknownFillerError("nope", ["zwsp", "bom"])

    def test_catch_all_with_base_class(self):
        exceptions = [
            ConfigurationError("a"),
            InputError("b"),
            TokenizerError("c"),
            AlignmentError("d"),
            AnchorLostError("t", 4),
            RetryExhaustedError("t", 3, 6),
        ]
        for exc in exceptions:
            with self.assertRaises(PhaseShiftingError):
                raise exc


class MessageAndDetailsTest(unittest.TestCase):
    """消息必须完整可读，details 必须结构化。"""

    def test_message_and_details_preserved(self):
        exc = ConfigurationError("slice_length 必须为正整数", {"slice_length": 0})
        self.assertEqual(str(exc), "slice_length 必须为正整数")
        self.assertEqual(exc.message, "slice_length 必须为正整数")
        self.assertEqual(exc.details, {"slice_length": 0})

    def test_default_details_is_empty_dict(self):
        exc = InputError("boom")
        self.assertEqual(exc.details, {})

    def test_unknown_filler_message_lists_candidates(self):
        exc = UnknownFillerError("zwnj", ["zwsp", "bom", "mix"])
        text = str(exc)
        self.assertIn("zwnj", text)
        self.assertIn("zwsp", text)
        self.assertIn("bom", text)
        self.assertIn("mix", text)
        # 不应被 KeyError 的 repr 引号包裹
        self.assertFalse(text.startswith("'") or text.startswith('"'))

    def test_unknown_filler_details(self):
        exc = UnknownFillerError("zwnj", ["zwsp"])
        self.assertEqual(exc.name, "zwnj")
        self.assertEqual(exc.available, ["zwsp"])
        self.assertEqual(exc.details["name"], "zwnj")

    def test_anchor_lost_details(self):
        exc = AnchorLostError("Hello", 4, "prefix mismatch")
        self.assertEqual(exc.details["token_text"], "Hello")
        self.assertEqual(exc.details["slice_length"], 4)
        self.assertIn("定位失败", str(exc))

    def test_retry_exhausted_details(self):
        exc = RetryExhaustedError("Hello", 3, 6, "last candidate merged")
        self.assertEqual(exc.details["token_index"], 3)
        self.assertEqual(exc.details["attempts"], 6)
        self.assertIn("6", str(exc))

    def test_custom_slice_length_flows_into_error_details(self):
        exc = AnchorLostError("世界", 16)
        self.assertEqual(exc.details["slice_length"], 16)


class DescribeTest(unittest.TestCase):
    """describe() 必须产出可 JSON 序列化的摘要。"""

    def test_describe_package_error(self):
        payload = describe(ConfigurationError("bad", {"field": "slice_length"}))
        self.assertEqual(payload["code"], "configuration_error")
        self.assertEqual(payload["message"], "bad")
        self.assertEqual(payload["details"], {"field": "slice_length"})

    def test_describe_without_details(self):
        payload = describe(InputError("boom"))
        self.assertEqual(payload, {"code": "input_error", "message": "boom"})

    def test_describe_foreign_error(self):
        payload = describe(ValueError("plain"))
        self.assertEqual(payload["code"], "ValueError")
        self.assertEqual(payload["message"], "plain")

    def test_describe_is_json_serializable(self):
        import json

        payload = describe(AnchorLostError("t", 4, "reason"))
        text = json.dumps(payload, ensure_ascii=False)
        self.assertIn("anchor_lost", text)


if __name__ == "__main__":
    unittest.main()
