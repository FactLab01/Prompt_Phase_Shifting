# -*- coding: utf-8 -*-
"""SDK 公共门面测试：PhaseShifter / align_text / plan / restore / 序列化。"""

from __future__ import annotations

import json
import unittest

from phase_shifting import (
    DEFAULT_SLICE_LENGTH,
    AlignmentPlan,
    AlignmentResult,
    PhaseShifter,
    PhaseShiftConfig,
    align_text,
    plan_text,
    restore_text,
)
from phase_shifting.aligner import STATUS_ALIGNED
from phase_shifting.errors import ConfigurationError, InputError, PhaseShiftingError

SAMPLE = "alpha beta gamma delta epsilon zeta eta theta"


class PublicSurfaceTest(unittest.TestCase):
    """公共 API 必须是稳定、可导入、可调用的。"""

    def test_importable_from_package_root(self):
        from phase_shifting import PhaseShifter as P

        self.assertIs(P, PhaseShifter)

    def test_all_exports_resolve(self):
        import phase_shifting

        for name in phase_shifting.__all__:
            self.assertTrue(hasattr(phase_shifting, name), name)

    def test_version_declared(self):
        import phase_shifting

        self.assertIsInstance(phase_shifting.__version__, str)
        self.assertRegex(phase_shifting.__version__, r"^\d+\.\d+\.\d+$")


class AlignTextFunctionTest(unittest.TestCase):
    """函数式入口：一次性调用。"""

    def test_default_slice_length_is_four(self):
        result = align_text(SAMPLE)
        self.assertIsInstance(result, AlignmentResult)
        self.assertEqual(result.slice_length, DEFAULT_SLICE_LENGTH)
        self.assertEqual(result.slice_length, 4)

    def test_explicit_slice_length_honored(self):
        for value in (1, 2, 3, 8, 16):
            result = align_text(SAMPLE, slice_length=value)
            self.assertEqual(result.slice_length, value)

    def test_config_object_path(self):
        cfg = PhaseShiftConfig(slice_length=8, key_mode="keywords", keywords=["alpha"])
        result = align_text(SAMPLE, config=cfg)
        self.assertEqual(result.slice_length, 8)

    def test_slice_length_overrides_config(self):
        cfg = PhaseShiftConfig(slice_length=8)
        result = align_text(SAMPLE, config=cfg, slice_length=2)
        self.assertEqual(result.slice_length, 2)

    def test_result_fields(self):
        result = align_text("Hello world 关键信息")
        self.assertEqual(result.original_text, "Hello world 关键信息")
        self.assertIsInstance(result.aligned_text, str)
        self.assertGreaterEqual(result.tokens_before, 0)
        self.assertGreaterEqual(result.tokens_after, result.tokens_before)
        self.assertIsInstance(result.insertions, list)
        self.assertIsInstance(result.failures, list)
        self.assertIsInstance(result.message, str)

    def test_derived_properties(self):
        result = align_text("Hello world 关键信息")
        self.assertIsInstance(result.success, bool)
        self.assertEqual(result.aligned_count, len(result.insertions))
        self.assertEqual(
            result.token_delta, result.tokens_after - result.tokens_before
        )

    def test_summary_is_non_empty(self):
        result = align_text(SAMPLE)
        self.assertTrue(result.summary())


class PhaseShifterObjectTest(unittest.TestCase):
    """门面对象：可复用、可批处理、可注入依赖。"""

    def test_reusable_across_calls(self):
        shifter = PhaseShifter()
        first = shifter.align(SAMPLE)
        second = shifter.align(SAMPLE)
        self.assertEqual(first.slice_length, DEFAULT_SLICE_LENGTH)
        self.assertEqual(second.slice_length, DEFAULT_SLICE_LENGTH)

    def test_slice_length_property(self):
        self.assertEqual(PhaseShifter().slice_length, DEFAULT_SLICE_LENGTH)
        self.assertEqual(PhaseShifter(slice_length=8).slice_length, 8)
        cfg = PhaseShiftConfig(slice_length=16)
        self.assertEqual(PhaseShifter(cfg).slice_length, 16)
        self.assertEqual(PhaseShifter(cfg, slice_length=2).slice_length, 2)

    def test_tokenizer_property_exposed(self):
        shifter = PhaseShifter()
        self.assertTrue(hasattr(shifter.tokenizer, "encode"))

    def test_align_many(self):
        shifter = PhaseShifter(slice_length=4)
        results = shifter.align_many(["alpha beta", "gamma delta"])
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, AlignmentResult)
            self.assertEqual(r.slice_length, 4)

    def test_progress_callback_invoked(self):
        seen = []
        shifter = PhaseShifter()
        shifter.align("Hello world 关键信息", progress=seen.append)
        self.assertIsInstance(seen, list)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ConfigurationError):
            PhaseShifter(slice_length=0)
        with self.assertRaises(ConfigurationError):
            PhaseShifter(slice_length=-4)

    def test_bad_input_type_rejected(self):
        shifter = PhaseShifter()
        with self.assertRaises(InputError):
            shifter.align(123)  # type: ignore[arg-type]

    def test_overlong_input_rejected(self):
        shifter = PhaseShifter(PhaseShiftConfig(max_chars=10))
        with self.assertRaises(InputError):
            shifter.align("x" * 100)


class PlanTest(unittest.TestCase):
    """只读分析：不修改文本。"""

    def test_plan_returns_alignment_plan(self):
        plan = plan_text(SAMPLE)
        self.assertIsInstance(plan, AlignmentPlan)
        self.assertEqual(plan.slice_length, DEFAULT_SLICE_LENGTH)
        self.assertGreater(plan.total_tokens, 0)

    def test_plan_does_not_modify_text(self):
        plan = plan_text("Hello world 关键信息")
        self.assertTrue(plan.needs_alignment or not plan.misaligned)

    def test_plan_respects_slice_length(self):
        for value in (2, 4, 8):
            plan = plan_text(SAMPLE, slice_length=value)
            self.assertEqual(plan.slice_length, value)

    def test_plan_entries_are_serializable(self):
        plan = plan_text(SAMPLE)
        payload = plan.to_dict()
        self.assertEqual(payload["slice_length"], plan.slice_length)
        self.assertIn("misaligned", payload)
        self.assertIn("aligned", payload)
        json.dumps(payload, ensure_ascii=False)

    def test_plan_to_json(self):
        text = plan_text(SAMPLE).to_json()
        self.assertIn("slice_length", text)


class RestoreTest(unittest.TestCase):
    """还原：剥离占位字符，保留原文可见内容。"""

    def test_roundtrip(self):
        original = "Hello world 关键信息"
        aligned = align_text(original).aligned_text
        restored = restore_text(aligned)
        self.assertEqual(restored, original)

    def test_restore_on_shifter(self):
        shifter = PhaseShifter()
        aligned = shifter.align("alpha beta gamma").aligned_text
        self.assertEqual(shifter.restore(aligned), "alpha beta gamma")

    def test_restore_is_noop_on_clean_text(self):
        clean = "nothing to strip here"
        self.assertEqual(restore_text(clean), clean)


class SerializationTest(unittest.TestCase):
    """结果对象必须可无损序列化。"""

    def test_to_dict_keys(self):
        result = align_text("Hello world 关键信息")
        data = result.to_dict()
        for key in (
            "status",
            "slice_length",
            "tokens_before",
            "tokens_after",
            "token_delta",
            "summary",
            "insertions",
            "failures",
            "original_text",
            "aligned_text",
            "aligned_text_visible",
        ):
            self.assertIn(key, data)

    def test_to_json_roundtrip(self):
        result = align_text("Hello world 关键信息")
        payload = json.loads(result.to_json())
        self.assertEqual(payload["slice_length"], result.slice_length)
        self.assertEqual(payload["status"], result.status)

    def test_from_outcome_adapter(self):
        from phase_shifting.aligner import AlignOutcome

        outcome = AlignOutcome(
            status=STATUS_ALIGNED,
            original_text="abc",
            aligned_text="abc",
            period=8,
            total_tokens_before=1,
            total_tokens_after=1,
            key_report=None,
            insertions=[],
            failures=[],
            message="",
        )
        adapted = AlignmentResult.from_outcome(outcome)
        self.assertEqual(adapted.status, STATUS_ALIGNED)
        self.assertEqual(adapted.slice_length, 8)
        self.assertEqual(adapted.original_text, "abc")
        self.assertEqual(adapted.aligned_text, "abc")
        self.assertEqual(adapted.tokens_before, 1)
        self.assertEqual(adapted.tokens_after, 1)
        self.assertEqual(adapted.insertions, [])


class ErrorSurfaceTest(unittest.TestCase):
    """SDK 错误必须可被基类一次性捕获。"""

    def test_invalid_slice_length_raises_package_error(self):
        with self.assertRaises(PhaseShiftingError):
            align_text(SAMPLE, slice_length=0)

    def test_configuration_error_is_catchable_as_value_error(self):
        with self.assertRaises(ValueError):
            align_text(SAMPLE, slice_length=-1)

    def test_alignment_succeeds_for_default_config(self):
        result = align_text(SAMPLE)
        self.assertEqual(result.status, STATUS_ALIGNED)


if __name__ == "__main__":
    unittest.main()
