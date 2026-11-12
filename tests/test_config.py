# -*- coding: utf-8 -*-
"""配置层测试：切片长度参数化、校验与序列化。"""

from __future__ import annotations

import unittest

from phase_shifting.config import PhaseShiftConfig, default_config
from phase_shifting.constants import (
    DEFAULT_INSERT_MODE,
    DEFAULT_KEY_MODE,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_SLICE_LENGTH,
    DEFAULT_WEIGHT_THRESHOLD,
)
from phase_shifting.errors import ConfigurationError


class DefaultsTest(unittest.TestCase):
    """默认值必须与常量层一致（唯一事实来源）。"""

    def test_defaults_match_constants(self):
        cfg = PhaseShiftConfig()
        self.assertEqual(cfg.slice_length, DEFAULT_SLICE_LENGTH)
        self.assertEqual(cfg.key_mode, DEFAULT_KEY_MODE)
        self.assertEqual(cfg.insert_mode, DEFAULT_INSERT_MODE)
        self.assertEqual(cfg.max_retries, DEFAULT_MAX_RETRIES)
        self.assertEqual(cfg.max_chars, DEFAULT_MAX_CHARS)
        self.assertAlmostEqual(cfg.weight_threshold, DEFAULT_WEIGHT_THRESHOLD)

    def test_default_config_factory(self):
        cfg = default_config()
        self.assertIsInstance(cfg, PhaseShiftConfig)
        self.assertEqual(cfg.slice_length, DEFAULT_SLICE_LENGTH)

    def test_default_config_with_overrides(self):
        cfg = default_config(slice_length=8, key_mode="keywords", keywords=["a", "b"])
        self.assertEqual(cfg.slice_length, 8)
        self.assertEqual(cfg.key_mode, "keywords")
        self.assertEqual(list(cfg.keywords), ["a", "b"])


class SliceLengthParameterizationTest(unittest.TestCase):
    """切片长度必须是显式可配置参数，接受任意正整数。"""

    def test_default_is_four(self):
        self.assertEqual(DEFAULT_SLICE_LENGTH, 4)
        self.assertEqual(PhaseShiftConfig().slice_length, 4)

    def test_arbitrary_positive_values_accepted(self):
        for value in (1, 2, 3, 4, 5, 7, 8, 16, 32, 128, 4096):
            cfg = PhaseShiftConfig(slice_length=value)
            self.assertEqual(cfg.slice_length, value)
            self.assertEqual(cfg.period, value)

    def test_period_is_readonly_alias(self):
        cfg = PhaseShiftConfig(slice_length=6)
        self.assertEqual(cfg.period, 6)
        with self.assertRaises(AttributeError):
            cfg.period = 8  # type: ignore[misc]

    def test_with_slice_length_returns_new_instance(self):
        base = PhaseShiftConfig(slice_length=4, key_mode="keywords", keywords=["x"])
        derived = base.with_slice_length(16)
        self.assertEqual(derived.slice_length, 16)
        self.assertEqual(derived.key_mode, "keywords")
        self.assertEqual(base.slice_length, 4)

    def test_invalid_slice_lengths_rejected(self):
        for bad in (0, -1, -4, -100):
            with self.assertRaises(ConfigurationError) as ctx:
                PhaseShiftConfig(slice_length=bad)
            self.assertEqual(ctx.exception.code, "configuration_error")
            self.assertIn("slice_length", ctx.exception.details)

    def test_non_int_slice_length_rejected(self):
        for bad in (4.0, "4", None, True):
            with self.assertRaises(ConfigurationError):
                PhaseShiftConfig(slice_length=bad)


class ValidationTest(unittest.TestCase):
    """其余字段的校验同样走结构化异常。"""

    def test_invalid_key_mode(self):
        with self.assertRaises(ConfigurationError):
            PhaseShiftConfig(key_mode="nope")

    def test_keywords_mode_requires_keywords(self):
        with self.assertRaises(ConfigurationError):
            PhaseShiftConfig(key_mode="keywords", keywords=[])
        with self.assertRaises(ConfigurationError):
            PhaseShiftConfig(key_mode="keywords", keywords=None)

    def test_keywords_mode_accepts_keywords(self):
        cfg = PhaseShiftConfig(key_mode="keywords", keywords=["secret"])
        self.assertEqual(list(cfg.keywords), ["secret"])

    def test_insert_mode_is_validated_by_engine_layer(self):
        # insert_mode 的合法性由引擎配置层（AlignConfig）负责校验；
        # 公共配置对象只做类型归一化，便于提前暴露拼写错误但不阻断构造。
        cfg = PhaseShiftConfig(insert_mode="bogus")
        self.assertEqual(cfg.insert_mode, "bogus")
        from phase_shifting.aligner import AlignConfig

        with self.assertRaises(ValueError):
            AlignConfig(insert_mode="bogus")

    def test_invalid_weight_threshold(self):
        for bad in (-0.1, 1.1, 2.0):
            with self.assertRaises(ConfigurationError):
                PhaseShiftConfig(weight_threshold=bad)

    def test_invalid_max_retries_and_chars(self):
        with self.assertRaises(ConfigurationError):
            PhaseShiftConfig(max_retries=0)
        with self.assertRaises(ConfigurationError):
            PhaseShiftConfig(max_chars=0)

    def test_invalid_filler_priority_type(self):
        with self.assertRaises(ConfigurationError):
            PhaseShiftConfig(filler_priority="zwsp")  # type: ignore[arg-type]


class SerializationTest(unittest.TestCase):
    """配置必须可无损往返，便于跨进程 / 跨语言传递。"""

    def test_to_dict_contains_slice_length(self):
        cfg = PhaseShiftConfig(slice_length=8)
        data = cfg.to_dict()
        self.assertEqual(data["slice_length"], 8)
        self.assertIn("key_mode", data)
        self.assertIn("insert_mode", data)

    def test_roundtrip(self):
        cfg = PhaseShiftConfig(
            slice_length=16,
            key_mode="keywords",
            keywords=["关键", "secret"],
            weight_threshold=0.7,
            insert_mode="segment",
            filler_priority=["zwsp", "bom"],
            max_retries=3,
            max_chars=1000,
            restore_first=False,
        )
        clone = PhaseShiftConfig.from_dict(cfg.to_dict())
        self.assertEqual(clone.slice_length, 16)
        self.assertEqual(clone.key_mode, "keywords")
        self.assertEqual(list(clone.keywords), ["关键", "secret"])
        self.assertEqual(clone.insert_mode, "segment")
        self.assertEqual(list(clone.filler_priority), ["zwsp", "bom"])
        self.assertEqual(clone.max_retries, 3)
        self.assertEqual(clone.max_chars, 1000)
        self.assertFalse(clone.restore_first)
        self.assertEqual(clone.to_dict(), cfg.to_dict())

    def test_from_dict_defaults_when_missing(self):
        cfg = PhaseShiftConfig.from_dict({})
        self.assertEqual(cfg.slice_length, DEFAULT_SLICE_LENGTH)

    def test_from_dict_none_is_treated_as_empty(self):
        cfg = PhaseShiftConfig.from_dict(None)  # type: ignore[arg-type]
        self.assertEqual(cfg.slice_length, DEFAULT_SLICE_LENGTH)

    def test_from_dict_ignores_unknown_keys(self):
        # 未知键被静默忽略：便于向前兼容未来新增字段
        cfg = PhaseShiftConfig.from_dict({"slice_length": 8, "future_field": 1})
        self.assertEqual(cfg.slice_length, 8)

    def test_from_dict_still_validates_known_keys(self):
        with self.assertRaises(ConfigurationError):
            PhaseShiftConfig.from_dict({"slice_length": 0})


if __name__ == "__main__":
    unittest.main()
