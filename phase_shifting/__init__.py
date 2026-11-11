# -*- coding: utf-8 -*-
"""phase_shifting — 前置相位偏移 SDK。

把用户输入中的关键 token 通过"动态插入无害占位字符"物理推进到目标模型的
注意力偏置位置（``token_index % slice_length == 0``）。

快速上手::

    from phase_shifting import PhaseShifter, PhaseShiftConfig, align_text

    # 方式一：门面对象（可复用，适合批处理）
    shifter = PhaseShifter(PhaseShiftConfig(slice_length=4))
    result = shifter.align("DeepSeek 的注意力偏置位置")
    assert result.success

    # 方式二：函数式入口（一次性调用）
    result = align_text("alpha beta gamma", slice_length=8)

    # 方式三：只读分析，不修改文本
    plan = shifter.plan("alpha beta gamma")

切片长度（``slice_length``）是目标模型一次前向处理的 token 数，默认 4；
不同模型应显式传入。原语层 :mod:`phase_shifting.primitives` 提供面向未来
C-ABI 绑定的稳定窄接口。

设计文档见 ``docs/SPEC.md``；使用说明见 ``README.md``。
"""

from __future__ import annotations

__version__ = "1.0.0"

# ------------------------------------------------------------------ 常量
from .constants import (
    DEFAULT_INSERT_MODE,
    DEFAULT_KEY_MODE,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_SLICE_LENGTH,
    DEFAULT_WEIGHT_THRESHOLD,
)

# ------------------------------------------------------------------ 异常
from .errors import (
    AlignmentError,
    AnchorLostError,
    ConfigurationError,
    InputError,
    PhaseShiftingError,
    RetryExhaustedError,
    TokenizerError,
    UnknownFillerError,
)

# ------------------------------------------------------------------ 配置
from .config import PhaseShiftConfig, default_config

# ------------------------------------------------------------------ SDK 门面
from .api import (
    AlignmentPlan,
    AlignmentResult,
    PhaseShifter,
    align_text,
    plan_text,
    restore_text,
)

# ------------------------------------------------------------------ 原语层
from .primitives import PRIMITIVE_SIGNATURES

__all__ = [
    "__version__",
    # 常量
    "DEFAULT_SLICE_LENGTH",
    "DEFAULT_KEY_MODE",
    "DEFAULT_INSERT_MODE",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_WEIGHT_THRESHOLD",
    # 异常
    "PhaseShiftingError",
    "ConfigurationError",
    "UnknownFillerError",
    "InputError",
    "TokenizerError",
    "AlignmentError",
    "AnchorLostError",
    "RetryExhaustedError",
    # 配置
    "PhaseShiftConfig",
    "default_config",
    # SDK 门面
    "PhaseShifter",
    "AlignmentResult",
    "AlignmentPlan",
    "align_text",
    "plan_text",
    "restore_text",
    # 原语层
    "PRIMITIVE_SIGNATURES",
]
