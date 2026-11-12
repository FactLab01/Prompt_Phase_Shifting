# -*- coding: utf-8 -*-
"""SDK 配置对象。

:class:`PhaseShiftConfig` 是调用方与 SDK 之间的唯一配置契约，字段稳定、
可序列化、带校验。其中 **``slice_length``（切片长度）** 是核心参数：
不同模型一次前向处理的 token 数可能不同，调用方应依据目标模型的特性显式
传入；默认取 :data:`phase_shifting.constants.DEFAULT_SLICE_LENGTH`。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .constants import (
    DEFAULT_INSERT_MODE,
    DEFAULT_KEY_MODE,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_SLICE_LENGTH,
    DEFAULT_WEIGHT_THRESHOLD,
    DEFAULT_WINDOW_AGG,
    DEFAULT_WINDOW_SIZE,
)
from .errors import ConfigurationError
from .keytokens import KEY_MODES, KEY_MODE_WEIGHT
from .weights import WINDOW_AGGS

__all__ = ["PhaseShiftConfig"]


def _validate_slice_length(value: Any) -> int:
    """校验切片长度：必须为正整数。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(
            "slice_length 必须为正整数，收到 %r（%s）"
            % (value, type(value).__name__),
            {"slice_length": value},
        )
    if value <= 0:
        raise ConfigurationError(
            "slice_length 必须为正整数，收到 %d" % value,
            {"slice_length": value},
        )
    return int(value)


@dataclass
class PhaseShiftConfig:
    """一次相位偏移运行的全部配置。

    Args:
        slice_length: 切片长度，即相位偏移的目标周期。关键 token 的索引会被
            推进到满足 ``index % slice_length == 0`` 的新索引。默认 4；
            若目标模型的切片长度不同，请显式传入。
        key_mode: 关键 token 判定模式，``auto`` / ``keywords`` / ``weight``。
        keywords: 关键词列表（``key_mode="keywords"`` 时必填）。
        weight_threshold: 语义权重阈值（``key_mode="weight"`` 时使用）。
        window_size: 宏观窗口长度（token 数），``0`` 表示关闭窗口化判定，
            逐 token 独立判定（默认）。建议取目标模型步长的整数倍。
        window_agg: 窗口内聚合方式，``sum``（权重总和）或 ``count``
            （不低于 ``weight_threshold`` 的高分词个数）。
        insert_mode: 插入位置策略，``inline``（紧贴目标 token，默认）或
            ``segment``（段落头部）。
        filler_priority: 占位符优先级，按名称列出；``None`` 表示自动排序。
        max_retries: 单个 token 的最大候选尝试预算。
        max_chars: 输入字符数上限保护。
        restore_first: 是否在运行前剥离历史占位字符（幂等保证）。

    Raises:
        ConfigurationError: 任一字段取值不合法。
    """

    slice_length: int = DEFAULT_SLICE_LENGTH
    key_mode: str = DEFAULT_KEY_MODE
    keywords: Optional[Sequence[str]] = None
    weight_threshold: float = DEFAULT_WEIGHT_THRESHOLD
    window_size: int = DEFAULT_WINDOW_SIZE
    window_agg: str = DEFAULT_WINDOW_AGG
    insert_mode: str = DEFAULT_INSERT_MODE
    filler_priority: Optional[Sequence[str]] = None
    max_retries: int = DEFAULT_MAX_RETRIES
    max_chars: int = DEFAULT_MAX_CHARS
    restore_first: bool = True

    def __post_init__(self) -> None:
        self.slice_length = _validate_slice_length(self.slice_length)

        if self.key_mode not in KEY_MODES:
            raise ConfigurationError(
                "未知 key_mode %r（可用：%s）" % (self.key_mode, ", ".join(KEY_MODES)),
                {"key_mode": self.key_mode, "available": list(KEY_MODES)},
            )
        self.keywords = [k for k in (self.keywords or []) if k] or None
        if self.key_mode == "keywords" and not self.keywords:
            raise ConfigurationError(
                "key_mode='keywords' 必须提供 keywords 列表",
                {"key_mode": self.key_mode},
            )

        if not 0.0 <= float(self.weight_threshold) <= 1.0:
            raise ConfigurationError(
                "weight_threshold 必须在 [0, 1] 内，收到 %r" % self.weight_threshold,
                {"weight_threshold": self.weight_threshold},
            )
        self.weight_threshold = float(self.weight_threshold)

        if isinstance(self.window_size, bool) or not isinstance(self.window_size, int):
            raise ConfigurationError(
                "window_size 必须为整数（0 表示关闭窗口化），收到 %r（%s）"
                % (self.window_size, type(self.window_size).__name__),
                {"window_size": self.window_size},
            )
        if self.window_size < 0:
            raise ConfigurationError(
                "window_size 必须为非负整数（0 表示关闭窗口化），收到 %d"
                % self.window_size,
                {"window_size": self.window_size},
            )
        self.window_size = int(self.window_size)

        if self.window_size and self.key_mode != KEY_MODE_WEIGHT:
            raise ConfigurationError(
                "window_size 仅在 key_mode='weight' 下生效，当前 key_mode=%s"
                % self.key_mode,
                {"key_mode": self.key_mode, "window_size": self.window_size},
            )

        if self.window_agg not in WINDOW_AGGS:
            raise ConfigurationError(
                "未知 window_agg %r（可用：%s）"
                % (self.window_agg, ", ".join(WINDOW_AGGS)),
                {"window_agg": self.window_agg, "available": list(WINDOW_AGGS)},
            )

        if self.filler_priority is not None:
            if isinstance(self.filler_priority, str) or not isinstance(
                self.filler_priority, (list, tuple)
            ):
                raise ConfigurationError(
                    "filler_priority 必须是名称序列（list/tuple），收到 %s"
                    % type(self.filler_priority).__name__,
                    {"filler_priority": self.filler_priority},
                )
            if not all(isinstance(n, str) and n for n in self.filler_priority):
                raise ConfigurationError(
                    "filler_priority 必须全部为非空字符串",
                    {"filler_priority": list(self.filler_priority)},
                )
            self.filler_priority = [str(n) for n in self.filler_priority]

        if self.max_retries <= 0:
            raise ConfigurationError(
                "max_retries 必须为正整数，收到 %r" % self.max_retries,
                {"max_retries": self.max_retries},
            )
        if self.max_chars <= 0:
            raise ConfigurationError(
                "max_chars 必须为正整数，收到 %r" % self.max_chars,
                {"max_chars": self.max_chars},
            )

    # ------------------------------------------------------------ 兼容别名
    @property
    def period(self) -> int:
        """``slice_length`` 的兼容别名（只读）。"""
        return self.slice_length

    # ------------------------------------------------------------ 序列化
    def to_dict(self) -> Dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "slice_length": self.slice_length,
            "key_mode": self.key_mode,
            "keywords": list(self.keywords) if self.keywords else None,
            "weight_threshold": self.weight_threshold,
            "window_size": self.window_size,
            "window_agg": self.window_agg,
            "insert_mode": self.insert_mode,
            "filler_priority": list(self.filler_priority)
            if self.filler_priority
            else None,
            "max_retries": self.max_retries,
            "max_chars": self.max_chars,
            "restore_first": self.restore_first,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseShiftConfig":
        """从 :meth:`to_dict` 的输出还原配置对象。"""
        known = {
            "slice_length",
            "key_mode",
            "keywords",
            "weight_threshold",
            "window_size",
            "window_agg",
            "insert_mode",
            "filler_priority",
            "max_retries",
            "max_chars",
            "restore_first",
        }
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    def with_slice_length(self, slice_length: int) -> "PhaseShiftConfig":
        """返回一个仅替换 ``slice_length`` 的新配置对象。"""
        return PhaseShiftConfig.from_dict({**self.to_dict(), "slice_length": slice_length})

    def with_max_chars(self, max_chars: int) -> "PhaseShiftConfig":
        """返回一个仅替换 ``max_chars`` 的新配置对象。

        本方法只产出配置对象；已构造的 :class:`PhaseShifter` 仍持有原配置，
        需要让新上限生效时应重新构造门面。
        """
        return PhaseShiftConfig.from_dict({**self.to_dict(), "max_chars": max_chars})

    def with_keywords(self, keywords: Optional[Sequence[str]]) -> "PhaseShiftConfig":
        """返回一个仅替换 ``keywords`` 的新配置对象。

        用于"关键词在运行时才确定"的场景（例如调用方先从待发送文本里捕获出
        随机标识，再把捕获结果作为关键词）。``key_mode="keywords"`` 而
        ``keywords`` 为空时抛 :class:`ConfigurationError`，调用方应确保传入
        非空列表。
        """
        return PhaseShiftConfig.from_dict(
            {**self.to_dict(), "keywords": list(keywords) if keywords else None}
        )


def default_config(**overrides: Any) -> PhaseShiftConfig:
    """构造默认配置，可用关键字参数覆盖任意字段。"""
    return PhaseShiftConfig(**overrides)
