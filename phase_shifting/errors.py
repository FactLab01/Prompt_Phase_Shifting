# -*- coding: utf-8 -*-
"""结构化异常体系。

设计要点：

* 所有异常都继承自 :class:`PhaseShiftingError`，便于调用方一次性捕获本包的
  全部错误；
* 每个异常**同时**继承对应的内置异常（``ValueError`` / ``KeyError`` /
  ``RuntimeError``），因此既有依赖内置异常的 ``except`` 分支继续生效，
  调用方无需改写错误处理代码；
* 每条消息都应包含"发生了什么 + 如何修正"，不暴露内部堆栈细节。
"""

from __future__ import annotations

from typing import Optional, Sequence

__all__ = [
    "PhaseShiftingError",
    "ConfigurationError",
    "UnknownFillerError",
    "InputError",
    "TokenizerError",
    "AlignmentError",
    "AnchorLostError",
    "RetryExhaustedError",
    "describe",
]


class PhaseShiftingError(Exception):
    """本包全部异常的基类。

    Args:
        message: 人类可读的错误描述，应说明成因与修正方式。
        details: 可选的结构化上下文（如参数名、实际值、候选集合）。
    """

    #: 稳定的错误类别标识，便于程序化分支处理。
    code: str = "phase_shifting_error"

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return self.message


# ---------------------------------------------------------------------- 配置
class ConfigurationError(PhaseShiftingError, ValueError):
    """调用方传入的参数不合法。"""

    code = "configuration_error"


class UnknownFillerError(ConfigurationError, KeyError):
    """引用了占位符注册表中不存在的候选名。"""

    code = "unknown_filler"

    def __init__(self, name: str, available: Sequence[str] = ()) -> None:
        msg = "未知占位符 %r" % name
        if available:
            msg += "（可用：%s）" % ", ".join(available)
        super().__init__(msg, {"name": name, "available": list(available)})
        # KeyError 的 str() 会对参数加引号，这里显式覆盖以保持信息完整
        self.name = name
        self.available = list(available)

    def __str__(self) -> str:
        return self.message


# ---------------------------------------------------------------------- 输入
class InputError(PhaseShiftingError, ValueError):
    """输入内容不合法（超长、无法解码等）。"""

    code = "input_error"


# ------------------------------------------------------------------ tokenizer
class TokenizerError(PhaseShiftingError, RuntimeError):
    """tokenizer 不可用或编码往返不一致。"""

    code = "tokenizer_error"


# -------------------------------------------------------------------- 对齐
class AlignmentError(PhaseShiftingError, RuntimeError):
    """对齐过程中发生不可恢复的错误。"""

    code = "alignment_error"


class AnchorLostError(AlignmentError):
    """插入后目标 token 的锚点在新的 token 流中丢失。"""

    code = "anchor_lost"

    def __init__(self, token_text: str, slice_length: int, reason: str = "") -> None:
        msg = "目标 token %r 在插入后的 token 流中定位失败" % token_text
        if reason:
            msg += "：%s" % reason
        super().__init__(
            msg,
            {
                "token_text": token_text,
                "slice_length": slice_length,
                "reason": reason,
            },
        )


class RetryExhaustedError(AlignmentError):
    """所有候选（占位种类 × 插入点 × 数量）均未通过校验。"""

    code = "retry_exhausted"

    def __init__(
        self,
        token_text: str,
        token_index: int,
        attempts: int,
        last_reason: str = "",
    ) -> None:
        msg = "token %r（index=%d）在 %d 次尝试后仍未对齐" % (
            token_text,
            token_index,
            attempts,
        )
        if last_reason:
            msg += "；最后一次失败原因：%s" % last_reason
        super().__init__(
            msg,
            {
                "token_text": token_text,
                "token_index": token_index,
                "attempts": attempts,
                "last_reason": last_reason,
            },
        )


def describe(exc: BaseException) -> dict:
    """把异常转成可序列化的摘要（用于报告与日志）。"""
    if isinstance(exc, PhaseShiftingError):
        payload: dict = {"code": exc.code, "message": exc.message}
        if exc.details:
            payload["details"] = exc.details
        return payload
    return {"code": type(exc).__name__, "message": str(exc)}
