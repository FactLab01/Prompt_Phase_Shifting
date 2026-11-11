# -*- coding: utf-8 -*-
"""C-ABI 原语层。

本模块是面向未来 C-ABI 绑定的**稳定原语面**，刻意保持最窄的签名约定：

* 仅使用位置参数（positional-only），无 ``*args`` / ``**kwargs``；
* 参数与返回值仅限可无损映射到 C 的基础类型：``str`` / ``int`` /
  ``Sequence[int]`` / ``Dict[str, Any]``；
* 无可变默认值、无回调、无异常穿越边界（错误以返回值与 ``ok`` 标志表达）；
* 函数名与语义保持稳定，供后续绑定生成器（ctypes / cffi / PyO3 等）读取。

上层（:mod:`phase_shifting.api`）负责把富对象结果适配到本层，本层不依赖
上层任何类型。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .constants import DEFAULT_SLICE_LENGTH
from .tokenizer_api import get_tokenizer

__all__ = [
    "PRIMITIVE_SIGNATURES",
    "encode",
    "decode",
    "token_strings",
    "token_count",
    "align_text",
    "align_text_detailed",
    "restore_text",
]


#: 原语签名清单：名称 → 人类可读签名。供文档与未来绑定生成器使用。
PRIMITIVE_SIGNATURES: Dict[str, str] = {
    "encode": "(text: str) -> List[int]",
    "decode": "(ids: Sequence[int]) -> str",
    "token_strings": "(ids: Sequence[int]) -> List[str]",
    "token_count": "(text: str) -> int",
    "align_text": "(text: str, slice_length: int) -> str",
    "align_text_detailed": "(text: str, slice_length: int) -> Dict[str, Any]",
    "restore_text": "(text: str) -> str",
}


def _normalize_ids(ids: Sequence[int]) -> List[int]:
    """把任意整型序列规范化为 ``list[int]``。"""
    if ids is None:
        return []
    return [int(i) for i in ids]


def encode(text: str) -> List[int]:
    """把文本编码为 token id 列表。

    Args:
        text: 待编码文本；空串返回空列表。

    Returns:
        token id 列表。
    """
    if not text:
        return []
    return get_tokenizer().encode(text)


def decode(ids: Sequence[int]) -> str:
    """把 token id 列表还原为文本。

    Args:
        ids: token id 序列。

    Returns:
        还原后的文本；空序列返回空串。
    """
    if not ids:
        return ""
    return get_tokenizer().decode(_normalize_ids(ids))


def token_strings(ids: Sequence[int]) -> List[str]:
    """把 token id 列表转为 ByteLevel token 字符串。

    Args:
        ids: token id 序列。

    Returns:
        与 ``ids`` 等长的 token 字符串列表。
    """
    if not ids:
        return []
    return get_tokenizer().convert_ids_to_tokens(_normalize_ids(ids))


def token_count(text: str) -> int:
    """返回文本的 token 数。

    Args:
        text: 待统计文本。

    Returns:
        token 数；空串返回 0。
    """
    if not text:
        return 0
    return get_tokenizer().count_tokens(text)


def align_text(text: str, slice_length: int = DEFAULT_SLICE_LENGTH) -> str:
    """执行相位对齐，仅返回对齐后的文本。

    Args:
        text: 输入文本。
        slice_length: 切片长度，关键 token 索引会被推进到
            ``index % slice_length == 0``。

    Returns:
        对齐后的文本（可能插入零宽占位字符）。
    """
    from .api import align_text as _align_text

    return _align_text(text, slice_length=slice_length).aligned_text


def align_text_detailed(
    text: str, slice_length: int = DEFAULT_SLICE_LENGTH
) -> Dict[str, Any]:
    """执行相位对齐，返回结构化摘要。

    Args:
        text: 输入文本。
        slice_length: 切片长度。

    Returns:
        包含 ``status`` / ``slice_length`` / ``original_text`` / ``aligned_text`` /
        ``tokens_before`` / ``tokens_after`` / ``insertions`` / ``failures`` /
        ``summary`` 的字典，可无损序列化为 JSON。
    """
    from .api import align_text as _align_text

    return _align_text(text, slice_length=slice_length).to_dict()


def restore_text(text: str) -> str:
    """剥离文本中的占位字符，还原为干净原文。

    Args:
        text: 可能含占位字符的文本。

    Returns:
        剥离占位字符后的文本。仅处理不可见且无结构干扰的候选，
        原文中合法的换行等结构字符原样保留。
    """
    from .api import PhaseShifter

    return PhaseShifter().restore(text)
