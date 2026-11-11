# -*- coding: utf-8 -*-
"""包级常量。

切片长度（slice length）指模型一次前向处理的 token 数。相位偏移的目标是把
关键 token 的索引推进到 ``index % slice_length == 0``。不同模型的切片长度
可能不同，因此这里只提供**默认值**，实际取值必须由调用方通过公共 API 传入。
"""

from __future__ import annotations

#: 默认切片长度。
#:
#: 这是本包中唯一允许出现该数字字面量的位置；其余模块一律引用本常量或
#: 由调用方显式传入的 ``slice_length`` 参数。
DEFAULT_SLICE_LENGTH: int = 4

#: 默认关键 token 判定模式（``auto`` / ``keywords`` / ``weight``）。
DEFAULT_KEY_MODE: str = "auto"

#: 默认宏观窗口长度（token 数），0 表示关闭窗口化判定。
#:
#: 窗口长度建议取目标模型步长（stride）的整数倍，常见为 64 / 128 token。
DEFAULT_WINDOW_SIZE: int = 0

#: 默认窗口聚合方式（``sum`` = 窗口内权重总和，``count`` = 高分词个数）。
DEFAULT_WINDOW_AGG: str = "sum"

#: 默认插入位置策略（``inline`` = 紧贴目标 token，``segment`` = 段落头部）。
DEFAULT_INSERT_MODE: str = "inline"

#: 单个 token 的最大候选尝试预算。
DEFAULT_MAX_RETRIES: int = 6

#: 输入字符数上限保护。
DEFAULT_MAX_CHARS: int = 200_000

#: 默认语义权重阈值。
DEFAULT_WEIGHT_THRESHOLD: float = 0.5

__all__ = [
    "DEFAULT_SLICE_LENGTH",
    "DEFAULT_KEY_MODE",
    "DEFAULT_INSERT_MODE",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_WEIGHT_THRESHOLD",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_WINDOW_AGG",
]
