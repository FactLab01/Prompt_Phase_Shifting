# -*- coding: utf-8 -*-
"""phase_shifting — 前置相位偏移 Demo。

把用户输入中的关键 token 通过"动态插入无害 Token"物理推进到
DeepSeek 的注意力偏置位置（``token_index % period == 0``）。

设计文档见 ``docs/SPEC.md``；使用说明见 ``README.md``。
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "aligner",
    "cli",
    "diffview",
    "fillers",
    "keytokens",
    "logsetup",
    "report",
    "segments",
    "tokenizer_api",
    "validator",
    "weights",
]
