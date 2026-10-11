# -*- coding: utf-8 -*-
"""插入前后的可读差异视图。

把不可见字符渲染成 ``<ZWSP>`` 之类的记号后，用行级 unified diff 展示变化，
让人一眼看出"到底在哪一段前面插了几个不可见 token"。
"""

from __future__ import annotations

import difflib
from typing import List, Optional

from .tokenizer_api import visible


def _split_lines(text: str) -> List[str]:
    return text.splitlines(keepends=True)


def render_diff(before: str, after: str, context: int = 1) -> str:
    """返回 unified diff 字符串；无差异时返回空串。"""
    if before == after:
        return ""
    before_lines = _split_lines(visible(before)) or [""]
    after_lines = _split_lines(visible(after)) or [""]
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile="before (原文)",
        tofile="after (相位对齐后)",
        n=context,
    )
    return "".join(diff)


def render_side_by_side(before: str, after: str, width: int = 48) -> str:
    """并排视图（截断显示），用于快速人工核对。"""
    b = visible(before)
    a = visible(after)
    out = ["%-*s | %-*s" % (width, "BEFORE", width, "AFTER"), "-" * (width * 2 + 3)]
    b_lines = _split_lines(b) or [""]
    a_lines = _split_lines(a) or [""]
    for i in range(max(len(b_lines), len(a_lines))):
        bl = b_lines[i].rstrip("\n") if i < len(b_lines) else ""
        al = a_lines[i].rstrip("\n") if i < len(a_lines) else ""
        if len(bl) > width:
            bl = bl[: width - 1] + "…"
        if len(al) > width:
            al = al[: width - 1] + "…"
        out.append("%-*s | %-*s" % (width, bl, width, al))
    return "\n".join(out)


def render_inline_markers(text: str, markers: Optional[List[int]] = None) -> str:
    """把占位字符高亮为 ``⟦ZWSP⟧`` 形式，便于在报告里指出插入位置。"""
    if not text:
        return ""
    marks = set(markers or [])
    out = []
    for i, ch in enumerate(text):
        if ch == "​":
            tag = "ZWSP"
        elif ch == "﻿":
            tag = "BOM"
        elif ch == "⁠":
            tag = "WJ"
        elif ch == "\n":
            tag = "\\n"
        else:
            out.append(ch)
            continue
        out.append("⟦%s⟧" % tag if i in marks else visible(ch))
    return "".join(out)
