# -*- coding: utf-8 -*-
"""报告渲染：把 ``AlignOutcome`` 输出为人类可读文本 / JSON。

输出内容（对齐需求）：

1. 对齐前后的 token index 对比表；
2. 插入占位符的详细信息（字符、Unicode 码点、对应 token 数、位置、尝试次数）；
3. 整体对齐结论。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .aligner import (
    STATUS_ALIGNED,
    STATUS_FAILED,
    STATUS_NO_KEY,
    STATUS_PARTIAL,
    AlignOutcome,
)
from .tokenizer_api import TokenizerHandle, visible

STATUS_BADGE = {
    STATUS_ALIGNED: "✅ ALIGNED",
    STATUS_PARTIAL: "⚠️  PARTIAL",
    STATUS_FAILED: "❌ FAILED",
    STATUS_NO_KEY: "ℹ️  NO_KEY_TOKENS",
}


def _index_rows(
    outcome: AlignOutcome, tokenizer: TokenizerHandle, limit: int = 40
) -> List[Dict[str, Any]]:
    """构造 index 对比表：只列出关键 token。

    ``before`` 用原文 tokenize 的下标；``after`` 用"对齐后文本"重新 tokenize 后
    按原顺序做贪心匹配得到的下标（插入的占位 token 排在被插入 token 之前，
    因此匹配结果天然反映"物理推进"）。
    """
    report = outcome.key_report
    if report is None:
        return []

    after_ids = tokenizer.encode(outcome.aligned_text)
    inserted_after = _map_positions(report, after_ids)

    rows: List[Dict[str, Any]] = []
    for d in report.decisions:
        if not d.is_key:
            continue
        before = d.index
        after = inserted_after.get(before, before)
        rows.append(
            {
                "before": before,
                "after": after,
                "phase_before": before % outcome.period,
                "phase_after": after % outcome.period,
                "aligned": (after % outcome.period) == 0,
                "token": visible(d.text),
                "token_id": d.token_id,
                "reason": d.reason,
            }
        )
    return rows[:limit]


def _map_positions(report, after_ids: List[int]) -> Dict[int, int]:
    """把"对齐前的 token 下标"映射到"对齐后 token 流中的下标"。

    贪心顺序匹配：按原 token 顺序在 ``after_ids`` 中向前查找相同 id。
    """
    mapping: Dict[int, int] = {}
    cursor = 0
    for pos, d in enumerate(report.decisions):
        if cursor >= len(after_ids):
            break
        found = -1
        for j in range(cursor, len(after_ids)):
            if after_ids[j] == d.token_id:
                found = j
                break
        if found < 0:
            # 极端情况（BPE 边界变化导致 id 消失）：退化为等距位移
            mapping[d.index] = min(pos, len(after_ids) - 1)
            continue
        mapping[d.index] = found
        cursor = found + 1
    return mapping


def render_text(outcome: AlignOutcome, tokenizer: TokenizerHandle, verbose: bool = False) -> str:
    """渲染完整文本报告。"""
    lines: List[str] = []
    bar = "=" * 72

    lines.append(bar)
    lines.append("前置相位偏移报告  |  period=%d  |  虚拟环境: .venv" % outcome.period)
    lines.append(bar)

    lines.append("【输入规模】")
    lines.append("  原文长度        : %d 字符" % len(outcome.original_text))
    lines.append("  token 数 (对齐前): %d" % outcome.total_tokens_before)
    lines.append("  token 数 (对齐后): %d" % outcome.total_tokens_after)
    if outcome.key_report is not None:
        lines.append(
            "  关键 token 数    : %d / %d"
            % (len(outcome.key_report.key_indices), len(outcome.key_report.decisions))
        )
    lines.append("")

    lines.append("【Token Index 对比】")
    rows = _index_rows(outcome, tokenizer)
    if not rows:
        lines.append("  （无关键 token）")
    else:
        header = "  %-6s %-6s %-7s %-7s %-8s %s" % (
            "before",
            "after",
            "phase↓",
            "phase→",
            "aligned",
            "token",
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for r in rows:
            lines.append(
                "  %-6d %-6d %-7d %-7d %-8s %s"
                % (
                    r["before"],
                    r["after"],
                    r["phase_before"],
                    r["phase_after"],
                    "YES" if r["aligned"] else "no",
                    r["token"],
                )
            )
    lines.append("")

    lines.append("【插入的占位符详情】")
    if not outcome.insertions:
        lines.append("  （零干预：未插入任何占位符）")
    else:
        lines.append(
            "  %-4s %-8s %-10s %-12s %-7s %-7s %-6s %s"
            % ("#", "filler", "codepoint", "char", "chars", "tokens", "seg", "target token")
        )
        lines.append("  " + "-" * 74)
        for i, ins in enumerate(outcome.insertions, start=1):
            lines.append(
                "  %-4d %-8s %-10s %-12s %-7d %-7d %-6s %s"
                % (
                    i,
                    ins.filler_name,
                    ins.filler_codepoint,
                    visible(ins.filler_char),
                    ins.char_count,
                    ins.token_cost,
                    ins.segment_index if ins.segment_index >= 0 else "-",
                    "%s (idx %d→%d)"
                    % (visible(ins.token_text), ins.token_index_before, ins.token_index_after),
                )
            )
            if verbose:
                lines.append(
                    "        └─ 字符偏移 %d，第 %d 次尝试成功，%s"
                    % (ins.char_offset, ins.attempt, ins.note or "无备注")
                )
        if any(ins.structural for ins in outcome.insertions):
            lines.append("  ⚠️  部分占位符会改变段落结构（见 structural 标记）")
    lines.append("")

    if outcome.failures:
        lines.append("【未对齐的 token】")
        for f in outcome.failures:
            lines.append(
                "  token#%s %r 需要 +%s 位：%s"
                % (f.get("token_index"), f.get("token_text"), f.get("delta"), f.get("reason"))
            )
        lines.append("")

    lines.append("【结论】")
    lines.append("  %s" % STATUS_BADGE.get(outcome.status, outcome.status))
    lines.append("  %s" % outcome.summary())
    if outcome.message:
        lines.append("  %s" % outcome.message)
    lines.append(bar)
    return "\n".join(lines)


def render_json(outcome: AlignOutcome, tokenizer: TokenizerHandle) -> str:
    """渲染 JSON 报告（便于程序化消费）。"""
    payload: Dict[str, Any] = {
        "status": outcome.status,
        "period": outcome.period,
        "input_chars": len(outcome.original_text),
        "tokens_before": outcome.total_tokens_before,
        "tokens_after": outcome.total_tokens_after,
        "summary": outcome.summary(),
        "insertions": [ins.as_dict() for ins in outcome.insertions],
        "failures": outcome.failures,
        "aligned_text": outcome.aligned_text,
        "aligned_text_visible": visible(outcome.aligned_text),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_index_table_only(outcome: AlignOutcome, tokenizer: TokenizerHandle) -> str:
    """只渲染 index 对比表（``--table`` 用）。"""
    rows = _index_rows(outcome, tokenizer)
    if not rows:
        return "（无关键 token）"
    out = ["before  after  phase↓  phase→  aligned  token"]
    for r in rows:
        out.append(
            "%6d  %5d  %6d  %6d  %-7s  %s"
            % (
                r["before"],
                r["after"],
                r["phase_before"],
                r["phase_after"],
                "YES" if r["aligned"] else "no",
                r["token"],
            )
        )
    return "\n".join(out)
