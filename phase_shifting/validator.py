# -*- coding: utf-8 -*-
"""动态校验器。

每次插入后立即重新 tokenize 完整文本，然后：

1. 用"关键 token 的原始 id 序列"作锚点，在**新的** token 流里做带偏移的滑窗匹配，
   重新定位该 token 的新下标；
2. 判断新下标是否满足 ``index % period == 0``；
3. 若锚点丢失（BPE 边界变化导致 token 被合并/拆分），判定为"定位失败"，
   交由对齐引擎切换占位符或调整数量重试。

滑窗匹配带一个"容忍带"：允许锚点在新流中的绝对位置与预期位置相差
``tolerance`` 个 token，用于吸收插入导致的下游 token 边界抖动。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .tokenizer_api import TokenizerHandle

log = logging.getLogger("phase_shifting.validator")


@dataclass
class LocateResult:
    found: bool
    index: Optional[int]
    expected_index: Optional[int] = None
    drift: int = 0
    detail: str = ""


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""
    locate: Optional[LocateResult] = None
    new_index: Optional[int] = None
    period: int = 4
    aligned: bool = False
    token_count_before: int = 0
    token_count_after: int = 0
    inserted_tokens: int = 0
    notes: List[str] = field(default_factory=list)


class DynamicValidator:
    """插入后的动态校验器。"""

    def __init__(self, tokenizer: TokenizerHandle, period: int = 4) -> None:
        if period <= 0:
            raise ValueError("period 必须为正整数")
        self.tokenizer = tokenizer
        self.period = period

    # ------------------------------------------------------------------ 定位
    def locate_anchor(
        self,
        anchor_ids: Sequence[int],
        ids_after: Sequence[int],
        hint: Optional[int] = None,
        tolerance: int = 8,
    ) -> LocateResult:
        """在新 token 流中定位锚点（一段连续 token id）。

        Args:
            anchor_ids: 锚点的 token id 序列（关键 token 本身，通常 1~2 个 id）。
            ids_after: 插入并重新 tokenize 后的完整 token id 流。
            hint: 预期下标（用于缩小搜索范围、计算漂移）。
            tolerance: 与预期下标允许的偏差。
        """
        anchor = [int(i) for i in anchor_ids]
        stream = [int(i) for i in ids_after]
        if not anchor:
            return LocateResult(False, None, hint, 0, "空锚点")
        if len(anchor) > len(stream):
            return LocateResult(False, None, hint, 0, "token 流比锚点还短")

        # 若给了 hint，优先在 [hint-tolerance, hint+tolerance] 窗口内找
        candidates: List[int] = []
        if hint is not None:
            lo = max(0, hint - tolerance)
            hi = min(len(stream) - len(anchor), hint + tolerance)
            for start in range(lo, hi + 1):
                if stream[start : start + len(anchor)] == anchor:
                    candidates.append(start)
        if not candidates:
            # 全流扫描兜底
            for start in range(0, len(stream) - len(anchor) + 1):
                if stream[start : start + len(anchor)] == anchor:
                    candidates.append(start)

        if not candidates:
            return LocateResult(False, None, hint, 0, "锚点在插入后的 token 流中丢失")

        # 取离 hint 最近的那个
        if hint is None:
            best = candidates[0]
        else:
            best = min(candidates, key=lambda c: abs(c - hint))
        return LocateResult(True, best, hint, best - (hint or 0), "ok")

    # ------------------------------------------------------------------ 校验
    def validate(
        self,
        text_before: str,
        text_after: str,
        anchor_ids: Sequence[int],
        expected_index: int,
        expected_inserted_tokens: int,
    ) -> ValidationResult:
        """对一次插入做完整校验。"""
        ids_before = self.tokenizer.encode(text_before)
        encoded_after = self.tokenizer.encode_with_tokens(text_after)
        ids_after = encoded_after.ids

        locate = self.locate_anchor(anchor_ids, ids_after, hint=expected_index)
        inserted = len(ids_after) - len(ids_before)

        result = ValidationResult(
            ok=False,
            period=self.period,
            token_count_before=len(ids_before),
            token_count_after=len(ids_after),
            inserted_tokens=inserted,
            locate=locate,
        )

        if not locate.found or locate.index is None:
            result.reason = "定位失败：%s" % locate.detail
            result.notes.append(
                "插入后 token 流长度 %d → %d（预期 +%d）"
                % (len(ids_before), len(ids_after), expected_inserted_tokens)
            )
            return result

        result.new_index = locate.index
        result.aligned = (locate.index % self.period) == 0
        if inserted != expected_inserted_tokens:
            result.notes.append(
                "token 增量与预期不符：预期 +%d，实际 +%d（可能发生合并/拆分）"
                % (expected_inserted_tokens, inserted)
            )
        if result.aligned:
            result.ok = True
            result.reason = "已对齐：新 index=%d，%d %% %d == 0" % (
                locate.index,
                locate.index,
                self.period,
            )
        else:
            result.reason = "未对齐：新 index=%d，%d %% %d == %d" % (
                locate.index,
                locate.index,
                self.period,
                locate.index % self.period,
            )
        return result

    # ------------------------------------------------------------------ 工具
    def is_aligned(self, index: int) -> bool:
        return (index % self.period) == 0

    def phase(self, index: int) -> int:
        return index % self.period
