# -*- coding: utf-8 -*-
"""语义权重打分。

分两层，先微观、后宏观：

1. **单点权重**（微观）：:func:`score_token` / :func:`score_tokens`，
   公式（可插拔，仅用于演示"自定义阈值"能力）::

       w = 0.5 * len_norm + 0.3 * is_word + 0.2 * rarity

   * ``len_norm = min(len(token), 8) / 8``：越长的 token 通常携带越多语义；
   * ``is_word``：是否为"字母/CJK 连续块"（数字串给 0.5）；
   * ``rarity = 1 - min(id, 100000) / vocab_size``：token id 越大通常越罕见。

   权重范围 ``[0, 1]``；纯标点/空白的权重接近 0。

2. **窗口权重**（宏观）：:func:`window_scores` / :func:`peak_window`，
   在单点权重数组上滑动固定长度窗口，把"局部微观高分"汇总为"段落级宏观
   高分"，用于定位全文信息密度最高的核心段落。窗口大小建议取目标模型
   步长（stride）的整数倍，常见取值为 64 或 128 token。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

log = logging.getLogger("phase_shifting.weights")

#: 默认窗口聚合方式：``sum``（窗口内权重总和）或 ``count``（窗口内高分词个数）。
WINDOW_AGG_SUM = "sum"
WINDOW_AGG_COUNT = "count"
WINDOW_AGGS = (WINDOW_AGG_SUM, WINDOW_AGG_COUNT)

_LETTER_RUN = re.compile(r"[^\W\d_]+", re.UNICODE)
_DIGIT_RUN = re.compile(r"\d+")
_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x3040, 0x30FF),
    (0xAC00, 0xD7AF),
)


@dataclass(frozen=True)
class WeightResult:
    token_id: int
    token: str
    weight: float
    is_word: float
    len_norm: float
    rarity: float


def _clean(token: str) -> str:
    """去掉 ByteLevel 的 ``Ġ`` 前缀与首尾空白。"""
    return token.replace("Ġ", " ").replace("Ċ", "\n").strip()


def _is_cjk_block(s: str) -> bool:
    return any(lo <= ord(c) <= hi for c in s for lo, hi in _CJK_RANGES)


def is_word_like(s: str) -> bool:
    """是否像"词"（字母连续块 / CJK 块 / 数字串）。"""
    if not s:
        return 0.0
    if _CJK_RANGES and _is_cjk_block(s):
        return 1.0
    if _LETTER_RUN.fullmatch(s):
        return 1.0
    if _DIGIT_RUN.fullmatch(s):
        return 0.5
    return 0.0


def score_token(token: str, token_id: int, vocab_size: int) -> WeightResult:
    """给单个 token 打分。"""
    clean = _clean(token)
    if not clean:
        return WeightResult(token_id, token, 0.0, 0.0, 0.0, 0.0)

    len_norm = min(len(clean), 8) / 8.0
    word = is_word_like(clean)
    rarity = 1.0 - min(token_id, 100000) / float(max(vocab_size, 1))
    weight = 0.5 * len_norm + 0.3 * word + 0.2 * rarity
    weight = max(0.0, min(1.0, weight))
    return WeightResult(token_id, token, weight, word, len_norm, rarity)


def score_tokens(
    tokens: List[str], ids: List[int], vocab_size: int
) -> List[WeightResult]:
    """批量打分。"""
    return [
        score_token(tok, int(tid), vocab_size) for tok, tid in zip(tokens, ids)
    ]


def window_scores(
    weights: Sequence[float],
    window_size: int,
    agg: str = WINDOW_AGG_SUM,
    threshold: float = 0.5,
) -> List[float]:
    """在单点权重数组上滑动定长窗口，返回窗口分数序列 ``S``。

    Args:
        weights: 单点权重数组 ``W``，长度 N，元素通常来自 :func:`score_tokens`。
        window_size: 窗口长度（token 数），建议为模型步长的整数倍。
        agg: 窗口内聚合方式，``sum`` 为权重总和，``count`` 为不低于 ``threshold``
            的高分词个数。
        threshold: ``agg="count"`` 时判定"高分"所用的阈值。

    Returns:
        长度 ``max(0, N - window_size + 1)`` 的分数序列；``S[i]`` 是以 token
        ``i`` 为起点的窗口分数。``window_size <= 0`` 时返回空列表。

    Raises:
        ValueError: ``agg`` 不是受支持的聚合方式，或 ``window_size`` 为负。
    """
    if window_size < 0:
        raise ValueError("window_size 不能为负，收到 %r" % window_size)
    if agg not in WINDOW_AGGS:
        raise ValueError(
            "未知 window agg：%r（可用：%s）" % (agg, ", ".join(WINDOW_AGGS))
        )
    n = len(weights)
    if window_size == 0 or n < window_size:
        return []

    scores: List[float] = []
    if agg == WINDOW_AGG_SUM:
        running = sum(weights[:window_size])
        scores.append(running)
        for i in range(1, n - window_size + 1):
            running += weights[i + window_size - 1] - weights[i - 1]
            scores.append(running)
        return scores

    flags = [1 if w >= threshold else 0 for w in weights]
    running = sum(flags[:window_size])
    scores.append(running)
    for i in range(1, n - window_size + 1):
        running += flags[i + window_size - 1] - flags[i - 1]
        scores.append(running)
    return scores


@dataclass(frozen=True)
class PeakWindow:
    """全文信息密度最高的窗口（Peak Window）。

    Attributes:
        start: 峰窗首个 token 在全文中的下标。
        end: 峰窗末个 token 的下标（含）。
        score: 峰窗分数。
        size: 窗口长度（token 数）。
        agg: 产生该结果所用的聚合方式。
    """

    start: int
    end: int
    score: float
    size: int
    agg: str


def peak_window(
    weights: Sequence[float],
    window_size: int,
    agg: str = WINDOW_AGG_SUM,
    threshold: float = 0.5,
) -> Optional[PeakWindow]:
    """找出得分最高的窗口；多个同分窗口取最靠前的一个。

    Args:
        weights: 单点权重数组 ``W``。
        window_size: 窗口长度（token 数）。
        agg: 窗口内聚合方式，见 :func:`window_scores`。
        threshold: ``agg="count"`` 时的高分阈值。

    Returns:
        :class:`PeakWindow`；文本短于一个窗口、或窗口未启用时返回 ``None``。

    Raises:
        ValueError: 参数不合法，与 :func:`window_scores` 相同。
    """
    scores = window_scores(weights, window_size, agg=agg, threshold=threshold)
    if not scores:
        return None
    best = max(range(len(scores)), key=lambda i: (scores[i], -i))
    return PeakWindow(
        start=best,
        end=best + window_size - 1,
        score=scores[best],
        size=window_size,
        agg=agg,
    )
