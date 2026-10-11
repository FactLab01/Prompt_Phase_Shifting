# -*- coding: utf-8 -*-
"""语义权重打分。

权重公式（可插拔，仅用于演示"自定义阈值"能力）::

    w = 0.5 * len_norm + 0.3 * is_word + 0.2 * rarity

* ``len_norm = min(len(token), 8) / 8``：越长的 token 通常携带越多语义；
* ``is_word``：是否为"字母/CJK 连续块"（数字串给 0.5）；
* ``rarity = 1 - min(id, 100000) / vocab_size``：token id 越大通常越罕见。

权重范围 ``[0, 1]``；纯标点/空白的权重接近 0。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

log = logging.getLogger("phase_shifting.weights")

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
