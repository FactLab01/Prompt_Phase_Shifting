# -*- coding: utf-8 -*-
"""关键 token 判定。

三种模式：

* ``auto``（默认）：token 去掉空白与纯标点后仍非空 → 关键 token；
* ``keywords``：token 文本包含任一关键词（大小写不敏感）→ 关键 token；
* ``weight``：token 的语义权重 ≥ 阈值 → 关键 token。
"""

from __future__ import annotations

import logging
import string
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .weights import WeightResult, score_tokens

log = logging.getLogger("phase_shifting.keytokens")

KEY_MODE_AUTO = "auto"
KEY_MODE_KEYWORDS = "keywords"
KEY_MODE_WEIGHT = "weight"
KEY_MODES = (KEY_MODE_AUTO, KEY_MODE_KEYWORDS, KEY_MODE_WEIGHT)


def clean_token(token: str, tokenizer=None) -> str:
    """ByteLevel token → 去 ``Ġ``/首尾空白的可读文本。

    deepseek-tokenizer 的 token 字符串是 GPT-2 byte-level 表示（例如 ZWSP 的
    token 串是 ``'芒蘑沫'``，其实是 UTF-8 字节 E2 80 8B 经映射后的字符）。
    仅替换 ``Ġ``/``Ċ`` 无法还原这类字符，因此提供了 ``tokenizer`` 参数：
    传入 tokenizer 时优先用 ``decode`` 精确还原。
    """
    if tokenizer is not None:
        try:
            return tokenizer.decode_clean([token]) if False else _decode_token(token, tokenizer)
        except Exception:
            pass
    return token.replace("Ġ", " ").replace("Ċ", "\n").strip()


def _decode_token(token: str, tokenizer) -> str:
    """用 tokenizer 把单个 ByteLevel token 还原为原始字符。"""
    tid = tokenizer.token_id(token)
    if tid is None:
        return token.replace("Ġ", " ").replace("Ċ", "\n").strip()
    return tokenizer.decode([tid]).strip()


def is_pure_punct_or_space(text: str) -> bool:
    """是否"空白或纯标点/符号"（不承载语义）。"""
    if not text:
        return True
    for ch in text:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S") or cat.startswith("Z"):
            continue
        if ch in string.punctuation:
            continue
        return False
    return True


def _matches_keyword(token: str, keywords: Sequence[str]) -> Optional[str]:
    low = token.lower()
    for kw in keywords:
        if kw and kw.lower() in low:
            return kw
    return None


@dataclass
class KeyTokenDecision:
    """一个 token 的关键性判定结果。"""

    index: int
    token_id: int
    token: str
    text: str
    is_key: bool
    reason: str
    weight: Optional[float] = None
    matched_keyword: Optional[str] = None

    def phase(self, period: int) -> int:
        return self.index % period if period > 0 else 0

    def needs_align(self, period: int) -> bool:
        return self.is_key and period > 0 and (self.index % period) != 0

    def delta_needed(self, period: int) -> int:
        """需要向前推进多少位才能落到目标相位。"""
        if period <= 0 or not self.is_key:
            return 0
        return (period - (self.index % period)) % period


@dataclass
class KeyTokenReport:
    decisions: List[KeyTokenDecision] = field(default_factory=list)
    weights: List[Optional[WeightResult]] = field(default_factory=list)

    @property
    def key_indices(self) -> List[int]:
        return [d.index for d in self.decisions if d.is_key]

    @property
    def misaligned_indices(self) -> List[int]:
        return [d.index for d in self.decisions if d.is_key]

    def by_index(self, index: int) -> Optional[KeyTokenDecision]:
        for d in self.decisions:
            if d.index == index:
                return d
        return None


class KeyTokenSelector:
    """关键 token 选择器。"""

    def __init__(
        self,
        mode: str = KEY_MODE_AUTO,
        keywords: Optional[Sequence[str]] = None,
        weight_threshold: float = 0.5,
        tokenizer=None,
        filler_chars: Optional[Sequence[str]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        #: 占位符字符集合：这些字符对应的 token 永不被判为关键 token
        self.filler_chars = set(filler_chars or ())
        if mode not in KEY_MODES:
            raise ValueError("未知 key-mode：%s（可用：%s）" % (mode, ", ".join(KEY_MODES)))
        self.mode = mode
        self.keywords = [k for k in (keywords or []) if k]
        self.weight_threshold = float(weight_threshold)
        if mode == KEY_MODE_KEYWORDS and not self.keywords:
            raise ValueError("keywords 模式必须提供 --keywords")
        if not 0.0 <= self.weight_threshold <= 1.0:
            raise ValueError("weight-threshold 必须在 [0, 1] 内")

    # ------------------------------------------------------------------ 判定
    def select(
        self, tokens: Sequence[str], ids: Sequence[int], vocab_size: int
    ) -> KeyTokenReport:
        decisions: List[KeyTokenDecision] = []
        weights: List[Optional[WeightResult]] = []

        if self.mode == KEY_MODE_WEIGHT:
            weights = score_tokens(list(tokens), list(ids), vocab_size)

        for idx, (tok, tid) in enumerate(zip(tokens, ids)):
            text = clean_token(tok, self.tokenizer)
            is_filler = bool(text) and all(c in self.filler_chars for c in text)
            if self.mode == KEY_MODE_AUTO:
                if is_filler:
                    is_key = False
                    reason = "占位符 token（ filler ）"
                else:
                    is_key = not is_pure_punct_or_space(text)
                    reason = "非空白且非纯标点" if is_key else "空白或纯标点"
                decisions.append(
                    KeyTokenDecision(idx, int(tid), tok, text, is_key, reason)
                )
            elif self.mode == KEY_MODE_KEYWORDS:
                hit = _matches_keyword(text, self.keywords)
                is_key = hit is not None and not is_filler
                reason = "命中关键词 %r" % hit if hit else "未命中任何关键词"
                decisions.append(
                    KeyTokenDecision(
                        idx, int(tid), tok, text, is_key, reason, matched_keyword=hit
                    )
                )
            else:  # weight
                w = 0.0 if is_filler else weights[idx].weight
                is_key = w >= self.weight_threshold
                reason = "权重 %.3f %s 阈值 %.3f" % (
                    w,
                    "≥" if is_key else "<",
                    self.weight_threshold,
                )
                decisions.append(
                    KeyTokenDecision(idx, int(tid), tok, text, is_key, reason, weight=w)
                )

        return KeyTokenReport(decisions, weights)

    def describe(self) -> str:
        if self.mode == KEY_MODE_AUTO:
            return "auto（非空白且非纯标点）"
        if self.mode == KEY_MODE_KEYWORDS:
            return "keywords=%s" % ", ".join(self.keywords)
        return "weight≥%.3f" % self.weight_threshold
