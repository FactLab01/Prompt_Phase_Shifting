# -*- coding: utf-8 -*-
"""关键 token 判定。

三种模式：

* ``auto``（默认）：token 去掉空白与纯标点后仍非空 → 关键 token；
* ``keywords``：token 文本包含任一关键词（大小写不敏感）→ 关键 token；
* ``weight``：token 的语义权重 ≥ 阈值 → 关键 token。可选叠加"宏观窗口"策略
  （``window_size > 0``）：先在全文上做单点打分得到权重数组 ``W``，再用定长
  窗口滑动求出窗口分数序列 ``S``，只把 **Peak Window（最高分窗口）** 内的
  token 判为关键 token，从而定位全文信息密度最高的核心段落。

判定结果由 :class:`~phase_shifting.aligner.Aligner` 消费并换算为相位推进量；
窗口化判定的设计取舍见 ``DEVELOPMENT.md`` §3.3。
"""

from __future__ import annotations

import logging
import string
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .errors import ConfigurationError
from .weights import (
    WINDOW_AGGS,
    WeightResult,
    peak_window,
    score_tokens,
    window_scores,
)

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
            return _decode_token(token, tokenizer)
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
    #: 窗口化判定时的峰值窗口（未启用窗口化时为 ``None``）
    peak: Optional[object] = None
    #: 窗口化判定时的窗口分数序列（未启用窗口化时为 ``None``）
    window_scores: Optional[List[float]] = None

    @property
    def key_indices(self) -> List[int]:
        """被判定为关键的 token 下标列表。"""
        return [d.index for d in self.decisions if d.is_key]

    def by_index(self, index: int) -> Optional[KeyTokenDecision]:
        """按下标取判定结果；不存在时返回 ``None``。"""
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
        window_size: int = 0,
        window_agg: str = "sum",
        tokenizer=None,
        filler_chars: Optional[Sequence[str]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        #: 占位符字符集合：这些字符对应的 token 永不被判为关键 token
        self.filler_chars = set(filler_chars or ())
        if mode not in KEY_MODES:
            raise ConfigurationError(
                "未知 key-mode：%s（可用：%s）" % (mode, ", ".join(KEY_MODES)),
                {"key_mode": mode, "available": list(KEY_MODES)},
            )
        self.mode = mode
        self.keywords = [k for k in (keywords or []) if k]
        self.weight_threshold = float(weight_threshold)
        #: 宏观窗口长度（token 数）；``0`` 表示关闭窗口化，逐 token 独立判定
        self.window_size = int(window_size)
        #: 窗口内聚合方式：``sum`` / ``count``
        self.window_agg = window_agg
        if mode == KEY_MODE_KEYWORDS and not self.keywords:
            raise ConfigurationError(
                "keywords 模式必须提供 keywords 列表",
                {"key_mode": mode},
            )
        if not 0.0 <= self.weight_threshold <= 1.0:
            raise ConfigurationError(
                "weight-threshold 必须在 [0, 1] 内，收到 %r" % self.weight_threshold,
                {"weight_threshold": self.weight_threshold},
            )
        if self.window_size < 0:
            raise ConfigurationError(
                "window-size 必须为非负整数（0 表示关闭），收到 %d" % self.window_size,
                {"window_size": self.window_size},
            )
        if self.window_agg not in WINDOW_AGGS:
            raise ConfigurationError(
                "未知 window-agg：%s（可用：%s）"
                % (self.window_agg, ", ".join(WINDOW_AGGS)),
                {"window_agg": self.window_agg, "available": list(WINDOW_AGGS)},
            )
        if self.window_size and mode != KEY_MODE_WEIGHT:
            raise ConfigurationError(
                "window-size 仅在 key-mode=weight 下生效，当前 mode=%s" % mode,
                {"key_mode": mode, "window_size": self.window_size},
            )
        #: 判定结论缓存（``mode, token_id, weight_threshold`` → 结论）。
        #: 关键性与位置无关故可复用；窗口化判定与位置相关，故该模式不使用本缓存。
        self._verdict_cache: Dict[tuple, tuple] = {}

    # ------------------------------------------------------------------ 判定
    def select(
        self, tokens: Sequence[str], ids: Sequence[int], vocab_size: int
    ) -> KeyTokenReport:
        """对一段 token 流做关键性判定。

        非窗口化模式下，判定只依赖 token 自身（文本、id、是否占位符、语义权重），
        与它在流中的位置无关，因此按 ``(mode, token_id, weight_threshold)`` 缓存
        判定结论：对齐主循环每轮都会对整篇文本重新判定，缓存可把重复代价从
        O(插入次数 × token 数) 降到 O(不同 token 数)。

        窗口化模式（``mode="weight"`` 且 ``window_size > 0``）下，判定依赖 token
        在全文中的位置，因此不走缓存：先对全流做单点打分得到权重数组，再滑动
        窗口求 Peak Window，只把峰窗内的 token 判为关键 token。
        """
        if self.mode == KEY_MODE_WEIGHT and self.window_size > 0:
            return self._select_windowed(tokens, ids, vocab_size)

        decisions: List[KeyTokenDecision] = []
        weights: List[Optional[WeightResult]] = []

        if self.mode == KEY_MODE_WEIGHT:
            weights = score_tokens(list(tokens), list(ids), vocab_size)

        for idx, (tok, tid) in enumerate(zip(tokens, ids)):
            cache_key = (self.mode, int(tid), self.weight_threshold)
            cached = self._verdict_cache.get(cache_key)
            if cached is None:
                cached = self._judge(tok, tid, idx, weights)
                self._verdict_cache[cache_key] = cached
            text, is_key, reason, weight, matched = cached
            decisions.append(
                KeyTokenDecision(
                    idx,
                    int(tid),
                    tok,
                    text,
                    is_key,
                    reason,
                    weight=weight,
                    matched_keyword=matched,
                )
            )

        return KeyTokenReport(decisions, weights)

    def _select_windowed(
        self, tokens: Sequence[str], ids: Sequence[int], vocab_size: int
    ) -> KeyTokenReport:
        """窗口化判定：定位 Peak Window，只保留峰窗内的关键 token。

        步骤：全流单点打分 → 权重数组 ``W`` → 定长窗口滑动求窗口分数序列 ``S``
        → 取最高分窗口（同分取最靠前者）→ 峰窗内权重 ≥ 阈值的 token 判为关键。

        全文短于一个窗口时没有窗口可选，此时"全流即峰窗"，等价于逐 token 判定。
        """
        weights = score_tokens(list(tokens), list(ids), vocab_size)
        w = [r.weight for r in weights]

        peak = peak_window(
            w,
            self.window_size,
            agg=self.window_agg,
            threshold=self.weight_threshold,
        )
        scores = window_scores(
            w,
            self.window_size,
            agg=self.window_agg,
            threshold=self.weight_threshold,
        )

        if peak is None:
            # 全文短于一个窗口：全流视为峰窗，峰窗信息留空
            in_peak = [True] * len(w)
            span = "未定位"
        else:
            in_peak = [peak.start <= i <= peak.end for i in range(len(w))]
            span = "#%d..#%d" % (peak.start, peak.end)

        decisions: List[KeyTokenDecision] = []
        for idx, (tok, tid) in enumerate(zip(tokens, ids)):
            text = clean_token(tok, self.tokenizer)
            is_filler = bool(text) and all(c in self.filler_chars for c in text)
            weight = 0.0 if is_filler else weights[idx].weight
            is_key = bool(in_peak[idx]) and weight >= self.weight_threshold
            if not in_peak[idx]:
                reason = "窗口外（峰窗 %s）" % span
            else:
                reason = "峰窗内，权重 %.3f %s 阈值 %.3f" % (
                    weight,
                    "≥" if is_key else "<",
                    self.weight_threshold,
                )
            decisions.append(
                KeyTokenDecision(
                    idx,
                    int(tid),
                    tok,
                    text,
                    is_key,
                    reason,
                    weight=weight,
                    matched_keyword=None,
                )
            )

        return KeyTokenReport(decisions, weights, peak=peak, window_scores=scores)

    def _judge(
        self,
        tok: str,
        tid: int,
        idx: int,
        weights: Sequence[Optional[WeightResult]],
    ):
        """对单个 token 做关键性判定，返回可缓存的结论元组。"""
        text = clean_token(tok, self.tokenizer)
        is_filler = bool(text) and all(c in self.filler_chars for c in text)

        if self.mode == KEY_MODE_AUTO:
            if is_filler:
                return text, False, "占位符 token（ filler ）", None, None
            is_key = not is_pure_punct_or_space(text)
            reason = "非空白且非纯标点" if is_key else "空白或纯标点"
            return text, is_key, reason, None, None

        if self.mode == KEY_MODE_KEYWORDS:
            hit = _matches_keyword(text, self.keywords)
            is_key = hit is not None and not is_filler
            reason = "命中关键词 %r" % hit if hit else "未命中任何关键词"
            return text, is_key, reason, None, hit

        w = 0.0 if is_filler else weights[idx].weight
        is_key = w >= self.weight_threshold
        reason = "权重 %.3f %s 阈值 %.3f" % (
            w,
            "≥" if is_key else "<",
            self.weight_threshold,
        )
        return text, is_key, reason, w, None

    def describe(self) -> str:
        if self.mode == KEY_MODE_AUTO:
            return "auto（非空白且非纯标点）"
        if self.mode == KEY_MODE_KEYWORDS:
            return "keywords=%s" % ", ".join(self.keywords)
        if self.window_size > 0:
            return "weight≥%.3f + 窗口%d(%s)" % (
                self.weight_threshold,
                self.window_size,
                self.window_agg,
            )
        return "weight≥%.3f" % self.weight_threshold
