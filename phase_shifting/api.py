# -*- coding: utf-8 -*-
"""SDK 公共门面。

本模块是外部调用方（Python / 未来 C-ABI 绑定）应当**唯一**依赖的入口：

* :class:`PhaseShifter` —— 可复用的门面对象，持有配置与依赖，适合批处理；
* :func:`align_text` —— 函数式单次入口，适合一次性调用；
* :class:`AlignmentResult` —— 结构化结果，可序列化、可直接断言。

门面只做"参数归一化 → 组装内部组件 → 调用引擎 → 适配结果"，不含对齐算法；
算法实现位于 :mod:`phase_shifting.aligner`。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from .aligner import (
    STATUS_ALIGNED,
    STATUS_FAILED,
    STATUS_NO_KEY,
    STATUS_PARTIAL,
    AlignOutcome,
    Aligner,
    Insertion,
)
from .config import PhaseShiftConfig
from .constants import DEFAULT_SLICE_LENGTH
from .errors import AlignmentError, ConfigurationError, InputError
from .fillers import FillerRegistry
from .keytokens import KeyTokenSelector
from .tokenizer_api import TokenizerHandle, get_tokenizer, visible

__all__ = [
    "PhaseShifter",
    "AlignmentResult",
    "AlignmentPlan",
    "align_text",
    "plan_text",
    "restore_text",
]


# ---------------------------------------------------------------------- 结果
@dataclass
class AlignmentResult:
    """一次相位对齐的结构化结果。

    Attributes:
        status: ``ALIGNED`` / ``PARTIAL`` / ``FAILED`` / ``NO_KEY_TOKENS``。
        aligned_text: 对齐后的文本（可能插入零宽占位字符）。
        original_text: 原始输入文本。
        slice_length: 本次运行使用的切片长度。
        tokens_before: 对齐前的 token 数。
        tokens_after: 对齐后的 token 数。
        insertions: 插入明细列表。
        failures: 未对齐 token 的失败明细列表。
        key_tokens: 被判定为关键的 token 列表（含原下标与判定理由）。
        message: 补充说明（失败原因或零干预说明）。
    """

    status: str
    aligned_text: str
    original_text: str
    slice_length: int
    tokens_before: int
    tokens_after: int
    insertions: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    key_tokens: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""

    # ---------------------------------------------------------------- 派生
    @property
    def success(self) -> bool:
        """是否全部对齐。"""
        return self.status == STATUS_ALIGNED

    @property
    def aligned_count(self) -> int:
        """成功推进的 token 数。"""
        return len(self.insertions)

    @property
    def token_delta(self) -> int:
        """插入导致的 token 数增量。"""
        return self.tokens_after - self.tokens_before

    def summary(self) -> str:
        """人类可读的一句话结论。"""
        if self.status == STATUS_ALIGNED:
            if not self.insertions:
                return "所有关键 token 已天然对齐，无需插入（零干预）"
            return "全部 %d 个未对齐关键 token 均已推进到相位 0" % self.aligned_count
        if self.status == STATUS_NO_KEY:
            return "未识别到任何关键 token，未做插入"
        if self.status == STATUS_PARTIAL:
            return "部分对齐：成功 %d 个，失败 %d 个" % (
                self.aligned_count,
                len(self.failures),
            )
        return "对齐失败：%s" % self.message

    # ---------------------------------------------------------------- 序列化
    def to_dict(self) -> Dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "status": self.status,
            "slice_length": self.slice_length,
            "input_chars": len(self.original_text),
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "token_delta": self.token_delta,
            "summary": self.summary(),
            "insertions": self.insertions,
            "failures": self.failures,
            "key_tokens": self.key_tokens,
            "original_text": self.original_text,
            "aligned_text": self.aligned_text,
            "aligned_text_visible": visible(self.aligned_text),
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_outcome(cls, outcome: AlignOutcome) -> "AlignmentResult":
        """由内部 :class:`AlignOutcome` 适配而来。"""
        key_tokens: List[Dict[str, Any]] = []
        report = outcome.key_report
        if report is not None:
            for d in report.decisions:
                if not d.is_key:
                    continue
                key_tokens.append(
                    {
                        "index": d.index,
                        "token": visible(d.text),
                        "token_id": d.token_id,
                        "reason": d.reason,
                    }
                )
        return cls(
            status=outcome.status,
            aligned_text=outcome.aligned_text,
            original_text=outcome.original_text,
            slice_length=outcome.period,
            tokens_before=outcome.total_tokens_before,
            tokens_after=outcome.total_tokens_after,
            insertions=[i.as_dict() for i in outcome.insertions],
            failures=list(outcome.failures),
            key_tokens=key_tokens,
            message=outcome.message,
        )


@dataclass
class AlignmentPlan:
    """只读分析结果：哪些关键 token 需要推进、推进多少。

    ``plan`` 不修改文本，适合"先分析后决策"的场景（例如调用方想自行决定
    插入策略时）。
    """

    slice_length: int
    total_tokens: int
    key_count: int
    misaligned: List[Dict[str, Any]] = field(default_factory=list)
    aligned: List[Dict[str, Any]] = field(default_factory=list)
    #: 窗口化判定时的峰值窗口信息；未启用窗口化或未定位到峰窗时为 ``None``
    peak_window: Optional[Dict[str, Any]] = None

    @property
    def needs_alignment(self) -> bool:
        """是否存在未对齐的关键 token。"""
        return bool(self.misaligned)

    def to_dict(self) -> Dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "slice_length": self.slice_length,
            "total_tokens": self.total_tokens,
            "key_count": self.key_count,
            "needs_alignment": self.needs_alignment,
            "misaligned": self.misaligned,
            "aligned": self.aligned,
            "peak_window": self.peak_window,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------- 门面
class PhaseShifter:
    """相位偏移 SDK 门面。

    推荐用法::

        from phase_shifting import PhaseShifter, PhaseShiftConfig

        shifter = PhaseShifter(PhaseShiftConfig(slice_length=4))
        result = shifter.align("DeepSeek 的注意力偏置位置")

        assert result.success
        print(result.summary(), result.token_delta)

    Args:
        config: 配置对象；``None`` 表示使用默认配置。
        slice_length: 切片长度便捷参数。若同时传入 ``config``，以本参数为准。
        tokenizer: 自定义 tokenizer 封装（测试替身用）；``None`` 表示使用
            包内默认 tokenizer。

    Raises:
        ConfigurationError: 配置参数不合法。
        TokenizerError: tokenizer 不可用。
    """

    def __init__(
        self,
        config: Optional[PhaseShiftConfig] = None,
        *,
        slice_length: Optional[int] = None,
        tokenizer: Optional[TokenizerHandle] = None,
    ) -> None:
        base = config or PhaseShiftConfig()
        if slice_length is not None:
            base = base.with_slice_length(slice_length)
        self.config: PhaseShiftConfig = base

        self._tokenizer: TokenizerHandle = tokenizer or get_tokenizer()
        self._registry: FillerRegistry = FillerRegistry(self._tokenizer)
        self._filler_chars: List[str] = [
            self._registry.get(n).char for n in self._registry.names()
        ]

    # ---------------------------------------------------------------- 属性
    @property
    def slice_length(self) -> int:
        """当前生效的切片长度。"""
        return self.config.slice_length

    @property
    def tokenizer(self) -> TokenizerHandle:
        """内部使用的 tokenizer 封装。"""
        return self._tokenizer

    # ---------------------------------------------------------------- 内部
    def _build_selector(self) -> KeyTokenSelector:
        """按配置构造关键 token 选择器。"""
        return KeyTokenSelector(
            mode=self.config.key_mode,
            keywords=self.config.keywords,
            weight_threshold=self.config.weight_threshold,
            window_size=self.config.window_size,
            window_agg=self.config.window_agg,
            tokenizer=self._tokenizer,
            filler_chars=self._filler_chars,
        )

    def _build_aligner(self) -> Aligner:
        """按配置组装内部对齐引擎。"""
        from .aligner import AlignConfig, INSERT_MODES

        if self.config.insert_mode not in INSERT_MODES:
            raise ConfigurationError(
                "未知 insert_mode %r（可用：%s）"
                % (self.config.insert_mode, ", ".join(INSERT_MODES)),
                {"insert_mode": self.config.insert_mode, "available": list(INSERT_MODES)},
            )
        engine_config = AlignConfig(
            period=self.config.slice_length,
            max_retries=self.config.max_retries,
            insert_mode=self.config.insert_mode,
            filler_priority=self.config.filler_priority,
            max_chars=self.config.max_chars,
            restore_first=self.config.restore_first,
        )
        return Aligner(self._tokenizer, self._build_selector(), self._registry, engine_config)

    def _check_input(self, text: str) -> str:
        """归一化并校验输入文本。"""
        if text is None:
            return ""
        if not isinstance(text, str):
            raise InputError(
                "text 必须是 str，收到 %s" % type(text).__name__,
                {"type": type(text).__name__},
            )
        if len(text) > self.config.max_chars:
            raise InputError(
                "输入过长：%d 字符 > 上限 %d（可调大 max_chars）"
                % (len(text), self.config.max_chars),
                {"length": len(text), "max_chars": self.config.max_chars},
            )
        return text

    # ---------------------------------------------------------------- 公共
    def align(
        self,
        text: str,
        progress: Optional[Any] = None,
    ) -> AlignmentResult:
        """执行相位对齐。

        Args:
            text: 输入文本。
            progress: 可选的进度回调，签名为 ``(message: str) -> None``；
                引擎每完成一次插入会调用一次。

        Returns:
            :class:`AlignmentResult`。

        Raises:
            InputError: 输入不是字符串或超过 ``max_chars`` 上限。
            ConfigurationError: 配置不合法。
            AlignmentError: 对齐过程中发生不可恢复的错误。
        """
        text = self._check_input(text)
        outcome = self._build_aligner().align(text, progress=progress)
        return AlignmentResult.from_outcome(outcome)

    def plan(self, text: str) -> AlignmentPlan:
        """只分析不修改：列出需要推进的关键 token。

        Args:
            text: 输入文本。

        Returns:
            :class:`AlignmentPlan`。窗口化判定（``key_mode="weight"`` 且
            ``window_size > 0``）时，``peak_window`` 字段带有所定位的峰值窗口
            区间与得分。

        Raises:
            InputError: 输入不是字符串或超过 ``max_chars`` 上限。
        """
        text = self._check_input(text)
        encoded = self._tokenizer.encode_with_tokens(text)
        selector = self._build_selector()
        report = selector.select(encoded.tokens, encoded.ids, self._tokenizer.vocab_size)

        misaligned: List[Dict[str, Any]] = []
        aligned: List[Dict[str, Any]] = []
        for d in report.decisions:
            if not d.is_key:
                continue
            entry = {
                "index": d.index,
                "token": visible(d.text),
                "token_id": d.token_id,
                "phase": d.phase(self.config.slice_length),
                "delta_needed": d.delta_needed(self.config.slice_length),
                "reason": d.reason,
            }
            if d.needs_align(self.config.slice_length):
                misaligned.append(entry)
            else:
                aligned.append(entry)

        peak_info: Optional[Dict[str, Any]] = None
        peak = getattr(report, "peak", None)
        if peak is not None:
            peak_info = {
                "start": peak.start,
                "end": peak.end,
                "score": peak.score,
                "size": peak.size,
                "agg": peak.agg,
            }

        return AlignmentPlan(
            slice_length=self.config.slice_length,
            total_tokens=len(encoded),
            key_count=len(report.key_indices),
            misaligned=misaligned,
            aligned=aligned,
            peak_window=peak_info,
        )

    def restore(self, text: str) -> str:
        """剥离文本中的占位字符。

        仅处理不可见且无结构干扰的候选；原文中合法的换行等结构字符原样保留。

        Args:
            text: 可能含占位字符的文本。

        Returns:
            剥离占位字符后的文本。
        """
        text = self._check_input(text)
        aligner = self._build_aligner()
        return aligner.restore(text)[0]

    def align_many(
        self, texts: Sequence[str], progress: Optional[Any] = None
    ) -> List[AlignmentResult]:
        """批量对齐多个文本（复用同一门面实例与 tokenizer）。

        Args:
            texts: 文本序列。
            progress: 可选的进度回调，签名为 ``(message: str) -> None``。

        Returns:
            与 ``texts`` 等长的 :class:`AlignmentResult` 列表。
        """
        return [self.align(t, progress=progress) for t in texts]


# ------------------------------------------------------------- 函数式入口
def align_text(
    text: str,
    *,
    config: Optional[PhaseShiftConfig] = None,
    slice_length: Optional[int] = None,
    tokenizer: Optional[TokenizerHandle] = None,
    progress: Optional[Any] = None,
) -> AlignmentResult:
    """一次性执行相位对齐。

    Args:
        text: 输入文本。
        config: 配置对象；``None`` 表示默认配置。
        slice_length: 切片长度便捷参数；若同时传入 ``config``，以本参数为准。
        tokenizer: 自定义 tokenizer 封装。
        progress: 可选的进度回调。

    Returns:
        :class:`AlignmentResult`。

    Examples:
        >>> from phase_shifting import align_text
        >>> result = align_text("alpha beta gamma", slice_length=4)
        >>> result.success
        True
    """
    shifter = PhaseShifter(config, slice_length=slice_length, tokenizer=tokenizer)
    return shifter.align(text, progress=progress)


def plan_text(
    text: str,
    *,
    config: Optional[PhaseShiftConfig] = None,
    slice_length: Optional[int] = None,
    tokenizer: Optional[TokenizerHandle] = None,
) -> AlignmentPlan:
    """一次性执行只读分析。

    Args:
        text: 输入文本。
        config: 配置对象；``None`` 表示默认配置。
        slice_length: 切片长度便捷参数。
        tokenizer: 自定义 tokenizer 封装。

    Returns:
        :class:`AlignmentPlan`。
    """
    shifter = PhaseShifter(config, slice_length=slice_length, tokenizer=tokenizer)
    return shifter.plan(text)


def restore_text(
    text: str,
    *,
    tokenizer: Optional[TokenizerHandle] = None,
) -> str:
    """一次性剥离文本中的占位字符。

    Args:
        text: 可能含占位字符的文本。
        tokenizer: 自定义 tokenizer 封装。

    Returns:
        剥离占位字符后的文本。
    """
    return PhaseShifter(tokenizer=tokenizer).restore(text)
