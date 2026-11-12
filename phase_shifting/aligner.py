# -*- coding: utf-8 -*-
"""对齐引擎：最小干预地把关键 token 推进到目标相位。

算法流程见 ``docs/SPEC.md`` §7。核心是不变量：插入点只允许落在目标 token 正前方，
并以"前缀零破坏"作为硬性验收条件，使已对齐 token 的相位永不受扰动，迭代必然收敛。

插入数量取实测值而非预测值：BPE 合并/拆分导致的偏差由数量补偿候选自动吸收。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .constants import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_SLICE_LENGTH,
)
from .errors import InputError
from .fillers import FillerRegistry, FillerSpec
from .keytokens import KeyTokenReport, KeyTokenSelector
from .segments import (
    Segment,
    map_tokens_to_segments,
    split_segments,
    strip_fillers,
    token_start_offsets,
)
from .tokenizer_api import Encoded, TokenizerHandle, codepoint_str, visible
from .validator import DynamicValidator, ValidationResult

log = logging.getLogger("phase_shifting.aligner")

INSERT_MODE_SEGMENT = "segment"  # 段落头部
INSERT_MODE_INLINE = "inline"  # token 正前方（默认）
INSERT_MODES = (INSERT_MODE_SEGMENT, INSERT_MODE_INLINE)

#: 插入点相对 token 起始偏移的候选（0 = 紧贴 token）
_OFFSET_CANDIDATES = (0, -1, 1, -2, 2, -3, 3)
#: 占位字符数量的候选偏移（相对理论 delta）
_COUNT_CANDIDATES = (0, 1, 2, -1)
#: 字符数上限基数；实际上限为 max(_CHAR_CAP_BASE, delta + 2)，为大切片长度留补偿余地
_CHAR_CAP_BASE = 8
#: 主循环轮数上限 = 决策数 × _ROUND_FACTOR，下限 _ROUND_FLOOR（防御性护栏）
_ROUND_FACTOR = 4
_ROUND_FLOOR = 16
#: 插入后重新定位锚点的搜索容差基数
_LOCATE_TOLERANCE_BASE = 4
#: 单 token 候选尝试预算系数（相对 max_retries）
_ATTEMPT_BUDGET_FACTOR = 12

STATUS_ALIGNED = "ALIGNED"
STATUS_PARTIAL = "PARTIAL"
STATUS_FAILED = "FAILED"
STATUS_NO_KEY = "NO_KEY_TOKENS"


@dataclass
class Insertion:
    """一次成功的插入记录。"""

    token_index_before: int
    token_index_after: int
    token_id: int
    token_text: str
    filler_name: str
    filler_char: str
    filler_codepoint: str
    char_count: int
    token_cost: int
    segment_index: int
    char_offset: int
    attempt: int
    structural: bool = False
    note: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "token_index_before": self.token_index_before,
            "token_index_after": self.token_index_after,
            "token_id": self.token_id,
            "token_text": visible(self.token_text),
            "filler": self.filler_name,
            "filler_char": visible(self.filler_char),
            "filler_codepoint": self.filler_codepoint,
            "char_count": self.char_count,
            "token_cost": self.token_cost,
            "segment_index": self.segment_index,
            "char_offset": self.char_offset,
            "attempt": self.attempt,
            "structural_interference": self.structural,
            "note": self.note,
        }


@dataclass
class AlignOutcome:
    """一次对齐运行的完整结果。"""

    status: str
    original_text: str
    aligned_text: str
    period: int
    total_tokens_before: int
    total_tokens_after: int
    key_report: Optional[KeyTokenReport]
    insertions: List[Insertion] = field(default_factory=list)
    failures: List[Dict[str, object]] = field(default_factory=list)
    validations: List[ValidationResult] = field(default_factory=list)
    message: str = ""

    @property
    def aligned_count(self) -> int:
        return len(self.insertions)

    @property
    def success(self) -> bool:
        return self.status == STATUS_ALIGNED

    def summary(self) -> str:
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


def _occurrence_of(report: KeyTokenReport, decision) -> int:
    """该 decision 的 token id 在关键 token 序列中是第几次出现（0 起）。

    用于在"同一 token id 多次出现"的文本里精确定位。
    例如 "alpha beta alpha beta alpha" → alpha 三次分别记 0/1/2，
    beta 两次分别记 0/1。
    """
    count = 0
    for d in report.decisions:
        if d is decision or d.index == decision.index:
            return count
        if d.token_id == decision.token_id:
            count += 1
    return count


@dataclass
class AlignConfig:
    """内部引擎配置。

    ``period`` 为切片长度（phase period），即相位偏移的目标周期：关键 token
    的索引会被推进到满足 ``index % period == 0`` 的新索引。默认取
    :data:`phase_shifting.constants.DEFAULT_SLICE_LENGTH`；公共 API 层
    （:mod:`phase_shifting.api`）以 ``slice_length`` 为名暴露同一参数，
    本类另提供只读别名 :attr:`slice_length`。

    Attributes:
        period: 切片长度 / 相位周期，必须为正整数。
        max_retries: 单个 token 的最大候选尝试预算。
        insert_mode: 插入位置策略，``inline`` / ``segment``。
        filler_priority: 占位符优先级（按名称）。
        max_chars: 输入字符数上限保护。
        restore_first: 是否在运行前剥离历史占位字符（幂等保证）。
    """

    period: int = DEFAULT_SLICE_LENGTH
    max_retries: int = DEFAULT_MAX_RETRIES
    insert_mode: str = INSERT_MODE_INLINE
    filler_priority: Optional[Sequence[str]] = None
    max_chars: int = DEFAULT_MAX_CHARS
    restore_first: bool = True

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError("period（切片长度）必须为正整数，收到 %r" % self.period)
        if self.insert_mode not in INSERT_MODES:
            raise ValueError(
                "未知 insert-mode：%s（可用：%s）"
                % (self.insert_mode, ", ".join(INSERT_MODES))
            )
        if self.max_retries <= 0:
            raise ValueError("max_retries 必须为正整数，收到 %r" % self.max_retries)
        if self.max_chars <= 0:
            raise ValueError("max_chars 必须为正整数，收到 %r" % self.max_chars)

    @property
    def slice_length(self) -> int:
        """``period`` 的语义化别名：切片长度。"""
        return self.period


class Aligner:
    """对齐引擎。"""

    def __init__(
        self,
        tokenizer: TokenizerHandle,
        selector: KeyTokenSelector,
        registry: FillerRegistry,
        config: Optional[AlignConfig] = None,
        validator: Optional[DynamicValidator] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.selector = selector
        self.registry = registry
        # selector 未注入 tokenizer / filler_chars 时补齐，确保 filler token
        # 不会被误判为关键 token
        if selector.tokenizer is None:
            selector.tokenizer = tokenizer
        if not selector.filler_chars:
            selector.filler_chars = {registry.get(n).char for n in registry.names()}
        # 仅"不可见且无结构干扰"的候选可参与还原：lf 的换行与原文合法换行
        # 无法区分，纳入剥离会造成不可逆的段落破坏
        self._strippable = {
            registry.get(n).char
            for n in registry.names()
            if registry.get(n).invisible and not registry.get(n).structural
        }
        if not self._strippable:  # pragma: no cover - 防御
            self._strippable = {registry.primary().char}
        self.config = config or AlignConfig()
        self.validator = validator or DynamicValidator(tokenizer, self.config.period)

    # ------------------------------------------------------------------ 公共
    def align(self, text: str, progress: Optional[Callable[[str], None]] = None) -> AlignOutcome:
        """对 ``text`` 执行相位对齐，返回完整结果。"""
        emit = progress or (lambda _msg: None)

        if text is None:
            text = ""
        if len(text) > self.config.max_chars:
            raise InputError(
                "输入过长：%d 字符 > 上限 %d（可用 max_chars 调大）"
                % (len(text), self.config.max_chars),
                {"length": len(text), "max_chars": self.config.max_chars},
            )

        # 幂等策略：先对原始输入（含历史占位符）判定，已全部对齐即零干预返回；
        # 仍有未对齐项时剥离历史占位符后重算，保证同一份原文得到同一份结果。
        encoded_before: Encoded = self.tokenizer.encode_with_tokens(text)
        report = self.selector.select(
            encoded_before.tokens, encoded_before.ids, self.tokenizer.vocab_size
        )

        if not report.key_indices:
            log.info("未识别到关键 token（mode=%s）", self.selector.mode)
            # 空文本 / 纯空白没有任何关键 token，天然"已对齐"（零干预）
            if not encoded_before.ids or not text.strip():
                return AlignOutcome(
                    status=STATUS_ALIGNED,
                    original_text=text,
                    aligned_text=text,
                    period=self.config.period,
                    total_tokens_before=len(encoded_before),
                    total_tokens_after=len(encoded_before),
                    key_report=report,
                    message="输入为空或纯空白，无需对齐",
                )
            return AlignOutcome(
                status=STATUS_NO_KEY,
                original_text=text,
                aligned_text=text,
                period=self.config.period,
                total_tokens_before=len(encoded_before),
                total_tokens_after=len(encoded_before),
                key_report=report,
                message="选择器 %s 未命中任何 token" % self.selector.describe(),
            )

        # 待处理目标 token（按下标升序）
        misaligned = [
            d for d in report.decisions if d.needs_align(self.config.period)
        ]
        if not misaligned:
            log.info("全部 %d 个关键 token 已天然对齐", len(report.key_indices))
            return AlignOutcome(
                status=STATUS_ALIGNED,
                original_text=text,
                aligned_text=text,
                period=self.config.period,
                total_tokens_before=len(encoded_before),
                total_tokens_after=len(encoded_before),
                key_report=report,
            )

        # 仍有未对齐项：剥离历史占位符后从干净原文重算
        if self.config.restore_first:
            text, removed = self._restore(text)
            if removed:
                emit("检测到 %d 个历史占位字符，已还原以保证幂等" % removed)
                log.info("还原 %d 个历史占位字符", removed)
                encoded_before = self.tokenizer.encode_with_tokens(text)
                report = self.selector.select(
                    encoded_before.tokens, encoded_before.ids, self.tokenizer.vocab_size
                )

        working = text
        insertions: List[Insertion] = []
        failures: List[Dict[str, object]] = []
        validations: List[ValidationResult] = []

        # 每轮重新 tokenize，取当前第一个未对齐的关键 token；插入点紧贴该 token
        # 且满足前缀零破坏，故每轮至少消灭一个未对齐项，循环必然收敛。
        max_rounds = max(len(report.decisions) * _ROUND_FACTOR, _ROUND_FLOOR)
        seen_states: set = set()
        for round_no in range(max_rounds):
            enc_now = self.tokenizer.encode_with_tokens(working)
            rep_now = self.selector.select(
                enc_now.tokens, enc_now.ids, self.tokenizer.vocab_size
            )
            pending = [
                d for d in rep_now.decisions if d.needs_align(self.config.period)
            ]
            if not pending:
                log.info("第 %d 轮：全部关键 token 已对齐", round_no)
                break

            decision = pending[0]
            target_before = decision.index
            delta = decision.delta_needed(self.config.period)
            anchor = enc_now.ids[target_before : target_before + 1]

            # 段落映射是 O(文本长度) 的全量计算，仅非 inline 模式需要
            seg = None
            if self.config.insert_mode != INSERT_MODE_INLINE:
                segments = split_segments(working)
                token_segments = map_tokens_to_segments(working, enc_now.tokens, segments)
                seg = (
                    token_segments[target_before]
                    if target_before < len(token_segments)
                    else None
                )

            # 只需目标 token 的字符偏移，计算限制在前缀范围内
            offsets_now = token_start_offsets(
                working, enc_now.tokens, limit=target_before + 1
            )
            base_offset = (
                offsets_now[target_before]
                if offsets_now is not None and target_before < len(offsets_now)
                else None
            )

            state = (working, target_before)
            if state in seen_states:
                failures.append(
                    {
                        "token_index": target_before,
                        "token_text": visible(decision.text),
                        "delta": delta,
                        "reason": "检测到循环：同一状态重复出现，已中止",
                    }
                )
                emit("token#%d %r 出现循环，中止" % (target_before, decision.text))
                break
            seen_states.add(state)

            ok, insertion, validation, working = self._align_one(
                working=working,
                enc=enc_now,
                decision=decision,
                delta=delta,
                base_offset=base_offset,
                anchor=anchor,
                segment=seg,
                emit=emit,
            )
            validations.append(validation)
            if ok and insertion is not None:
                insertions.append(insertion)
                emit(
                    "token#%d %r 已对齐：index %d → %d（+%d token，%s x%d）"
                    % (
                        target_before,
                        decision.text,
                        target_before,
                        insertion.token_index_after,
                        insertion.token_cost,
                        insertion.filler_name,
                        insertion.char_count,
                    )
                )
            else:
                failures.append(
                    {
                        "token_index": target_before,
                        "token_text": visible(decision.text),
                        "delta": delta,
                        "reason": validation.reason,
                    }
                )
                emit("token#%d %r 对齐失败：%s" % (target_before, decision.text, validation.reason))
                break

        encoded_after = self.tokenizer.encode_with_tokens(working)
        if failures:
            status = STATUS_PARTIAL if insertions else STATUS_FAILED
            message = "%d 个 token 未对齐" % len(failures)
        else:
            status = STATUS_ALIGNED
            message = ""

        return AlignOutcome(
            status=status,
            original_text=text,
            aligned_text=working,
            period=self.config.period,
            total_tokens_before=len(encoded_before),
            total_tokens_after=len(encoded_after),
            key_report=report,
            insertions=insertions,
            failures=failures,
            validations=validations,
            message=message,
        )

    # ------------------------------------------------------------------ 内部
    def _align_one(
        self,
        working: str,
        enc: Encoded,
        decision,
        delta: int,
        base_offset: Optional[int],
        anchor: List[int],
        segment: Optional[Segment],
        emit: Callable[[str], None],
    ) -> Tuple[bool, Optional[Insertion], ValidationResult, str]:
        """对单个 token 做"校验式"插入 + 重试。

        不预测插入后目标 token 的确切新下标，而是把候选真正插进去、
        重新 tokenize，再用**双条件**验收：

        * 前缀零破坏：``ids2[:index] == ids[:index]``（前面已对齐的 token
          相位绝不受影响，这是循环收敛的根本保证）；
        * 目标归位：目标 token id 首次出现位置 ``j >= index`` 且
          ``j % period == 0``。

        候选维度：占位种类 × 插入点（base ± 0/1/2/3）× 字符数（delta + 0/1/2/-1）。
        字符数上限随 delta 放大（``max(_CHAR_CAP_BASE, delta + 2)``），保证大切片
        长度下仍有补偿余地。
        """
        chain = self._filler_chain()
        last_validation: Optional[ValidationResult] = None
        target = decision.index
        seg_index = segment.index if segment is not None else -1

        if base_offset is None:
            # byte-level 定位失败时按 token 文本查找
            base_offset = self._token_char_offset(working, decision)

        attempts = 0
        max_attempts = max(1, self.config.max_retries)
        budget = max_attempts * _ATTEMPT_BUDGET_FACTOR
        # 枚举顺序：插入点 → 字符数补偿 → 占位种类；每次验证都要整篇重新
        # tokenize，故把最可能成功的组合排在前面以减少无效全量编码。
        for off_delta in _OFFSET_CANDIDATES:
            offset = base_offset + off_delta
            if offset < 0 or offset > len(working):
                continue

            for adjust in _COUNT_CANDIDATES:
                for spec in chain[:max_attempts]:
                    char_count = self._chars_for(delta, spec)
                    if char_count <= 0:
                        continue

                    char_cap = max(_CHAR_CAP_BASE, char_count + 2)
                    real_chars = char_count + adjust
                    if real_chars <= 0 or real_chars > char_cap:
                        continue

                    attempts += 1
                    payload = self.registry.render(spec.name, real_chars)
                    candidate = working[:offset] + payload + working[offset:]

                    ok, new_index, validation = self._verify_candidate(
                        working=working,
                        candidate=candidate,
                        enc=enc,
                        target=target,
                        anchor=anchor,
                        payload_chars=real_chars,
                        spec=spec,
                    )
                    log.debug(
                        "filler=%s chars=%d(adj %+d) off=%d(%+d) → %s",
                        spec.name,
                        real_chars,
                        adjust,
                        offset,
                        off_delta,
                        validation.reason,
                    )
                    if ok and new_index is not None:
                        insertion = Insertion(
                            token_index_before=target,
                            token_index_after=new_index,
                            token_id=decision.token_id,
                            token_text=decision.token,
                            filler_name=spec.name,
                            filler_char=spec.char,
                            filler_codepoint=codepoint_str(spec.char),
                            char_count=real_chars,
                            token_cost=validation.inserted_tokens,
                            segment_index=seg_index,
                            char_offset=offset,
                            attempt=attempts,
                            structural=spec.structural,
                            note="%s（补偿 %+d）" % (spec.note, adjust)
                            if adjust
                            else spec.note,
                        )
                        return True, insertion, validation, candidate

                    last_validation = validation
                    if attempts >= budget:
                        break
                if attempts >= budget:
                    break
            if attempts >= budget:
                break

        return (
            False,
            None,
            (last_validation or ValidationResult(False, "未执行任何尝试")),
            working,
        )

    def _verify_candidate(
        self,
        working: str,
        candidate: str,
        enc: Encoded,
        target: int,
        anchor: List[int],
        payload_chars: int,
        spec: FillerSpec,
    ) -> Tuple[bool, Optional[int], ValidationResult]:
        """把候选文本真正 tokenize 一遍并做双条件验收。"""
        encoded_after = self.tokenizer.encode_with_tokens(candidate)
        ids_after = encoded_after.ids
        inserted = len(ids_after) - len(enc)

        result = ValidationResult(
            ok=False,
            period=self.config.period,
            token_count_before=len(enc),
            token_count_after=len(ids_after),
            inserted_tokens=inserted,
        )

        # 条件 A：前缀零破坏
        if ids_after[:target] != enc.ids[:target]:
            result.reason = "前缀被破坏（插入点落在已对齐区间内）"
            result.notes.append("ids2[:%d] != ids[:%d]" % (target, target))
            return False, None, result

        # 条件 B：目标 token 归位到相位 0
        locate = self.validator.locate_anchor(
            anchor,
            ids_after,
            hint=target + inserted,
            tolerance=max(_LOCATE_TOLERANCE_BASE, inserted + _LOCATE_TOLERANCE_BASE),
        )
        result.locate = locate
        if not locate.found or locate.index is None:
            result.reason = "定位失败：%s" % locate.detail
            return False, None, result

        new_index = locate.index
        if new_index < target:
            result.reason = "目标 token 被前移（%d < %d）" % (new_index, target)
            return False, None, result

        result.new_index = new_index
        result.aligned = (new_index % self.config.period) == 0
        if inserted != payload_chars:
            result.notes.append(
                "token 增量 %d 与插入字符数 %d 不符（BPE 合并/拆分）"
                % (inserted, payload_chars)
            )
        if not result.aligned:
            result.reason = "未对齐：新 index=%d，%d %% %d == %d" % (
                new_index,
                new_index,
                self.config.period,
                new_index % self.config.period,
            )
            return False, None, result

        result.ok = True
        result.reason = "已对齐：新 index=%d，%d %% %d == 0" % (
            new_index,
            new_index,
            self.config.period,
        )
        return True, new_index, result

    def _filler_chain(self) -> List[FillerSpec]:
        if self.config.filler_priority:
            specs = []
            for name in self.config.filler_priority:
                try:
                    specs.append(self.registry.get(name))
                except KeyError:
                    log.warning("忽略未知占位符：%s", name)
            if specs:
                return specs
        chain = self.registry.fallback_chain()
        # 结构有干扰的排到最后
        chain.sort(key=lambda s: (s.structural, s.declared_cost != 1))
        return chain

    @staticmethod
    def _chars_for(delta: int, spec: FillerSpec) -> int:
        """把"需要推进的 token 位数"换算成"要插入多少个字符"。"""
        if delta <= 0:
            return 0
        if spec.declared_cost <= 0:  # pragma: no cover - 防御
            return delta
        # 向上取整；cost=1 时恒等于 delta
        return -(-delta // spec.declared_cost)

    def _insertion_offset(
        self, text: str, segment: Optional[Segment], decision
    ) -> Tuple[int, int]:
        """按 ``insert_mode`` 计算插入点（字符偏移）与段落号。

        inline 模式下插入点紧贴目标 token 正前方（段落号记 ``-1``，仅用于报告）；
        segment 模式下插入点取段落首部第一个非空白字符之前。
        """
        if self.config.insert_mode == INSERT_MODE_INLINE:
            return self._token_char_offset(text, decision), -1
        if segment is None:
            return 0, -1
        return segment.insert_offset(), segment.index

    @staticmethod
    def _token_char_offset(text: str, decision) -> int:
        """按 token 文本在文中的位置定位；找不到时返回 0。"""
        needle = decision.text
        if not needle:
            return 0
        found = text.find(needle)
        return found if found >= 0 else 0

    def _restore(self, text: str) -> Tuple[str, int]:
        """剥离"可安全剥离"的占位字符（不可见、无结构干扰的候选）。

        结构型候选（``lf`` 换行符）不参与剥离：其字符与原文合法换行无法区分，
        纳入剥离会破坏段落结构。
        """
        return strip_fillers(text, sorted(self._strippable))

    # ------------------------------------------------------------------ 还原
    def restore(self, text: str) -> Tuple[str, int]:
        """公开的还原接口：剥离全部可安全剥离的占位字符。

        只处理不可见、无结构干扰的候选；结构型占位符（``lf``）的换行符
        无法与原文换行区分，故保留不动。
        """
        return self._restore(text)
