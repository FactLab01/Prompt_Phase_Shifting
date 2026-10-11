# -*- coding: utf-8 -*-
"""无害占位符注册表。

设计约束（全部由探针实验在 ``.venv`` 内实测得出，见 ``docs/SPEC.md`` 第 3 节）：

* 每个字符必须能被 deepseek-tokenizer **稳定编码**，且 **1 个字符 = N 个 token** 的关系
  在重复时保持线性（不合并、不拆分）；
* 必须 **不可见或低干扰**：零宽字符（Unicode ``Cf`` 类）优先；
* 必须 **不引入语义噪声**：不能与相邻字母/CJK 发生 BPE 合并。

按优先级排序的候选：

===== ======= ============= =========================
名称  字符    每字符token数  说明
===== ======= ============= =========================
zwsp  U+200B  1             零宽空格，首选
bom   U+FEFF  1             零宽不换行空格，备选
mix   U+200B+ 1             zwsp/bom 交替，防单一字符被过滤
wj    U+2060  2             word joiner，粒度 2，兜底
lf    U+000A  1             换行，会改变段落结构，最后兜底
===== ======= ============= =========================

``register()`` 会在运行时实测每个候选的 token 成本，成本与声明不符的候选会被自动降级，
保证"算法以为插了 1 个 token"与"实际真的是 1 个 token"一致。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .tokenizer_api import TokenizerHandle, codepoint_str, describe_char

log = logging.getLogger("phase_shifting.fillers")


@dataclass(frozen=True)
class FillerSpec:
    """一个占位符候选的完整描述。"""

    name: str
    char: str
    declared_cost: int  # 每个字符对应的 token 数（实测后校正）
    invisible: bool  # 是否不可见
    structural: bool  # 是否会改变段落结构（True = 有干扰）
    note: str = ""

    @property
    def codepoint(self) -> str:
        return codepoint_str(self.char)

    def describe(self) -> str:
        return "%s: %s [%s] cost=%d/token%s" % (
            self.name,
            describe_char(self.char),
            "不可见" if self.invisible else "可见",
            self.declared_cost,
            "，结构有干扰" if self.structural else "",
        )


#: 候选表（顺序即优先级）
CANDIDATES: List[FillerSpec] = [
    FillerSpec("zwsp", "​", 1, True, False, "零宽空格，token id 35020"),
    FillerSpec("bom", "﻿", 1, True, False, "零宽不换行空格，token id 15251"),
    FillerSpec("mix", "​", 1, True, False, "zwsp/bom 交替混合"),
    FillerSpec("wj", "⁠", 2, True, False, "word joiner，每字符 2 token"),
    FillerSpec("lf", "\n", 1, False, True, "换行符，会改变段落结构"),
]


def _measure(spec: FillerSpec, tokenizer: TokenizerHandle) -> FillerSpec:
    """实测该字符的 token 成本；不符则按实测值校正并告警。"""
    cost = tokenizer.probe_cost(spec.char)
    if cost != spec.declared_cost:
        log.warning(
            "占位符 %s 声明的每字符 token 数=%d，实测=%d，已按实测校正",
            spec.name,
            spec.declared_cost,
            cost,
        )
        return FillerSpec(spec.name, spec.char, cost, spec.invisible, spec.structural, spec.note)
    return spec


class FillerRegistry:
    """占位符注册表：按优先级提供可用候选，并负责生成"混合串"。"""

    def __init__(self, tokenizer: TokenizerHandle, verify: bool = True) -> None:
        self.tokenizer = tokenizer
        self._specs: Dict[str, FillerSpec] = {}
        for spec in CANDIDATES:
            final = _measure(spec, tokenizer) if verify else spec
            self._specs[spec.name] = final
        self._order = [s.name for s in CANDIDATES]
        log.info("占位符注册表就绪：%s", ", ".join(self._order))

    # ------------------------------------------------------------------ 查询
    def names(self) -> List[str]:
        return list(self._order)

    def get(self, name: str) -> FillerSpec:
        if name not in self._specs:
            raise KeyError("未知占位符：%s（可用：%s）" % (name, ", ".join(self._order)))
        return self._specs[name]

    def primary(self) -> FillerSpec:
        """首选的不可见、无结构干扰、成本为 1 的占位符。"""
        for name in self._order:
            spec = self._specs[name]
            if spec.invisible and not spec.structural and spec.declared_cost == 1:
                return spec
        # 退化：返回第一个不可见的
        for name in self._order:
            if self._specs[name].invisible:
                return self._specs[name]
        return self._specs[self._order[0]]

    def fallback_chain(self, exclude: Optional[str] = None) -> List[FillerSpec]:
        """重试时使用的占位符链（排除刚失败的那个）。"""
        chain = []
        for name in self._order:
            if name == exclude:
                continue
            chain.append(self._specs[name])
        return chain

    # ------------------------------------------------------------------ 生成
    def render(self, name: str, count: int) -> str:
        """生成 ``count`` 个占位字符组成的字符串。

        ``mix`` 会在 zwsp / bom 之间交替，降低"被下游按单一字符过滤"的风险。
        """
        if count <= 0:
            return ""
        spec = self.get(name)
        if name == "mix":
            zwsp = self.get("zwsp").char
            bom = self.get("bom").char
            return "".join(zwsp if i % 2 == 0 else bom for i in range(count))
        return spec.char * count

    def cost_of(self, name: str, count: int) -> int:
        """这 ``count`` 个字符预期消耗多少 token。"""
        return self.get(name).declared_cost * count

    # ------------------------------------------------------------------ 校验
    def verify_isolated(self, name: str, count: int) -> bool:
        """实测：把占位符夹在两个词中间，token 增量是否恰好等于预期成本。

        用于在插入前预检"该候选在此处是否会被合并或拆分"。
        """
        spec = self.get(name)
        payload = self.render(name, count)
        baseline = self.tokenizer.count_tokens("alphaalphabetagamma")
        actual = self.tokenizer.count_tokens("alpha%salphabetagamma" % payload)
        expected = spec.declared_cost * count
        delta = actual - baseline
        if delta != expected:
            log.debug(
                "占位符 %s x%d 隔离性校验失败：预期增量 %d，实际增量 %d",
                name,
                count,
                expected,
                delta,
            )
            return False
        return True

    def describe_all(self) -> List[str]:
        return [self._specs[name].describe() for name in self._order]


#: 显式 Unicode 转义形式，避免源码里出现不可见字符导致的误改
ZWSP = "​"
BOM = "﻿"
WORD_JOINER = "⁠"
LINE_FEED = "\n"
