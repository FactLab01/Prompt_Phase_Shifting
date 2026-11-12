# -*- coding: utf-8 -*-
"""段落切分与插入点计算。

把原文按"空行"切分为若干段落（segment），每个 token 归属它所在的段落。
占位符默认插入到**目标 token 自身的字符偏移**处，从而只推进该 token
及其后续 token，不破坏前面已对齐 token 的相位。

插入点通过 **byte-level 前缀累加**精确推导（见 ``token_start_offsets``），
这是唯一能同时正确处理以下情况的方案：

* ByteLevel 的 ``Ġ`` 前导空格（``' beta'`` 的起始字节就是那个空格）；
* 多字节字符（CJK / emoji）被 BPE 切到字节中间（``'😀'`` → 3+1 字节）；
* 首部 ``add_prefix_space`` 造成的整体 1 字节位移。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

log = logging.getLogger("phase_shifting.segments")

#: 段落分隔：一个或多个空行（容忍 \r\n 与行尾空白）
_PARAGRAPH_SPLIT = re.compile(r"(?:\r?\n[ \t]*){2,}")


@dataclass
class Segment:
    """一个文档段落。"""

    index: int  # 段落在全文中的序号
    start: int  # 起始字符偏移（含）
    end: int  # 结束字符偏移（不含）
    text: str

    @property
    def length(self) -> int:
        return self.end - self.start

    def insert_offset(self) -> int:
        """占位符插入位置：段落内第一个"非换行、非空白"字符之前。

        这样插入物落在段落首部，既不可见，也不会把换行结构打散。
        """
        i = self.start
        limit = min(self.end, len(self.text))
        while i < limit and self.text[i] in " \t\r\n":
            i += 1
        return i

    def preview(self, limit: int = 24) -> str:
        snippet = self.text[:limit].replace("\n", "\\n")
        return snippet + ("…" if len(self.text) > limit else "")


def split_segments(text: str) -> List[Segment]:
    """把文本切分为段落列表。空文本返回空列表。"""
    if not text:
        return []
    segments: List[Segment] = []
    pos = 0
    for match in _PARAGRAPH_SPLIT.finditer(text):
        segments.append(Segment(len(segments), pos, match.start(), text[pos:match.start()]))
        pos = match.end()
    segments.append(Segment(len(segments), pos, len(text), text[pos:]))
    return segments


def segment_of_offset(segments: Sequence[Segment], offset: int) -> Optional[Segment]:
    """字符偏移 → 所属段落。"""
    for seg in segments:
        if seg.start <= offset < seg.end:
            return seg
    return segments[-1] if segments and offset >= segments[-1].end else None


def token_offsets(text: str, tokens: Sequence[str]) -> List[Tuple[int, int]]:
    """估算每个 token 在原文中的字符区间 ``[start, end)``。

    ByteLevel 表示里 ``Ġ`` 代表一个空格，``Ċ`` 代表换行；这里做一次贪心匹配，
    对未插入占位符的原始文本足够精确，仅用于判断 token 归属哪个段落。
    """
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for tok in tokens:
        literal = tok.replace("Ġ", " ").replace("Ċ", "\n")
        if not literal:
            offsets.append((cursor, cursor))
            continue
        found = text.find(literal, cursor)
        if found < 0:
            # 找不到时按 token 长度推进，保持下标与 tokens 一一对应
            offsets.append((cursor, min(len(text), cursor + len(literal))))
            cursor = min(len(text), cursor + len(literal))
            continue
        offsets.append((found, found + len(literal)))
        cursor = found + len(literal)
    return offsets


def map_tokens_to_segments(
    text: str, tokens: Sequence[str], segments: Sequence[Segment]
) -> List[Optional[Segment]]:
    """每个 token → 所属段落（列表下标与 token 一一对应）。"""
    if not segments:
        return [None] * len(tokens)
    offsets = token_offsets(text, tokens)
    return [segment_of_offset(segments, start) for start, _ in offsets]


# ------------------------------------------------------------------ byte-level 偏移

#: GPT-2 / ByteLevel 字节 ↔ 字符 映射表（deepseek-tokenizer 的 ``_bytes_to_unicode``）
_BYTE_ENCODER: dict = {}


def _build_byte_encoder() -> dict:
    """构造 ByteLevel 的 ``unicode_char -> byte`` 映射。"""
    if _BYTE_ENCODER:
        return _BYTE_ENCODER
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    _BYTE_ENCODER.update(dict(zip(cs, bs)))
    return _BYTE_ENCODER


def token_bytes(token: str) -> bytes:
    """ByteLevel token 字符串 → 原始字节。"""
    table = _build_byte_encoder()
    try:
        return bytes(table[ord(c)] for c in token)
    except KeyError:  # pragma: no cover - 非法字符兜底
        return token.encode("utf-8", errors="replace")


def byte_to_char_map(text: str, max_bytes: Optional[int] = None) -> List[int]:
    """长度 ``len(text.encode('utf-8')) + 1`` 的表：字节偏移 → 字符下标。

    落在多字节字符内部的字节偏移映射到该字符的起点，
    这样"在该字节处插入"永远不会把一个字符劈成两半。

    Args:
        text: 原始文本。
        max_bytes: 只建立到第 ``max_bytes`` 个字节为止（含）。调用方若只需要
            前缀区域的映射，传入目标字节数可避免为整篇文本建表。
    """
    total = len(text.encode("utf-8"))
    cap = total if max_bytes is None else max(0, min(int(max_bytes), total))
    table = [0] * (cap + 1)
    pos = 0
    for i, ch in enumerate(text):
        if pos > cap:
            break
        n = len(ch.encode("utf-8"))
        for j in range(n):
            if pos + j <= cap:
                table[pos + j] = i
        pos += n
    if pos <= cap:
        table[pos] = len(text)
    return table


def token_start_offsets(
    text: str, tokens: Sequence[str], limit: Optional[int] = None
) -> Optional[List[int]]:
    """每个 token 在 ``text`` 中的起始字符下标。

    通过 ByteLevel 字节累加得到，能正确处理：

    * ``Ġ`` 前导空格（``' beta'`` 的起始字节就是那个空格本身）；
    * 多字节字符被切到字节中间（``'😀'`` → ``' \\xf0\\x9f\\x98'`` + ``'\\x80'``）；
    * 首部 ``add_prefix_space`` 造成的整体 1 字节位移。

    Args:
        text: 原始文本。
        tokens: 该文本的 ByteLevel token 字符串序列。
        limit: 只计算前 ``limit`` 个 token 的偏移。调用方若只需要某个目标
            token 的位置，传入目标下标可把代价从 O(全文) 降到 O(前缀)。

    Returns:
        与 ``tokens`` 等长（或前 ``limit`` 个）的列表；若字节流与文本无法对应
        （理论上不会发生），返回 ``None``，调用方应改用其它定位策略。
        """
    if not tokens:
        return []
    total = len(tokens) if limit is None else max(0, min(limit, len(tokens)))
    if total == 0:
        return []
    tb = text.encode("utf-8")
    raw = b"".join(token_bytes(t) for t in tokens)
    if raw == tb:
        shift = 0
    elif raw == b" " + tb:
        shift = 1
    elif raw == tb + b" ":
        shift = -1
    else:
        log.debug("token 字节流与文本不匹配：%d vs %d 字节", len(raw), len(tb))
        return None
    # byte_to_char_map 为 O(文本字节数)，只需前缀偏移时限量建表
    table = byte_to_char_map(text, max_bytes=len(raw) + abs(shift) + 1)
    offsets: List[int] = []
    pos = 0
    for i in range(total):
        idx = pos + shift
        if idx < 0:
            idx = 0
        elif idx >= len(table):
            idx = len(table) - 1
        offsets.append(table[idx])
        pos += len(token_bytes(tokens[i]))
    return offsets


def strip_fillers(text: str, filler_chars: Sequence[str]) -> Tuple[str, int]:
    """从文本中移除全部占位字符，返回 ``(新文本, 移除个数)``。

    调用方需保证传入的字符与原文内容可区分（不可见且无结构干扰）；
    这是幂等还原的底层手段，配合"同一份原文永远得到同一份结果"的约定使用。
    """
    if not text or not filler_chars:
        return text, 0
    chars = set(filler_chars)
    kept: List[str] = []
    removed = 0
    for ch in text:
        if ch in chars:
            removed += 1
        else:
            kept.append(ch)
    return "".join(kept), removed
