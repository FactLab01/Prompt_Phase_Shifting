# -*- coding: utf-8 -*-
"""deepseek-tokenizer 封装层。

只依赖本地 ``.venv`` 中预装的 ``deepseek-tokenizer==0.3.0``，不引入任何第三方依赖。

封装要点：
* 统一 ``encode`` / ``decode`` 入口，屏蔽 ``DeepSeekTokenizer`` 与 ``PurePythonTokenizer``
  两层 API 的差异；
* 提供 ``encode_with_tokens``：同时拿到 token id 与 token 字符串（ByteLevel 表示）；
* 提供 ``decode_clean``：把 ByteLevel 的 ``Ġ`` 还原为普通空格，得到人类可读文本；
* 提供 ``visible``：把不可见字符替换为 ``<U+XXXX>`` 记号，用于报告与差异视图；
* 所有方法对空串 / 超长串安全。
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

log = logging.getLogger("phase_shifting.tokenizer_api")

# 单例，避免重复加载 tokenizer.json（约 129k 词表）
_CACHED: Optional["TokenizerHandle"] = None


@dataclass
class Encoded:
    """一次编码结果的完整快照。"""

    ids: List[int] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)

    def __bool__(self) -> bool:
        return bool(self.ids)


class TokenizerHandle:
    """对 ``deepseek_tokenizer.DeepSeekTokenizer`` 的薄封装。"""

    def __init__(self, tokenizer=None) -> None:
        if tokenizer is None:
            tokenizer = _load_default()
        self._tk = tokenizer
        self._vocab_size = int(tokenizer.vocab_size)
        log.info("tokenizer 就绪：vocab_size=%d", self._vocab_size)

    # ------------------------------------------------------------------ 基础
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> List[int]:
        """编码为 token id 列表。空串返回 ``[]``。"""
        if not text:
            return []
        return [int(i) for i in self._tk.encode(text)]

    def encode_with_tokens(self, text: str) -> Encoded:
        """编码并同时返回 token 字符串（ByteLevel 表示，含 ``Ġ`` 前缀）。"""
        if not text:
            return Encoded([], [])
        ids = [int(i) for i in self._tk.encode(text)]
        tokens = [str(t) for t in self._tk.convert_ids_to_tokens(ids)]
        return Encoded(ids, tokens)

    def token_str(self, token_id: int) -> str:
        """token id → ByteLevel token 字符串（未知 id 返回空串）。"""
        raw = self._tk.convert_ids_to_tokens(int(token_id))
        return str(raw) if raw is not None else ""

    def token_id(self, token: str) -> Optional[int]:
        """ByteLevel token 字符串 → id（不存在返回 ``None``）。"""
        got = self._tk.convert_tokens_to_ids(token)
        return int(got) if got is not None else None

    def convert_ids_to_tokens(self, ids):
        """透传：token id → ByteLevel token 字符串列表（或单个字符串）。"""
        if isinstance(ids, int):
            raw = self._tk.convert_ids_to_tokens([ids])
            return str(raw[0]) if raw else ""
        return [str(t) for t in self._tk.convert_ids_to_tokens([int(i) for i in ids])]

    def convert_tokens_to_ids(self, tokens):
        """透传：ByteLevel token 字符串 → id。"""
        if isinstance(tokens, str):
            return self.token_id(tokens)
        return [self.token_id(t) for t in tokens]

    def decode(self, ids: Sequence[int]) -> str:
        """token id → 原始文本（ByteLevel 还原，保留零宽字符）。"""
        if not ids:
            return ""
        return self._tk.decode([int(i) for i in ids])

    def decode_clean(self, ids: Sequence[int]) -> str:
        """token id → 人类可读文本（``Ġ`` → 空格，首尾空白裁掉）。"""
        if not ids:
            return ""
        return self._tk.decode([int(i) for i in ids], clean_up_tokenization_spaces=True)

    # ------------------------------------------------------------- 工具方法
    def count_tokens(self, text: str) -> int:
        """文本的 token 数（等价 ``len(encode(text))``，但避免构造大列表）。"""
        return len(self._tk.encode(text)) if text else 0

    def probe_cost(self, char: str) -> int:
        """单个字符在该 tokenizer 下的 token 成本（用于占位符注册表自检）。"""
        return self.count_tokens(char)

    def locate_index(
        self, text: str, token_id: int, hint: Optional[int] = None, tolerance: int = 8
    ) -> int:
        """在 ``text`` 的 token 流中定位 ``token_id`` 的第 0 次出现的下标。"""
        return self.locate_index_nth(text, token_id, 0, hint=hint, tolerance=tolerance)

    def locate_index_nth(
        self,
        text: str,
        token_id: int,
        occurrence: int = 0,
        hint: Optional[int] = None,
        tolerance: int = 8,
    ) -> int:
        """定位 ``token_id`` 第 ``occurrence`` 次出现（0 起）的下标。

        以 ``hint`` 为中心先在 ±``tolerance`` 窗口内找，找不到再全局取第
        ``occurrence`` 个；仍找不到时返回 ``hint``（保持表格可用）。
        """
        ids = self.encode(text)
        if not ids:
            return hint if hint is not None else 0
        candidates = [i for i, v in enumerate(ids) if v == int(token_id)]
        if not candidates:
            return hint if hint is not None else 0
        if occurrence >= len(candidates):
            occurrence = len(candidates) - 1
        if hint is None:
            return candidates[occurrence]
        lo, hi = max(0, hint - tolerance), min(len(ids) - 1, hint + tolerance)
        windowed = [c for c in candidates if lo <= c <= hi]
        pool = windowed or candidates
        return min(pool, key=lambda c: abs(c - hint))

    def health_check(self) -> Tuple[bool, str]:
        """自检：往返编码是否一致。"""
        sample = "phase shifting 相位偏移 123"
        ids = self.encode(sample)
        back = self.decode(ids)
        ok = back == sample
        return ok, f"vocab={self._vocab_size} roundtrip={'ok' if ok else 'MISMATCH'}"


def _load_default():
    """加载包内置的 tokenizer；失败时抛出带指引的异常。"""
    try:
        from deepseek_tokenizer import ds_token  # noqa: WPS433 (本地 .venv 内依赖)
    except ImportError as exc:  # pragma: no cover - 环境缺失时
        raise RuntimeError(
            "未找到 deepseek-tokenizer。请在项目虚拟环境中安装：\n"
            "    .venv\\Scripts\\python.exe -m pip install deepseek-tokenizer"
        ) from exc
    return ds_token


def get_tokenizer() -> TokenizerHandle:
    """获取全局单例 tokenizer 封装。"""
    global _CACHED
    if _CACHED is None:
        _CACHED = TokenizerHandle()
    return _CACHED


def reset_tokenizer_cache() -> None:
    """清空单例（测试用）。"""
    global _CACHED
    _CACHED = None


# ---------------------------------------------------------------------- 显示

#: 常见不可见字符的可读别名
_VISIBLE_ALIASES = {
    "\u200b": "<ZWSP>",  # zero width space
    "\u200c": "<ZWNJ>",
    "\u200d": "<ZWJ>",
    "\u2060": "<WJ>",  # word joiner
    "\ufeff": "<BOM>",
    "\u00a0": "<NBSP>",
    "\u3000": "<IDSP>",
    "\t": "<TAB>",
    "\r": "<CR>",
    "\n": "<LF>",
}


def codepoint_str(char: str) -> str:
    """字符 → ``U+200B`` 形式的码点串。"""
    if not char:
        return ""
    return "U+%04X" % ord(char[0]) if len(char) == 1 else "+".join(
        "U+%04X" % ord(c) for c in char
    )


def visible(text: str) -> str:
    """把文本中的不可见/控制字符替换为可读记号，其余原样保留。"""
    if not text:
        return ""
    out: List[str] = []
    for ch in text:
        if ch in _VISIBLE_ALIASES:
            out.append(_VISIBLE_ALIASES[ch])
        elif unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp", "Zs") and ch != " ":
            out.append("<U+%04X>" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def describe_char(char: str) -> str:
    """字符的详细人类可读描述（含 Unicode 名称）。"""
    if not char:
        return "<empty>"
    try:
        name = unicodedata.name(char)
    except ValueError:
        name = "<unnamed>"
    return "%s (%s, %s)" % (visible(char), codepoint_str(char), name)
