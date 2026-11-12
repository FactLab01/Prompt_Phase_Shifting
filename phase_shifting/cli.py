# -*- coding: utf-8 -*-
"""命令行前端：参数解析 → 调用 SDK → 渲染结果。

本模块不含任何业务逻辑。所有对齐行为都由 :mod:`phase_shifting.api`
（:class:`~phase_shifting.api.PhaseShifter`）承载；这里只负责把命令行参数
翻译成 :class:`~phase_shifting.config.PhaseShiftConfig`，再把
:class:`~phase_shifting.api.AlignmentResult` 渲染给人看。

用法示例（假设已 ``pip install .``）::

    python -m phase_shifting --text "Hello world 关键信息"
    python -m phase_shifting --file input.txt --slice-length 4 --key-mode keywords
    echo "..." | python -m phase_shifting -
    python -m phase_shifting restore aligned.txt

安装后也可直接用 ``phase-shifting`` 命令，与 ``python -m phase_shifting`` 等价。
若需要以编程方式调用，请直接使用 SDK :func:`phase_shifting.align_text`。
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from typing import List, Optional, Tuple

from . import __version__
from .aligner import (
    INSERT_MODE_INLINE,
    INSERT_MODE_SEGMENT,
    STATUS_ALIGNED,
    STATUS_FAILED,
    STATUS_PARTIAL,
)
from .api import AlignmentResult, PhaseShifter
from .config import PhaseShiftConfig
from .constants import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_SLICE_LENGTH,
    DEFAULT_WEIGHT_THRESHOLD,
    DEFAULT_WINDOW_AGG,
    DEFAULT_WINDOW_SIZE,
)
from .diffview import render_diff, render_side_by_side
from .errors import ConfigurationError, PhaseShiftingError
from .fillers import FillerRegistry
from .weights import WINDOW_AGGS, WINDOW_AGG_SUM
from .keytokens import KEY_MODES, KEY_MODE_AUTO
from .logsetup import close_logger, get_logger, io_text_wrapper
from .report import render_index_table_only, render_text
from .tokenizer_api import TokenizerHandle, get_tokenizer

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_FAILED = 2
EXIT_USAGE = 3


# ---------------------------------------------------------------------- 输入
def read_input(args: argparse.Namespace) -> Tuple[str, str]:
    """按优先级读取输入：``--file`` > 位置参数/``--text`` > stdin。

    Returns:
        ``(文本, 来源描述)``
    """
    if getattr(args, "file", None):
        path = args.file
        if not os.path.exists(path):
            raise FileNotFoundError("文件不存在：%s" % path)
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            return fh.read(), "file:%s" % path

    text = getattr(args, "text", None)
    if text:
        return text, "cli-arg"

    positional = getattr(args, "positional", None)
    if positional:
        return positional[0] if isinstance(positional, list) else positional, "cli-positional"

    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data:
            return data, "stdin"

    return "", "empty"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase_shifting",
        description=(
            "前置相位偏移：把关键 token 物理推进到目标模型的注意力偏置位置"
            "（index %% slice_length == 0）"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("positional", nargs="*", help="可选的位置参数文本（等同于 --text）")
    parser.add_argument("--text", "-t", help="用户原始文本")
    parser.add_argument("--file", "-f", help="从指定 UTF-8 文件读取文本")
    parser.add_argument("-", dest="dash", action="store_true", help="显式从标准输入读取")

    parser.add_argument(
        "--slice-length",
        "-p",
        type=int,
        default=DEFAULT_SLICE_LENGTH,
        dest="slice_length",
        help=(
            "切片长度：目标模型一次前向处理的 token 数，关键 token 索引会被推进到 "
            "index %% slice_length == 0。默认 %d；不同模型请显式传入"
            % DEFAULT_SLICE_LENGTH
        ),
    )
    parser.add_argument(
        "--period",
        type=int,
        default=None,
        dest="period",
        help="--slice-length 的兼容别名（已废弃，优先使用 --slice-length）",
    )
    parser.add_argument(
        "--key-mode",
        choices=list(KEY_MODES),
        default=KEY_MODE_AUTO,
        help="关键 token 判定模式，默认 auto",
    )
    parser.add_argument("--keywords", help="关键词列表，逗号分隔（key-mode=keywords 时必填）")
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=DEFAULT_WEIGHT_THRESHOLD,
        help="语义权重阈值（key-mode=weight）",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=(
            "宏观窗口长度（token 数），0 表示关闭、逐 token 判定；"
            "建议取目标模型步长的整数倍（key-mode=weight 时生效）"
        ),
    )
    parser.add_argument(
        "--window-agg",
        choices=list(WINDOW_AGGS),
        default=WINDOW_AGG_SUM,
        help="窗口内聚合方式：sum=权重总和，count=高分词个数（默认 sum）",
    )
    parser.add_argument(
        "--fillers", help="占位符优先级，逗号分隔，如 zwsp,bom,mix（默认自动排序）"
    )
    parser.add_argument(
        "--insert-mode",
        choices=(INSERT_MODE_SEGMENT, INSERT_MODE_INLINE),
        default=INSERT_MODE_INLINE,
        help="插入位置：token 正前方（默认，满足前缀零破坏）或段落头部",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="单个 token 的最大候选尝试预算",
    )
    parser.add_argument(
        "--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="输入字符上限保护"
    )
    parser.add_argument("--no-restore", action="store_true", help="不做幂等还原（默认会先剥离历史占位符）")

    parser.add_argument("--diff", action="store_true", help="输出插入前后的可读差异视图")
    parser.add_argument("--side-by-side", action="store_true", help="输出并排差异视图")
    parser.add_argument("--table", action="store_true", help="只输出 token index 对比表")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    parser.add_argument("--output", "-o", help="把对齐后的文本写入指定文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示更多细节")
    parser.add_argument("--quiet", "-q", action="store_true", help="静默模式，只输出报告")
    parser.add_argument("--log-level", default="INFO", help="日志级别，默认 INFO")
    parser.add_argument("--log-file", default=None, help="日志文件路径，默认 .tmp/phase_shifting.log")
    parser.add_argument(
        "--list-fillers", action="store_true", help="列出全部占位符候选及其 token 成本后退出"
    )
    parser.add_argument("--version", action="version", version="phase_shifting %s" % __version__)
    return parser


def _parse_keywords(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def _parse_fillers(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [f.strip() for f in raw.split(",") if f.strip()]


#: ``--period`` 与 ``--slice-length`` 冲突时的告警文案（两处调用点共用）
_PERIOD_DEPRECATED_WARNING = (
    "警告：--period 已废弃，且与 --slice-length 同时给出；以 --slice-length 为准"
)


def _slice_length_explicit(argv: Optional[List[str]]) -> bool:
    """命令行里是否显式给出了 ``--slice-length`` / ``-p``。"""
    tokens = list(sys.argv[1:] if argv is None else argv)
    return any(
        tok in ("--slice-length", "-p")
        or tok.startswith("--slice-length=")
        or tok.startswith("-p=")
        for tok in tokens
    )


def _resolve_slice_length(
    args: argparse.Namespace, argv: Optional[List[str]] = None
) -> int:
    """解析生效的切片长度。

    ``--slice-length`` 是正式参数；``--period`` 仅为兼容别名。两者都显式给出时
    以 ``--slice-length`` 为准，并由调用方打印 :data:`_PERIOD_DEPRECATED_WARNING`。
    """
    slice_length = args.slice_length
    if args.period is None:
        return slice_length
    return slice_length if _slice_length_explicit(argv) else args.period


# ---------------------------------------------------------------------- 主流程
def run(args: argparse.Namespace) -> int:
    """CLI 薄壳：解析参数 → 调 SDK → 渲染结果。

    本函数不含任何业务逻辑；对齐行为完全由 :mod:`phase_shifting.api` 承载。
    """
    logger = get_logger(
        level=args.log_level,
        log_file=args.log_file,
        quiet=args.quiet,
    )
    tokenizer: TokenizerHandle = get_tokenizer()

    if args.list_fillers:
        registry = FillerRegistry(tokenizer)
        print("占位符候选（按优先级）：")
        for i, line in enumerate(registry.describe_all(), start=1):
            print("  %d. %s" % (i, line))
        return EXIT_OK

    ok, msg = tokenizer.health_check()
    logger.info("tokenizer 自检：%s", msg)
    if not ok:
        logger.error("tokenizer 往返编码不一致：%s", msg)

    text, source = read_input(args)
    logger.info("输入来源：%s，长度 %d 字符", source, len(text))

    # ------------------------------------------------------- 组装 SDK 配置
    slice_length = _resolve_slice_length(args)

    try:
        config = PhaseShiftConfig(
            slice_length=slice_length,
            key_mode=args.key_mode,
            keywords=_parse_keywords(args.keywords),
            weight_threshold=args.weight_threshold,
            window_size=args.window_size,
            window_agg=args.window_agg,
            insert_mode=args.insert_mode,
            filler_priority=_parse_fillers(args.fillers),
            max_retries=max(1, args.max_retries),
            max_chars=args.max_chars,
            restore_first=not args.no_restore,
        )
    except ConfigurationError as exc:
        print("错误：%s" % exc, file=sys.stderr)
        return EXIT_USAGE

    shifter = PhaseShifter(config, tokenizer=tokenizer)

    def emit(message: str) -> None:
        logger.info(message)

    try:
        result = shifter.align(text, progress=emit)
    except PhaseShiftingError as exc:
        print("错误：%s" % exc, file=sys.stderr)
        logger.error("%s: %s", exc.code, exc)
        return EXIT_FAILED

    outcome = _to_outcome(result, tokenizer)

    # ------------------------------------------------------------------ 输出
    if args.json:
        print(result.to_json())
    elif args.table:
        print(render_index_table_only(outcome, tokenizer))
    else:
        print(render_text(outcome, tokenizer, verbose=args.verbose))

    if args.diff:
        diff = render_diff(outcome.original_text, outcome.aligned_text)
        print("\n【可读差异视图】（不可见字符已渲染为 <ZWSP> 等记号）")
        print(diff or "  （无差异）")

    if args.side_by_side:
        print("\n【并排视图】")
        print(render_side_by_side(outcome.original_text, outcome.aligned_text))

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as fh:
            fh.write(outcome.aligned_text)
        logger.info("对齐后文本已写入 %s", args.output)

    close_logger(logger)

    if outcome.status == STATUS_ALIGNED:
        return EXIT_OK
    if outcome.status == STATUS_PARTIAL:
        return EXIT_PARTIAL
    if outcome.status == STATUS_FAILED:
        return EXIT_FAILED
    return EXIT_OK


def _to_outcome(result: "AlignmentResult", tokenizer: TokenizerHandle):
    """把 SDK 的 :class:`AlignmentResult` 适配为报告层使用的对象。

    报告渲染只需要 ``status`` / ``original_text`` / ``aligned_text`` /
    ``period`` / ``total_tokens_before`` / ``total_tokens_after`` /
    ``key_report`` / ``insertions`` / ``failures`` / ``message`` 这些字段；
    这里用轻量适配器补齐，避免 CLI 直接依赖引擎内部类型。
    """
    from .aligner import AlignOutcome
    from .keytokens import KeyTokenDecision, KeyTokenReport

    report = KeyTokenReport(decisions=[])
    for entry in result.key_tokens:
        report.decisions.append(
            KeyTokenDecision(
                index=entry["index"],
                token_id=entry["token_id"],
                token=entry["token"],
                text=entry["token"],
                is_key=True,
                reason=entry.get("reason", ""),
            )
        )

    # SDK 的插入明细用 JSON 友好键，报告层用引擎字段名，此处做一次键名映射。
    _KEY_ALIASES = {
        "filler_name": "filler",
        "structural": "structural_interference",
    }

    class _InsertionView:
        """把插入明细字典还原为报告层期望的属性访问形式。"""

        def __init__(self, data: dict) -> None:
            self._data = data

        def __getattr__(self, name: str):
            key = _KEY_ALIASES.get(name, name)
            try:
                return self._data[key]
            except KeyError as exc:  # pragma: no cover - 防御
                raise AttributeError(name) from exc

    return AlignOutcome(
        status=result.status,
        original_text=result.original_text,
        aligned_text=result.aligned_text,
        period=result.slice_length,
        total_tokens_before=result.tokens_before,
        total_tokens_after=result.tokens_after,
        key_report=report,
        insertions=[_InsertionView(d) for d in result.insertions],
        failures=list(result.failures),
        message=result.message,
    )


def main(argv: Optional[List[str]] = None) -> int:
    # Windows 控制台默认 GBK，强制 UTF-8 输出，避免不可见字符打印崩溃
    global sys
    try:
        sys.stdout = io_text_wrapper("utf-8")
    except Exception:
        pass

    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)

    # 支持 `restore` 子命令
    if argv and argv[0] == "restore":
        return _run_restore(argv[1:], parser)

    args = parser.parse_args(argv)

    # --period 是 --slice-length 的兼容别名；两者都显式给出时以 --slice-length 为准
    resolved = _resolve_slice_length(args, argv)
    if (
        args.period is not None
        and args.period != args.slice_length
        and resolved == args.slice_length
        and _slice_length_explicit(argv)
    ):
        print(_PERIOD_DEPRECATED_WARNING, file=sys.stderr)
    args.slice_length = resolved

    # 切片长度的最终校验交给 SDK 配置层（ConfigurationError → EXIT_USAGE），
    # 这里只保留一次快速失败，避免把明显的非正数带进引擎。
    if args.slice_length <= 0:
        print(
            "错误：--slice-length 必须为正整数，收到 %r" % (args.slice_length,),
            file=sys.stderr,
        )
        return EXIT_USAGE

    try:
        return run(args)
    except FileNotFoundError as exc:
        print("错误：%s" % exc, file=sys.stderr)
        return EXIT_USAGE
    except ValueError as exc:
        print("错误：%s" % exc, file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:  # pragma: no cover
        print("已中断", file=sys.stderr)
        return 130


def _run_restore(rest: List[str], parser: argparse.ArgumentParser) -> int:
    """``python -m phase_shifting restore <file>``：剥离占位字符。"""
    restore_parser = argparse.ArgumentParser(prog="phase_shifting restore")
    restore_parser.add_argument("path", nargs="?", help="待还原的文件；缺省读 stdin")
    restore_parser.add_argument("-o", "--output", help="输出文件；缺省打印到 stdout")
    restore_parser.add_argument("--quiet", action="store_true")
    ns = restore_parser.parse_args(rest)

    logger = get_logger(quiet=ns.quiet)
    tokenizer = get_tokenizer()

    if ns.path:
        with open(ns.path, "r", encoding="utf-8-sig", newline="") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()

    shifter = PhaseShifter(tokenizer=tokenizer)
    restored = shifter.restore(text)
    removed = len(text) - len(restored)
    logger.info("剥离 %d 个占位字符（%d → %d 字符）", removed, len(text), len(restored))

    if ns.output:
        with open(ns.output, "w", encoding="utf-8", newline="") as fh:
            fh.write(restored)
        print("已写入 %s（剥离 %d 个占位字符）" % (ns.output, removed))
    else:
        sys.stdout.write(restored)
    close_logger(logger)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
