# -*- coding: utf-8 -*-
"""CLI 入口。

用法示例::

    # 1) 直接传文本
    .venv\\Scripts\\python.exe -m phase_shifting --text "Hello world 关键信息"

    # 2) 从文件读
    .venv\\Scripts\\python.exe -m phase_shifting --file input.txt

    # 3) 从标准输入读（管道）
    echo "..." | .venv\\Scripts\\python.exe -m phase_shifting -

    # 4) 关键词模式 + 差异视图 + JSON
    .venv\\Scripts\\python.exe -m phase_shifting --file input.txt ^
        --key-mode keywords --keywords 关键,secret --diff --json

    # 5) 还原（幂等性兜底）
    .venv\\Scripts\\python.exe -m phase_shifting --restore aligned.txt

子命令 ``restore`` 也可用：``python -m phase_shifting restore aligned.txt``。
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
    AlignConfig,
    Aligner,
)
from .diffview import render_diff, render_side_by_side
from .fillers import FillerRegistry
from .keytokens import KEY_MODES, KEY_MODE_AUTO, KeyTokenSelector
from .logsetup import close_logger, get_logger, io_text_wrapper
from .report import render_index_table_only, render_json, render_text
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

    # stdin
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data:
            return data, "stdin"

    return "", "empty"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase_shifting",
        description="前置相位偏移 Demo：把关键 token 物理推进到 DeepSeek 注意力偏置位置（index %% 4 == 0）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("positional", nargs="*", help="可选的位置参数文本（等同于 --text）")
    parser.add_argument("--text", "-t", help="用户原始文本")
    parser.add_argument("--file", "-f", help="从指定 UTF-8 文件读取文本")
    parser.add_argument("-", dest="dash", action="store_true", help="显式从标准输入读取")

    parser.add_argument("--period", "-p", type=int, default=4, help="相位周期，默认 4")
    parser.add_argument(
        "--key-mode",
        choices=list(KEY_MODES),
        default=KEY_MODE_AUTO,
        help="关键 token 判定模式，默认 auto",
    )
    parser.add_argument("--keywords", help="关键词列表，逗号分隔（key-mode=keywords 时必填）")
    parser.add_argument(
        "--weight-threshold", type=float, default=0.5, help="语义权重阈值（key-mode=weight）"
    )
    parser.add_argument(
        "--fillers", help="占位符优先级，逗号分隔，如 zwsp,bom,mix（默认自动排序）"
    )
    parser.add_argument(
        "--insert-mode",
        choices=(INSERT_MODE_SEGMENT, INSERT_MODE_INLINE),
        default=INSERT_MODE_INLINE,
        help="插入位置：token 正前方（默认，保证循环收敛）或段落头部（兼容模式）",
    )
    parser.add_argument("--max-retries", type=int, default=6, help="单个 token 的最大重试次数")
    parser.add_argument("--max-chars", type=int, default=200000, help="输入字符上限保护")
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


# ---------------------------------------------------------------------- 主流程
def run(args: argparse.Namespace) -> int:
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

    registry = FillerRegistry(tokenizer)
    filler_chars = [registry.get(n).char for n in registry.names()]
    selector = KeyTokenSelector(
        mode=args.key_mode,
        keywords=_parse_keywords(args.keywords),
        weight_threshold=args.weight_threshold,
        tokenizer=tokenizer,
        filler_chars=filler_chars,
    )
    config = AlignConfig(
        period=args.period,
        max_retries=max(1, args.max_retries),
        insert_mode=args.insert_mode,
        filler_priority=_parse_fillers(args.fillers),
        max_chars=args.max_chars,
        restore_first=not args.no_restore,
    )
    aligner = Aligner(tokenizer, selector, registry, config)

    def emit(message: str) -> None:
        logger.info(message)

    outcome = aligner.align(text, progress=emit)

    # ------------------------------------------------------------------ 输出
    if args.json:
        print(render_json(outcome, tokenizer))
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
    if outcome.status == "PARTIAL":
        return EXIT_PARTIAL
    if outcome.status == "FAILED":
        return EXIT_FAILED
    return EXIT_OK


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

    if args.period <= 0:
        print("错误：--period 必须为正整数", file=sys.stderr)
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

    registry = FillerRegistry(tokenizer)
    aligner = Aligner(tokenizer, KeyTokenSelector(), registry, AlignConfig())
    restored, removed = aligner.restore(text)
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
