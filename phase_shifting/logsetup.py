# -*- coding: utf-8 -*-
"""日志模块：同时输出到控制台与文件，级别可调。

日志文件默认落在 ``.tmp/phase_shifting.log``（``.tmp`` 已在 ``.gitignore`` 中），
避免污染仓库；也可通过 ``--log-file`` 自定义。
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

LOGGER_NAME = "phase_shifting"
DEFAULT_LOG_FILE = os.path.join(".tmp", "phase_shifting.log")
_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATEFMT = "%H:%M:%S"


def _safe_stream(encoding: str = "utf-8"):
    """返回一个尽量使用 UTF-8 的 stdout 包装，Windows 控制台 GBK 下也不崩。"""
    try:
        return io_text_wrapper(encoding)
    except Exception:  # pragma: no cover - 极端环境
        return sys.stderr


def io_text_wrapper(encoding: str = "utf-8"):
    import io

    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:
        return sys.stdout
    return io.TextIOWrapper(buffer, encoding=encoding, errors="backslashreplace", newline="")


def get_logger(
    level: str = "INFO",
    log_file: Optional[str] = DEFAULT_LOG_FILE,
    quiet: bool = False,
) -> logging.Logger:
    """获取（并配置）Demo 的单例 logger。

    Args:
        level: 日志级别，``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR``。
        log_file: 日志文件路径；传 ``None`` 表示只输出到控制台。
        quiet: 为 ``True`` 时不挂控制台 handler（仅写文件），便于管道输出。
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # 幂等：重复调用不重复挂 handler
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    if not quiet:
        console = logging.StreamHandler(_safe_stream())
        console.setFormatter(formatter)
        console.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.addHandler(console)

    if log_file:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover - 只读目录等
            logger.warning("无法写入日志文件 %s: %s", log_file, exc)

    return logger


def close_logger(logger: logging.Logger) -> None:
    """Flush 并关闭所有 handler（进程退出前调用）。"""
    for handler in list(logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)
