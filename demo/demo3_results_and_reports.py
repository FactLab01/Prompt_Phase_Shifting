# -*- coding: utf-8 -*-
"""Demo 3 — 结果字段、序列化与报告输出：AlignmentResult 怎么用。

流程：构造输入 → 调用 SDK → 读取结果字段 → to_dict/to_json 序列化 →
还原回对象 → 用偏移后的文本发起真实 AI 请求。

运行：
    python demo/demo3_results_and_reports.py

依赖：openai；API_KEY 等凭证从环境变量读取，缺失时明确报错退出。
"""

import json
import os
import sys

for s in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
from phase_shifting import PhaseShiftConfig, PhaseShifter, align_text
from phase_shifting.tokenizer_api import visible

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = os.environ.get("BASE_URL", "https://api.deepseek.com/beta")
MODEL = os.environ.get("MODEL", "deepseek-chat")
PROXY = os.environ.get("PROXY", "")

SLICE_LENGTH = 4
KEYWORDS = ["密钥", "KEY"]

PROMPT = (
    "请阅读下面这段材料并回答：材料中提到的访问密钥是 KEY-7734，"
    "备用密钥是 KEY-9021。请问主密钥是什么？只回答密钥本身。"
)


def ask_ai(prompt):
    """发送一次补全请求，返回模型回复文本。"""
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=60.0,
        **({"http_client": openai.DefaultHttpxClient(proxy=PROXY)} if PROXY else {}),
    )
    resp = client.completions.create(
        model=MODEL, prompt=prompt, max_tokens=512, n=1, stop=None, temperature=1.0
    )
    return resp.choices[0].text.strip()


if __name__ == "__main__":
    if not API_KEY:
        sys.exit("未检测到 API_KEY，请先设置环境变量后运行："
                 "python demo/demo3_results_and_reports.py")

    print("输入原文：")
    print(PROMPT)

    # 用 keywords 模式偏移，便于观察 key_tokens 字段
    result = align_text(PROMPT, config=PhaseShiftConfig(
        slice_length=SLICE_LENGTH, key_mode="keywords", keywords=KEYWORDS
    ))

    # ------------------------------------------------------- 直接读字段
    print("\n=== 结果字段 ===")
    print("status        = %r" % result.status)
    print("summary()     = %r" % result.summary())
    print("success       = %r" % result.success)
    print("slice_length  = %r" % result.slice_length)
    print("tokens_before = %d" % result.tokens_before)
    print("tokens_after  = %d" % result.tokens_after)
    print("token_delta   = %d" % result.token_delta)
    print("aligned_count = %d" % result.aligned_count)
    print("key_tokens    = %d 条" % len(result.key_tokens))
    print("insertions    = %d 条" % len(result.insertions))
    print("failures      = %d 条" % len(result.failures))

    print("\nkey_tokens 前 5 条（含原下标、token、判定理由）：")
    for e in result.key_tokens[:5]:
        print("  下标 %-4d token=%-10s reason=%s"
              % (e["index"], e["token"], e["reason"]))

    print("\ninsertions 前 5 条（含插入位置、占位符、数量）：")
    for e in result.insertions[:5]:
        print("  下标 %-4s filler=%-6s count=%-3s token=%s"
              % (e.get("index"), e.get("filler"), e.get("count"), e.get("token")))

    # ------------------------------------------------------- 序列化
    print("\n=== 序列化：to_dict / to_json / from_dict ===")
    payload = result.to_dict()
    print("to_dict() 键：%s" % ", ".join(sorted(payload.keys())))

    text = result.to_json(indent=2)
    print("to_json() 前 8 行：")
    for line in text.splitlines()[:8]:
        print("  " + line)

    # 配置对象同样支持序列化往返：to_dict → from_dict
    cfg = PhaseShiftConfig(
        slice_length=SLICE_LENGTH, key_mode="keywords", keywords=KEYWORDS
    )
    cfg_dict = cfg.to_dict()
    cfg_restored = PhaseShiftConfig.from_dict(cfg_dict)
    print("\nPhaseShiftConfig.to_dict()：%s" % cfg_dict)
    print("PhaseShiftConfig.from_dict() 还原后与原对象相等：%s" % (cfg_restored == cfg))

    # ------------------------------------------------------- 批量调用
    print("\n=== 批量调用：PhaseShifter.align_many ===")
    shifter = PhaseShifter(PhaseShiftConfig(
        slice_length=SLICE_LENGTH, key_mode="keywords", keywords=KEYWORDS
    ))
    batch = shifter.align_many([
        "第一段：密钥是 AAA-111。",
        "第二段：密钥是 BBB-222。",
    ])
    for i, r in enumerate(batch, 1):
        print("  第 %d 段：status=%s  token %d → %d（+%d）"
              % (i, r.status, r.tokens_before, r.tokens_after, r.token_delta))

    # ------------------------------------------------------- 真实 AI 请求
    print("\n=== 真实 AI 请求 ===")
    print("--- 回复 A：原始提示词 ---")
    print(ask_ai(PROMPT))

    print("\n--- 回复 B：相位偏移后 ---")
    print(ask_ai(result.aligned_text))
