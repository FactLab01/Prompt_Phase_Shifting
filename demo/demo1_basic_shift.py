# -*- coding: utf-8 -*-
"""Demo 1 — 基础相位偏移：同一提示词，偏移前后 AI 回复对比。

流程：构造输入 → 调用 SDK → 用两种形态发起真实 AI 请求 → 对比回复。

运行：
    python demo/demo1_basic_shift.py

依赖：openai；API_KEY 等凭证从环境变量读取，缺失时明确报错退出。
"""

import os
import sys

for s in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8", errors="replace")

# 未 pip install 时，把仓库根目录加入 sys.path 以便导入 SDK
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
from phase_shifting import align_text
from phase_shifting.tokenizer_api import visible

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = os.environ.get("BASE_URL", "https://api.deepseek.com/beta")
MODEL = os.environ.get("MODEL", "deepseek-chat")
PROXY = os.environ.get("PROXY", "")

#: 切片长度：目标模型一次前向处理的 token 数，关键 token 索引会被推进到
#: index % slice_length == 0。默认 4，不同模型需显式调整。
SLICE_LENGTH = 4

PROMPT = (
    "请阅读下面这段材料并回答：材料中提到的访问密钥是 KEY-7734。"
    "请问这个密钥是什么？只回答密钥本身。"
)


def ask_ai(prompt):
    """发送一次补全请求，返回模型回复文本。

    Args:
        prompt: 提示词，可以是相位偏移后的文本。

    Returns:
        模型回复字符串（已 strip）。
    """
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
                 "python demo/demo1_basic_shift.py")

    print("输入原文：")
    print(PROMPT)

    # 核心调用：align_text 一次性完成相位偏移，返回结构化结果对象
    result = align_text(PROMPT, slice_length=SLICE_LENGTH)

    print("\n偏移结果：status=%s  slice_length=%d  token %d → %d（+%d）  插入 %d 次"
          % (result.status, result.slice_length, result.tokens_before,
             result.tokens_after, result.token_delta, result.aligned_count))
    print("偏移后文本（<ZWSP> 为插入的零宽占位符，屏幕上不可见）：")
    print(visible(result.aligned_text))

    # 同一提示词的两种形态，分别请求真实模型
    print("\n--- 回复 A：原始提示词 ---")
    print(ask_ai(PROMPT))

    print("\n--- 回复 B：相位偏移后 ---")
    print(ask_ai(result.aligned_text))
