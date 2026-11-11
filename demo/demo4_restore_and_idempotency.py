# -*- coding: utf-8 -*-
"""Demo 4 — restore 还原与幂等性：还原能力与多次执行的一致性。

流程：构造输入 → 偏移 → 对偏移结果再次偏移（验证一致）→ restore 还原 →
用原始与还原后的文本分别发起真实 AI 请求。

运行：
    python demo/demo4_restore_and_idempotency.py

依赖：openai；API_KEY 等凭证从环境变量读取，缺失时明确报错退出。
"""

import os
import sys

for s in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
from phase_shifting import PhaseShifter, restore_text
from phase_shifting.tokenizer_api import visible

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = os.environ.get("BASE_URL", "https://api.deepseek.com/beta")
MODEL = os.environ.get("MODEL", "deepseek-chat")
PROXY = os.environ.get("PROXY", "")

SLICE_LENGTH = 4

PROMPT = (
    "请阅读下面这段材料并回答：材料中提到的访问密钥是 KEY-7734。"
    "请问这个密钥是什么？只回答密钥本身。"
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
                 "python demo/demo4_restore_and_idempotency.py")

    print("输入原文：")
    print(PROMPT)

    # PhaseShifter 是可复用门面：同一实例可反复 align / restore
    shifter = PhaseShifter(slice_length=SLICE_LENGTH)

    # ------------------------------------------------------- 第一次偏移
    r1 = shifter.align(PROMPT)
    print("\n=== 第 1 次偏移 ===")
    print("status=%s  token %d → %d（+%d）  插入 %d 次"
          % (r1.status, r1.tokens_before, r1.tokens_after,
             r1.token_delta, r1.aligned_count))

    # ------------------------------------------------------- 第二次偏移
    # 输入第 1 次的结果，验证幂等：结果应与第 1 次完全相同
    r2 = shifter.align(r1.aligned_text)
    print("\n=== 第 2 次偏移（输入为第 1 次的结果）===")
    print("status=%s  token %d → %d（+%d）  插入 %d 次"
          % (r2.status, r2.tokens_before, r2.tokens_after,
             r2.token_delta, r2.aligned_count))
    print("两次结果完全相同：%s" % (r1.aligned_text == r2.aligned_text))

    # ------------------------------------------------------- restore 还原
    # restore_text 是函数式入口；门面对象同样提供 restore 方法
    restored = restore_text(r1.aligned_text)
    print("\n=== restore 还原 ===")
    print("还原后文本：%s" % visible(restored))
    print("与原文完全一致：%s" % (restored == PROMPT))
    print("门面 restore 与函数式入口结果一致：%s"
          % (shifter.restore(r1.aligned_text) == restored))

    # ------------------------------------------------------- 真实 AI 请求
    print("\n--- 回复 A：原始提示词 ---")
    print(ask_ai(PROMPT))

    print("\n--- 回复 B：相位偏移后 ---")
    print(ask_ai(r1.aligned_text))

    print("\n--- 回复 C：偏移后再 restore ---")
    print(ask_ai(restored))
