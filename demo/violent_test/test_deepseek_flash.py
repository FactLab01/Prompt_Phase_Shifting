# -*- coding: utf-8 -*-
"""长文档压力测试：相位偏移能否让模型读出混合文档里的关键数值。

流程：读入 ``article.txt`` 构造提示词 → 用 ``weight`` + 宏观窗口模式做相位偏移 →
分别请求原始提示词与偏移后提示词 → 把两次提示词与两次回复写入 ``answer.log``。

运行（在本文件所在目录，或任意目录均可，脚本会自行切目录）::

    python demo/violent_test/test_deepseek_flash.py

凭证：``API_KEY`` 必填；``BASE_URL`` / ``MODEL`` / ``PROXY`` 可选，均从环境变量读取。
"""

import os
import sys

for s in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(s, "reconfigure"):
        s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import openai
from phase_shifting import PhaseShiftConfig, PhaseShifter

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = os.environ.get("BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("MODEL", "deepseek-flash")
PROXY = os.environ.get("PROXY", "")

#: 切片长度：目标模型一次前向处理的 token 数，关键 token 索引会被推进到
#: index % slice_length == 0。默认 4，不同模型需显式调整。
SLICE_LENGTH = 4

#: weight 模式的语义权重阈值，范围 [0, 1]
WEIGHT_THRESHOLD = 0.5

#: 宏观窗口长度（token 数），0 表示逐 token 判定；建议取目标模型步长的整数倍
WINDOW_SIZE = 128

with open("article.txt", "r", encoding="utf-8") as f:
    PROMPT = f"""[系统提示词]：
请仔细阅读以下数篇混合文档，然后用专业严谨的学术语言，回答在文本末尾提出的问题。请直接输出答案，不要输出思考过程。

[混合文档的内容]
{f.read().rstrip()}

[用户提问]：
请问在论文的实验中，DeepSeek-V4-Flash-Base 在 128K NIAH 任务里，跨 phase 准确率差具体是多少？预训练压缩模型的最差-最好相位差又是多少？"""


def ask_ai(prompt):
    """发送一次补全请求，返回模型回复文本。

    Args:
        prompt: 提示词，可以是相位偏移后的文本。

    Returns:
        模型回复字符串。
    """
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=120.0,
        **({"http_client": openai.DefaultHttpxClient(proxy=PROXY)} if PROXY else {}),
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=65535,
        stop=None,
        temperature=1.0,
        stream=False,
        reasoning_effort="low",
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    if not API_KEY:
        sys.exit("未检测到 API_KEY，请先设置环境变量 API_KEY 再运行。")

    # 长文档用 weight + 宏观窗口：只把 Peak Window 内的 token 判为关键 token，
    # 避免逐 token 判定在长文本上产生大量插入
    shifter = PhaseShifter(PhaseShiftConfig(
        slice_length=SLICE_LENGTH,
        key_mode="weight",
        weight_threshold=WEIGHT_THRESHOLD,
        window_size=WINDOW_SIZE,
        window_agg="count",
        max_chars=300000,
    ))

    r_obj = shifter.align(PROMPT)

    print("START.")
    original_answer = ask_ai(PROMPT)
    obj_answer = ask_ai(r_obj.aligned_text)

    with open("answer.log", "w", encoding="utf-8") as f:
        f.write("→ 原始提示词：\n" + PROMPT + "\n\n")
        f.write("→ 相位偏移后：\n" + r_obj.aligned_text + "\n\n")
        f.write("→ 原始回复：\n" + original_answer + "\n\n")
        f.write("→ 相位偏移后回复：\n" + obj_answer + "\n\n")

    print("DONE.")
