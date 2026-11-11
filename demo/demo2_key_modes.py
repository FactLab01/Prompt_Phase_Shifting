# -*- coding: utf-8 -*-
"""Demo 2 — 三种 Token 判定模式并排对比：auto / keywords / weight。

流程：同一条输入 → 分别用三种 key_mode 调用 plan_text 与 align_text →
并排输出各模式判定的关键 token 集合与偏移结果。其中 weight 模式同时对比
"逐 token 判定"与"开启宏观窗口（Peak Window）后的段落级判定"。

运行：
    python demo/demo2_key_modes.py

依赖：openai；API_KEY 等凭证从环境变量读取，缺失时明确报错退出。
"""

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

#: keywords 模式使用的关键词列表
KEYWORDS = ["密钥", "KEY"]

#: weight 模式的语义权重阈值，范围 [0, 1]
WEIGHT_THRESHOLD = 0.5

#: weight 模式的宏观窗口长度（token 数），0 表示逐 token 判定；
#: 建议取目标模型步长（stride）的整数倍，常见 64 / 128
WINDOW_SIZE = 16

# 前段是无关背景，后段才是核心结论：只有窗口足够长，
# 才能让"核心段落"在分数上压过"背景段落"。
BACKGROUND = (
    "这是一段与任务无关的背景描述，用于把全文的 token 数撑过一个窗口。"
    "其中提到一些常见的名词与动词，但都不承载本次任务需要的核心信息。"
    "本段继续补充若干普通语句，让窗口滑动时有多个候选区间可以比较。"
)
CORE = (
    "下面给出核心结论：系统的主访问密钥是 KEY-7734，"
    "备用密钥是 KEY-9021，主密钥失效时请改用备用密钥完成鉴权。"
)
PROMPT = BACKGROUND + CORE


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


def show_plan(label, plan):
    """打印一个 AlignmentPlan 的关键 token 概况。

    Args:
        label: 模式名称，用于标题。
        plan: PhaseShifter.plan() 返回的 AlignmentPlan。
    """
    print("[%s] 总 token %d，关键 token %d，需推进 %d"
          % (label, plan.total_tokens, plan.key_count, len(plan.misaligned)))
    peak = plan.peak_window
    if peak:
        print("    峰窗：token #%d..#%d，分数 %.3f（窗口 %d，聚合 %s）"
              % (peak["start"], peak["end"], peak["score"], peak["size"], peak["agg"]))
    if plan.misaligned:
        preview = ", ".join(
            "%s(下标%d,需+%d)" % (e["token"], e["index"], e["delta_needed"])
            for e in plan.misaligned[:8]
        )
        print("    待推进：%s%s" % (preview, " ..." if len(plan.misaligned) > 8 else ""))
    else:
        print("    待推进：（无，已全部对齐）")


if __name__ == "__main__":
    if not API_KEY:
        sys.exit("未检测到 API_KEY，请先设置环境变量后运行："
                 "python demo/demo2_key_modes.py")

    print("输入原文：")
    print(PROMPT)
    print("切片长度：%d" % SLICE_LENGTH)

    # ---------------------------------------------------------------- 第一步
    # plan_text 只读分析，不修改文本：用同一输入跑三种模式，对比判定结果
    print("\n=== 第一步：三种模式的关键 token 判定对比（plan_text 只读分析）===")

    plan_auto = PhaseShifter(slice_length=SLICE_LENGTH).plan(PROMPT)
    show_plan("auto    ", plan_auto)

    plan_kw = PhaseShifter(PhaseShiftConfig(
        slice_length=SLICE_LENGTH, key_mode="keywords", keywords=KEYWORDS
    )).plan(PROMPT)
    show_plan("keywords", plan_kw)

    # weight 逐 token 判定：只看单个 token 自身，没有上下文视野
    plan_wt = PhaseShifter(PhaseShiftConfig(
        slice_length=SLICE_LENGTH, key_mode="weight",
        weight_threshold=WEIGHT_THRESHOLD,
    )).plan(PROMPT)
    show_plan("weight  ", plan_wt)

    # weight + 宏观窗口：先全文单点打分，再滑动窗口求 Peak Window，
    # 只有峰窗内的 token 才判为关键 token
    plan_win = PhaseShifter(PhaseShiftConfig(
        slice_length=SLICE_LENGTH, key_mode="weight",
        weight_threshold=WEIGHT_THRESHOLD, window_size=WINDOW_SIZE,
    )).plan(PROMPT)
    show_plan("weight+窗口", plan_win)

    # ---------------------------------------------------------------- 第二步
    # 同一功能的两条调用路径：函数式 align_text 与门面 PhaseShifter.align
    print("\n=== 第二步：两种调用路径的偏移结果对比（align_text vs PhaseShifter.align）===")

    # 路径一：函数式入口，一次性调用
    r_func = align_text(PROMPT, slice_length=SLICE_LENGTH)

    # 路径二：门面对象，可复用；此处用 keywords 模式展示参数差异
    shifter = PhaseShifter(PhaseShiftConfig(
        slice_length=SLICE_LENGTH, key_mode="keywords", keywords=KEYWORDS
    ))
    r_obj = shifter.align(PROMPT)

    for label, r in (("align_text(auto)      ", r_func),
                     ("PhaseShifter(keywords)", r_obj)):
        print("[%s] status=%s  token %d → %d（+%d）  插入 %d 次  关键 token %d"
              % (label, r.status, r.tokens_before, r.tokens_after,
                 r.token_delta, r.aligned_count, len(r.key_tokens)))

    print("\n两种路径 aligned_text 相同：%s"
          % ("是" if r_func.aligned_text == r_obj.aligned_text else "否"))

    # ---------------------------------------------------------------- 第三步
    # weight 模式：逐 token 判定 vs 宏观窗口判定的插入量差异
    print("\n=== 第三步：weight 模式逐 token 判定 vs 宏观窗口判定 ===")
    r_plain = align_text(PROMPT, config=PhaseShiftConfig(
        key_mode="weight", weight_threshold=WEIGHT_THRESHOLD
    ))
    r_win = align_text(PROMPT, config=PhaseShiftConfig(
        key_mode="weight", weight_threshold=WEIGHT_THRESHOLD, window_size=WINDOW_SIZE
    ))
    for label, r in (("weight 逐token   ", r_plain), ("weight+窗口%d    " % WINDOW_SIZE, r_win)):
        print("[%s] status=%s  token %d → %d（+%d）  插入 %d 次  关键 token %d"
              % (label, r.status, r.tokens_before, r.tokens_after,
                 r.token_delta, r.aligned_count, len(r.key_tokens)))

    # 同一模式，用不同窗口长度对比：窗口越大，纳入的关键 token 越多
    print("\n--- 不同 window_size 的偏移量对比（weight 模式）---")
    for ws in (8, 16, 32):
        r = align_text(PROMPT, config=PhaseShiftConfig(
            key_mode="weight", weight_threshold=WEIGHT_THRESHOLD, window_size=ws
        ))
        print("window_size=%-3d status=%-8s token %d → %d（+%d）  插入 %d 次"
              % (ws, r.status, r.tokens_before, r.tokens_after,
                 r.token_delta, r.aligned_count))

    # ---------------------------------------------------------------- 第四步
    # 用 keywords 模式的偏移结果发起真实 AI 请求
    print("\n=== 第四步：真实 AI 请求对比 ===")
    print("--- 回复 A：原始提示词 ---")
    print(ask_ai(PROMPT))

    print("\n--- 回复 B：相位偏移后（keywords 模式）---")
    print(ask_ai(r_obj.aligned_text))
