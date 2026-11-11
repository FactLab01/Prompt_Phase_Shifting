# Prompt Phase Shifting — 前置相位偏移

把用户输入里的**关键 token** 物理推进到目标模型的"注意力偏置位置"
（`token_index % slice_length == 0`，**默认 `slice_length = 4`**），
手段是在文本中插入**无害的零宽占位符**（默认 U+200B ZERO WIDTH SPACE），
插入后**立即重新 tokenize 全文做动态校验**，不达标就自动调整占位数量/种类并重试。

切片长度由调用方显式给定，**不写死在实现里**：不同模型一次前向处理的 token 数不同，
请按目标模型的特性传入 `slice_length`（SDK 参数、CLI `--slice-length` 均可）。

以可直接安装、导入、调用的 **Python SDK** 形式交付，CLI 仅为可选前端。

---

## 文档导航

| 文档 | 面向 | 内容 |
| --- | --- | --- |
| [使用教程（GitHub Wiki）](https://github.com/FactLab01/Prompt_Phase_Shifting/wiki/使用教程) | 使用者 | 安装、SDK 调用、命令行、三种判定模式对比、跑测试、调试排查 |
| [开发注意事项（GitHub Wiki）](https://github.com/FactLab01/Prompt_Phase_Shifting/wiki/开发注意事项) | 改动者 | 改动前必读的约束、性能守则、测试怎么加、自检清单 |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | 改动者 | **为什么这么设计**：问题定义、核心思路、算法不变量、分层理由、性能设计 |
| [`docs/SPEC.md`](docs/SPEC.md) | 两者 | 实现规范：术语、tokenizer 实测结论、模块划分、算法流程图、测试用例设计、分阶段计划 |
| [`CHANGELOG.md`](CHANGELOG.md) | 两者 | 版本变更记录 |

> 使用教程与开发注意事项托管在 GitHub Wiki，不随仓库分发。

---

## 环境要求

* Python ≥ 3.9
* 运行时唯一第三方依赖：[`deepseek-tokenizer`](https://pypi.org/project/deepseek-tokenizer/) ≥ 0.3.0

推荐（而非必须）在虚拟环境中开发，任选其一：

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

也可直接装进当前环境：`pip install .`

---

## 目录结构

```
Prompt_Phase_Shifting/
├─ demo/                         # 可真实运行的演示脚本（见下方「Demo 演示」）
│  ├─ demo1_basic_shift.py       # 基础相位偏移 + 偏移前后 AI 回复对比
│  ├─ demo2_key_modes.py         # 三种 Token 判定模式并排对比 + 调用路径 + 切片长度
│  ├─ demo3_results_and_reports.py  # 结果字段、序列化往返、批量调用
│  ├─ demo4_restore_and_idempotency.py   # restore 还原 + 幂等性
│  └─ violent_test/              # 长文档压力测试：对比原始/偏移后的模型回复
│     └─ test_deepseek_flash.py  # 入口脚本（需自备 article.txt，运行后生成 answer.log）
├─ docs/
│  └─ SPEC.md                    # 完整实现规范：术语、tokenizer 实测结论、E1–E12 边界用例、
│                                #   模块划分、算法流程图、测试用例设计、分阶段计划
├─ phase_shifting/               # 主包（可直接安装、导入、调用）
│  ├─ __init__.py                # 公共 API 再导出 + __version__ = 1.0.0
│  ├─ __main__.py                # `python -m phase_shifting` 入口
│  ├─ constants.py               # 【唯一事实来源】DEFAULT_SLICE_LENGTH 等默认值
│  ├─ errors.py                  # 【结构化异常体系】PhaseShiftingError 及其子类
│  ├─ config.py                  # 【SDK 配置契约】PhaseShiftConfig（slice_length 为规范字段）
│  ├─ api.py                     # 【SDK 门面】PhaseShifter / align_text / plan_text / restore_text
│  ├─ primitives.py              # 【C-ABI 原语层】位置参数 + 基础类型的稳定窄接口
│  ├─ cli.py                     # 【CLI 薄壳】只做参数解析 → 调 SDK → 渲染结果
│  ├─ tokenizer_api.py           # 【Tokenizer 封装】encode/decode/字节级工具/不可见字符可视化
│  ├─ fillers.py                 # 【占位符注册表】候选声明 + 运行时实测 token 成本
│  ├─ segments.py                # 【段落与插入点】段落切分、byte-level token 字符偏移推导
│  ├─ weights.py                 # 语义权重打分（weight 模式用）
│  ├─ keytokens.py               # 【关键 token 判定】auto / keywords / weight 三模式
│  ├─ validator.py               # 【动态校验器】锚点定位 + 相位校验
│  ├─ aligner.py                 # 【对齐引擎】迭代主循环 + 校验式插入
│  ├─ diffview.py                # 可读差异视图（把不可见字符渲染成 <ZWSP> 等记号）
│  ├─ report.py                  # 报告渲染：文本 / JSON / index 对比表
│  ├─ logsetup.py                # 【日志模块】控制台 + 文件双通道，UTF-8 安全输出
│  └─ py.typed                   # PEP 561 类型标记（随包分发）
├─ tests/                        # unittest 测试套件（324 项）
│  ├─ test_config.py             # 配置校验、slice_length 参数化、序列化往返
│  ├─ test_errors.py             # 异常层级、内置异常兼容、describe() 摘要
│  ├─ test_sdk_api.py            # PhaseShifter / align_text / plan / restore / 序列化
│  ├─ test_primitives.py         # 原语层签名约定、基础类型、slice_length 透传
│  ├─ test_packaging.py          # 切片长度不硬编码守卫、CLI 薄壳守卫、打包元数据
│  ├─ test_tokenizer_api.py      # 编码/解码/往返/字节映射/占位成本
│  ├─ test_fillers.py            # 注册表实测、隔离性校验、优先级链
│  ├─ test_segments.py           # 段落切分、token_start_offsets 前缀不变量
│  ├─ test_keytokens.py          # 三模式判定、filler token 永不被误判
│  ├─ test_validator.py          # 锚点定位、容忍带、定位失败
│  ├─ test_aligner.py            # E1–E12 边界用例、幂等性、occurrence 计数
│  ├─ test_diffview.py           # 差异视图渲染
│  ├─ test_report.py             # 报告渲染
│  └─ test_e2e.py                # CLI 端到端、restore 子命令、退出码
├─ pyproject.toml                # 构建系统 / 依赖 / 元数据 / console script
├─ CHANGELOG.md                  # 版本变更记录
└─ README.md                     # 本文件
```

---

## 快速开始

```bash
pip install .
```

```python
from phase_shifting import align_text

result = align_text("alpha beta gamma delta epsilon", slice_length=4)
print(result.status, result.summary(), result.aligned_text)
```

```bash
python -m phase_shifting --text "alpha beta gamma" --slice-length 4
```

更多用法见[使用教程（GitHub Wiki）](https://github.com/FactLab01/Prompt_Phase_Shifting/wiki/使用教程)。

---

## 三种 Token 判定模式

"哪些 token 需要被推进"由 `key_mode` 决定。三档模式在**精度与调用成本**之间取舍：

| 模式 | 判定规则 | 必填参数 | 干预强度 | 适用场景 |
| --- | --- | --- | --- | --- |
| `auto`（默认） | token 去掉空白与纯标点后仍非空 | 无 | **最强**（几乎所有实词） | 零配置兜底，不想或不能提供任何线索时 |
| `keywords` | token 文本包含任一关键词（大小写不敏感） | `keywords` 列表 | **最弱**（只命中指定词） | 明确知道要推什么，如密钥、专有名词、命令名 |
| `weight` | token 语义权重 ≥ `weight_threshold` | `weight_threshold`（默认 0.5） | **中等**（随阈值单调变化） | 只有模糊意图（"重要的词"），需按重要性梯度筛选 |

`weight` 模式还有一档**宏观窗口**子模式：不再逐 token 独立判定，而是先对全文
做单点打分得到权重数组，再用定长窗口滑动求出 **Peak Window（信息密度最高
的窗口）**，只把峰窗内的 token 判为关键 token。窗口长度建议取目标模型步长
（stride）的整数倍，常见 64 / 128 token。

| 子模式 | 判定视野 | 参数 | 关键 token 分布 |
| --- | --- | --- | --- |
| `weight` 逐 token（默认，`window_size=0`） | 单个 token 自身 | `weight_threshold` | 分散在全文 |
| `weight` + 宏观窗口 | 整篇文本的段落级密度 | `window_size`、`window_agg` | 集中在 Peak Window 内 |

### 调用方式

```python
from phase_shifting import align_text, PhaseShifter

# auto：默认模式，无需额外参数
align_text(text, slice_length=4)

# keywords：需提供关键词列表
align_text(text, slice_length=4, key_mode="keywords", keywords=["密钥", "KEY"])

# weight：需提供阈值，范围 [0, 1]
align_text(text, slice_length=4, key_mode="weight", weight_threshold=0.6)

# weight + 宏观窗口：窗口 16 token，聚合方式 sum（窗口内权重总和）
align_text(text, slice_length=4, key_mode="weight",
           weight_threshold=0.5, window_size=16)

# 门面对象形式（可复用）
shifter = PhaseShifter(slice_length=4, key_mode="keywords", keywords=["密钥"])
shifter.align(text)
```

CLI 等价参数：`--key-mode auto|keywords|weight`、`--keywords 密钥,KEY`、
`--weight-threshold 0.6`、`--window-size 16`、`--window-agg sum|count`。

`window_size` 为 `0`（默认）时关闭窗口化，退回逐 token 判定；该参数仅在
`key_mode="weight"` 下允许非零值。`window_agg` 二选一：

| `window_agg` | 窗口分数定义 | 适用 |
| --- | --- | --- |
| `sum` | 窗口内所有权重之和 | 默认；对长 token 占比高的段落敏感 |
| `count` | 窗口内权重 ≥ `weight_threshold` 的 token 个数 | 想看"高分词密度"而非总权重时 |

### 结果特征

| 模式 | `key_tokens` 数量 | 插入次数 | 典型结果 |
| --- | --- | --- | --- |
| `auto` | 多（数十个） | 多 | `token_delta` 大，干预范围广 |
| `keywords` | 少（几个） | 少 | `token_delta` 小，精确打击 |
| `weight` 逐 token | 介于两者之间 | 介于两者之间 | 阈值越高，`key_tokens` 越少 |
| `weight` + 窗口 | **最少**（仅峰窗内） | **最少** | 阈值与窗口越大，纳入的 token 越多 |

三者的 `status` 都应为 `ALIGNED`（只要存在未对齐的关键 token 且全部推进成功）。
差异只在"推进了多少个"，不在"推进得对不对"。

窗口化判定时，`plan_text()` 返回的 `AlignmentPlan` 会多出 `peak_window` 字段，
带有所定位峰窗的起止下标、分数与聚合方式，便于确认"核心段落"是否被找准：

```python
plan = plan_text(text, key_mode="weight", window_size=16)
print(plan.peak_window)  # {'start': 62, 'end': 77, 'score': 9.28, 'size': 16, 'agg': 'sum'}
```

全文短于一个窗口时自动退化为全流判定（`peak_window` 为 `None`）。

可运行的并排对比见 `demo/demo2_key_modes.py`。

---

## Demo 演示

`demo/` 下有 4 个演示脚本与 1 个长文档压力测试，全部**单文件自包含**：
自带导入、路径处理、凭证读取、错误提示与退出码，无需额外安装或配置。

| Demo | 演示内容 | 核心 API |
| --- | --- | --- |
| `demo1_basic_shift.py` | 基础相位偏移，同一提示词偏移前后的 AI 回复对比 | `align_text(text, slice_length=)` |
| `demo2_key_modes.py` | **三种判定模式并排对比** + weight 逐 token vs 宏观窗口 + 两条调用路径 + 多组切片长度 | `plan_text`、`align_text`、`PhaseShifter` |
| `demo3_results_and_reports.py` | 结果字段、序列化往返、批量调用 | `result.to_dict/to_json`、`AlignmentResult.from_dict`、`align_many` |
| `demo4_restore_and_idempotency.py` | restore 还原与多次执行的一致性（幂等性） | `PhaseShifter.align/restore`、`restore_text` |
| `violent_test/test_deepseek_flash.py` | 长文档压力测试：`weight` + 宏观窗口偏移后，对比原始/偏移两次真实模型回复 | `PhaseShifter`、`PhaseShiftConfig` |

### 运行方式

Demo 需要调用真实模型，凭证通过环境变量提供。也可以自行导出环境变量后
直接运行（任选其一）：

```bash
# Unix / macOS
export API_KEY="..." BASE_URL="https://api.deepseek.com/beta" MODEL="deepseek-flash"

# Windows PowerShell
$env:API_KEY="..."; $env:BASE_URL="https://api.deepseek.com/beta"; $env:MODEL="deepseek-flash"

python demo/demo1_basic_shift.py
```

长文档压力测试需自备 `demo/violent_test/article.txt`（UTF-8 长文本，数万字符以上，
不与仓库一同分发），脚本在其所在目录读取该文件，运行结束后把两次提示词与两次回复
写入同目录的 `answer.log`。

### 预期输出摘要

以 Demo 1 为例，输入一条含关键信息的提示词后：

```
偏移结果：status=ALIGNED  slice_length=4  token 29 → 94（+65）  插入 23 次

--- 回复 A：原始提示词 ---
作为 AI 我应该拒绝回答这个问题。……（模型未给出关键信息）

--- 回复 B：相位偏移后 ---
KEY-7734
```

关键信息在相位偏移后被模型稳定输出——因为承载它的 token 已被物理推进到
`index % slice_length == 0` 的注意力偏置位置。Demo 2 会展示三种判定模式
导致的干预强度差异，Demo 3 展示结果字段与序列化，Demo 4 展示还原与幂等保证。

Demo 2 在同一输入上的实测对比（全文 89 token，关键段落在后 1/3）：

```
[auto    ] 总 token 89，关键 token 76，需推进 56
[keywords] 总 token 89，关键 token  6，需推进  3
[weight  ] 总 token 89，关键 token 62，需推进 44
[weight+窗口] 总 token 89，关键 token 14，需推进 10
    峰窗：token #4..#19，分数 9.278（窗口 16，聚合 sum）
```

`weight` 逐 token 判定会把背景段落里的普通词一并算成关键 token；加上宏观窗口后，
判定收窄到 Peak Window 内，插入次数从 59 次降到 5 次。

---

## 测试

```bash
python -m unittest discover -s tests
```

---

## 许可

[MIT](LICENSE)
