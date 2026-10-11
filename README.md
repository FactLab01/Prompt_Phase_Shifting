# Prompt Phase Shifting — 前置相位偏移

把用户输入里的**关键 token** 物理推进到 DeepSeek 的"注意力偏置位置"（`token_index % 4 == 0`），
手段是在文本中插入**无害的零宽占位符**（默认 U+200B ZERO WIDTH SPACE），
插入后**立即重新 tokenize 全文做动态校验**，不达标就自动调整占位数量/种类并重试。

> 虚拟环境（本项目全部分析与运行均在下列环境内完成，未污染全局 Python）：
>
> ```
> c:/Users/Administrator/Documents/Test/Prompt_Phase_Shifting/.venv
> ```
>
> Python 3.13.12 · `deepseek-tokenizer==0.3.0` · `pip 26.2.1`
> 下文所有命令都用 `.venv\Scripts\python.exe` 显式调用。

---

## 1. 目录结构

```
Prompt_Phase_Shifting/
├─ .venv/                        # 项目虚拟环境（已预装 deepseek-tokenizer==0.3.0）
├─ .tmp/                         # 临时探针/日志（已在 .gitignore）
├─ docs/
│  └─ SPEC.md                    # 完整实现规范：术语、tokenizer 实测结论、E1–E11 边界用例、
│                                #   模块划分、算法流程图、测试用例设计、分阶段计划
├─ phase_shifting/               # 主包
│  ├─ __init__.py                # 版本号 1.0.0
│  ├─ __main__.py                # `python -m phase_shifting` 入口
│  ├─ cli.py                     # 【CLI 入口】参数解析、输入读取（--text/--file/stdin）、输出编排
│  ├─ tokenizer_api.py           # 【Tokenizer 封装】encode/decode/字节级工具/不可见字符可视化
│  ├─ fillers.py                 # 【占位符注册表】候选声明 + 运行时实测 token 成本
│  ├─ segments.py                # 【段落与插入点】段落切分、byte-level token 字符偏移推导
│  ├─ weights.py                 # 语义权重打分（weight 模式用）
│  ├─ keytokens.py               # 【关键 token 判定】auto / keywords / weight 三模式
│  ├─ validator.py               # 【动态校验器】锚点定位 + 相位校验
│  ├─ aligner.py                 # 【对齐引擎】迭代主循环 + 校验式插入
│  ├─ diffview.py                # 可读差异视图（把不可见字符渲染成 <ZWSP> 等记号）
│  ├─ report.py                  # 报告渲染：文本 / JSON / index 对比表
│  └─ logsetup.py                # 【日志模块】控制台 + 文件双通道，UTF-8 安全输出
├─ tests/                        # unittest 测试套件（159 项）
│  ├─ test_tokenizer_api.py      # 编码/解码/往返/字节映射/占位成本
│  ├─ test_fillers.py            # 注册表实测、隔离性校验、优先级链
│  ├─ test_segments.py           # 段落切分、token_start_offsets 前缀不变量
│  ├─ test_keytokens.py          # 三模式判定、filler token 永不被误判
│  ├─ test_validator.py          # 锚点定位、容忍带、定位失败
│  ├─ test_aligner.py            # E1–E11 边界用例、幂等性、occurrence 计数
│  ├─ test_diffview.py           # 差异视图渲染
│  ├─ test_report.py             # 报告渲染
│  └─ test_e2e.py                # CLI 端到端、restore 子命令、退出码
└─ README.md                     # 本文件
```

---

## 2. 核心算法流程图

```
                     ┌──────────────────────────────────────┐
   用户输入 ────────▶│ 0. 读取输入（--text / --file / stdin） │
                     └──────────────────┬───────────────────┘
                                        ▼
                     ┌──────────────────────────────────────┐
                     │ 1. 幂等还原：剥离历史占位字符          │
                     │    （保证同一原文永远得到同一结果）     │
                     └──────────────────┬───────────────────┘
                                        ▼
                     ┌──────────────────────────────────────┐
                     │ 2. tokenize 全文 → 关键 token 判定     │
                     │    auto / keywords / weight           │
                     └──────────────────┬───────────────────┘
                                        ▼
              ┌─────────────────────────────────────────────┐
              │ 3. 迭代主循环（每轮重新 tokenize）            │
              │                                             │
              │   3.1 取"当前"第一个未对齐关键 token t        │
              │       delta = (4 - t.index % 4) % 4         │
              │   3.2 base = token_start_offsets()[t.index] │
              │       （byte-level 前缀累加，精确到字符）      │
              │   3.3 候选：占位种类 × 插入点(base±0/1/2/3)   │
              │              × 字符数(delta + 0/1/2)         │
              │   3.4 逐个真正插入 → 重新 tokenize → 双条件   │
              │       验收：                                │
              │         A. ids2[:t] == ids[:t]  前缀零破坏   │
              │         B. 目标 id 首次出现位置 j≥t 且        │
              │            j % 4 == 0                        │
              │   3.5 成功 → 记录明细，进入下一轮             │
              │       失败 → 换候选重试（≤ max_retries）      │
              └──────────────────┬──────────────────────────┘
                                        ▼
                     ┌──────────────────────────────────────┐
                     │ 4. 全部对齐？                          │
                     │    ALIGNED / PARTIAL / FAILED          │
                     └──────────────────┬───────────────────┘
                                        ▼
                     ┌──────────────────────────────────────┐
                     │ 5. 输出：index 对比表 + 占位符明细      │
                     │    （字符、U+码点、token 数、位置）     │
                     │    + 整体结论 + 可读差异视图            │
                     └──────────────────────────────────────┘
```

### 2.1 插入点为什么紧贴目标 token

插入点若落在目标 token 所属**段落头部**，会推动该段落内所有 token 的下标，
造成"修好一个、推歪另一个"的级联抖动，无法保证收敛。

因此插入点紧贴目标 token，并以**前缀零破坏**作为硬性验收条件：
每轮至少消灭一个未对齐 token，循环必然收敛。
以 `"alpha beta gamma delta epsilon zeta"`（7 个 token 全部未对齐）为例，
该策略下只需 **6 次插入**即可让全部 token 归位。

### 2.2 插入点为什么用 byte-level 前缀累加

ByteLevel 表示里 `Ġ` 代表空格，多字节字符可能被 BPE 切到字节中间
（`'😀'` → `' \xf0\x9f\x98'` + `'\x80'`）。只有把所有 token 的字节串拼接起来、
与 `text.encode("utf-8")` 对齐，才能得到每个 token 的**真实起始字符偏移**：

```
raw = b"".join(token_bytes(t) for t in tokens)      # 全部 token 的原始字节
shift = 0 或 ±1                                     # add_prefix_space 的整体位移
offs[k] = byte_to_char_map(text)[pos_k + shift]     # 字节偏移 → 字符偏移
```

由此得到的不变量（`tests/test_segments.py::test_token_start_offsets_prefix_property`）：

> 在 `offs[k]` 处切分，`encode(text[:offs[k]])` 必须等于 `ids[:k]`。

---

## 3. 占位符候选（全部在 `.venv` 内实测）

| 名称 | 字符 | 每字符 token 数 | 是否可见 | 结构干扰 | 说明 |
|------|------|----------------|---------|---------|------|
| `zwsp` | U+200B | 1 | 不可见 | 无 | 零宽空格，**首选**，id 35020 |
| `bom`  | U+FEFF | 1 | 不可见 | 无 | 零宽不换行空格，备选，id 15251 |
| `mix`  | U+200B/U+FEFF 交替 | 1 | 不可见 | 无 | 降低"被按单一字符过滤"的风险 |
| `wj`   | U+2060 | 2 | 不可见 | 无 | word joiner，粒度 2，兜底 |
| `lf`   | U+000A | 1 | **可见** | **有** | 换行符，会改变段落结构，最后兜底 |

实测中被否决的候选（不线性 / 会合并）：

| 字符 | 行为 | 结论 |
|------|------|------|
| U+200C ZWNJ | 1 字符 = 2 token | 无法表达奇数 delta |
| U+200D ZWJ | 1,1,2,2,3… 非线性 | 数量不可预测 |
| U+00A0 NBSP / 空格 / TAB | N 个字符合并成 1 token | 非线性 |
| U+3000 表意空格 | 非线性 | 否决 |

`FillerRegistry` 会在启动时**逐个实测**候选的 token 成本，与声明不符自动降级，
保证"算法以为插了 1 个 token"与"实际真的是 1 个 token"一致。

---

## 4. 安装 / 运行 / 调试（全部在 `.venv` 内）

> ⚠️ 本项目**禁止**向全局 Python 环境安装/卸载任何包。若虚拟环境尚未就绪，
> 只在 `.venv` 内操作：
>
> ```powershell
> py -3.13 -m venv .venv
> .venv\Scripts\python.exe -m pip install deepseek-tokenizer
> ```
>
> 当前环境已满足：`.venv` 中 `deepseek-tokenizer==0.3.0` 已预装。

### 4.1 Windows PowerShell 前置设置

控制台默认 GBK，打印零宽字符会崩，先设 UTF-8：

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "."
```

### 4.2 运行 Demo

```powershell
# 1) 直接传文本，输出完整报告
.venv\Scripts\python.exe -X utf8 -m phase_shifting --text "alpha beta gamma"

# 2) 只输出 token index 对比表
.venv\Scripts\python.exe -X utf8 -m phase_shifting --text "alpha beta gamma" --table --quiet

# 3) 从文件读
.venv\Scripts\python.exe -X utf8 -m phase_shifting --file input.txt

# 4) 从标准输入读（管道）
echo "key 关键信息 must be pushed" | .venv\Scripts\python.exe -X utf8 -m phase_shifting -

# 5) 关键词模式 + 差异视图 + JSON
.venv\Scripts\python.exe -X utf8 -m phase_shifting --file input.txt `
    --key-mode keywords --keywords 关键,secret --diff --json

# 6) 语义权重模式
.venv\Scripts\python.exe -X utf8 -m phase_shifting --text "..." --key-mode weight --weight-threshold 0.6

# 7) 把对齐后的文本写入文件
.venv\Scripts\python.exe -X utf8 -m phase_shifting --file input.txt --output aligned.txt

# 8) 还原（幂等性兜底：剥离全部占位字符）
.venv\Scripts\python.exe -X utf8 -m phase_shifting --restore aligned.txt

# 9) 列出全部占位符候选及其实测 token 成本
.venv\Scripts\python.exe -X utf8 -m phase_shifting --list-fillers
```

### 4.3 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--period / -p` | 4 | 相位周期，目标 `index % period == 0` |
| `--key-mode` | auto | `auto`（非空白非纯标点）/ `keywords` / `weight` |
| `--keywords` | — | 逗号分隔关键词列表（keywords 模式必填） |
| `--weight-threshold` | 0.5 | 语义权重阈值（weight 模式） |
| `--fillers` | 自动排序 | 占位符优先级，如 `zwsp,bom,mix` |
| `--insert-mode` | inline | `inline`（token 正前方，保证收敛）/ `segment`（段落头部，兼容模式） |
| `--max-retries` | 6 | 单个 token 的最大候选尝试预算 |
| `--max-chars` | 200000 | 输入字符上限保护 |
| `--no-restore` | 关 | 关闭幂等还原（默认会先剥离历史占位符） |
| `--diff` / `--side-by-side` | 关 | 差异视图 |
| `--table` | 关 | 只输出 index 对比表 |
| `--json` | 关 | JSON 报告 |
| `--verbose / -v` | 关 | 显示插入位置、尝试次数等细节 |
| `--log-level` | INFO | 日志级别 |
| `--log-file` | `.tmp/phase_shifting.log` | 日志文件路径 |

退出码：`0` 全部对齐 · `1` 部分对齐 · `2` 失败 · `3` 用法错误。

### 4.4 运行测试

```powershell
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONPATH = "."
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
```

当前状态：**166 项测试全部通过**（`OK`，约 18 分钟，含大量真实 tokenize）。

只跑某一组：

```powershell
.venv\Scripts\python.exe -X utf8 -m unittest tests.test_aligner.SemanticIntegrityTest -v
```

### 4.5 调试

```powershell
# 打开 DEBUG 日志，看每次候选插入的验收原因
.venv\Scripts\python.exe -X utf8 -m phase_shifting --text "..." --log-level DEBUG

# 指定日志文件
.venv\Scripts\python.exe -X utf8 -m phase_shifting --text "..." --log-file .tmp/debug.log
```

临时探针脚本一律写到 `.tmp/`（已在 `.gitignore` 中），不要写 `/tmp/` 或工作区外。

---

## 5. 边界用例设计（E1–E11）

| 编号 | 场景 | 期望行为 | 覆盖测试 |
|------|------|---------|---------|
| E1 | 空输入 `""` | `ALIGNED`，零干预 | `test_empty_input` |
| E2 | 纯空白 `"   \n\n  "` | `ALIGNED`，零干预 | `test_whitespace_only` |
| E3 | 已天然对齐 | `ALIGNED`，零干预 | `AlreadyAlignedTest` |
| E4 | 多个连续未对齐 token | 逐个推进，全部归位 | `test_multi_token_*` |
| E5 | 超长输入（数百段） | 不超时，状态 `ALIGNED`/`PARTIAL` | `test_long_input` |
| E6 | 输入超过 `--max-chars` | 抛 `ValueError` | `test_max_chars_limit` |
| E7 | 关键词模式无命中 | `NO_KEY_TOKENS`，文本不变 | `test_keywords_no_match` |
| E8 | 幂等性：对已对齐文本再跑一次 | 结果一致，**插入数为 0**，不累积不可见字符 | `IdempotencyTest` / `CliIdempotencyTest` |
| E9 | 多字节字符（CJK / emoji）被 BPE 切到字节中间 | 插入点回退到字符起点 + 候选重试，仍收敛 | `test_token_start_offsets_never_splits_char` |
| E10 | 同一 token id 多次出现 | occurrence 计数正确，锚点不串位 | `OccurrenceOfTest` |
| E11 | 占位符 token 自身 | 永不被误判为关键 token | `test_filler_token_never_key` |
| E12 | **换行/段落结构保留** | 换行数、段落数与原文完全一致；`restore` 后无损还原 | `SemanticIntegrityTest` |

---

## 6. 输出示例

`--text "alpha beta gamma delta epsilon zeta" --table --quiet`：

```
before  after  phase↓  phase→  aligned  token
     0      0       0       0    YES     alpha
     1      4       1       0    YES     beta
     2      8       2       0    YES     gamma
     3     12       3       0    YES     delta
     4     16       0       0    YES     epsilon
     5     20       1       0    YES     zeta
     6     24       2       0    YES     eta
```

完整报告（默认输出）额外包含：

* 【输入规模】字符数、对齐前后 token 数、关键 token 数
* 【插入的占位符详情】序号、filler 名、Unicode 码点、字符记号、字符数、token 数、所属段落、目标 token 及 `idx → idx`
* 【未对齐的 token】（若有）
* 【结论】`✅ ALIGNED` / `⚠️ PARTIAL` / `❌ FAILED` / `ℹ️ NO_KEY_TOKENS`

JSON 报告（`--json`）额外带 `aligned_text` 与 `aligned_text_visible`（不可见字符已转成 `<ZWSP>` 记号），便于程序化消费。

---

## 7. 设计约束与已知限制

1. **只依赖 `.venv`**：不引入任何第三方依赖，仅用标准库 + 预装的 `deepseek-tokenizer`。
2. **语义完整性**：默认只插入 Unicode `Cf` 类零宽字符，pre-tokenizer 既不与相邻
   字母/CJK 发生 BPE 合并，也不吞掉后续 token 的 `Ġ` 前缀。
3. **不变量**：插入后前 `t` 个 token 的 id 序列完全不变（`ids2[:t] == ids[:t]`），
   因此已对齐 token 的相位永不被扰动 —— 这是循环收敛的根本保证。
4. **LF 不参与还原剥离**：`lf`（换行）是**结构可见**占位符，与原文中合法的换行
   无法区分，因此**不得**纳入幂等还原的剥离集合 —— 否则 `restore` 会连带删掉
   原文的换行，造成不可逆的段落结构破坏。
   故 `_strippable` 只收"不可见且无结构干扰"的候选，
   并由 `test_newlines_preserved` / `test_restore_roundtrips_to_original` 守护。
5. **已知限制**：当目标 token 从一个多字节字符**内部**开始（emoji 被切成 3+1 字节，
   第 4 字节单独成 token）时，插入点只能回退到该字符起点，此时"紧贴插入"不可达。
   引擎通过 `_OFFSET_CANDIDATES`（base ±0/1/2/3）与数量补偿候选吸收这类偏差，
   实测全部收敛；极端情况下该 token 会被记入 `failures`，状态降为 `PARTIAL` 而不会死循环。
6. **幂等（两段式）**：
   * 先对**原始输入**编码判定 —— 若已全部对齐，直接零干预返回（第二次运行插入数为 0）；
   * 若仍有未对齐项，才剥离历史占位字符、从干净原文重新计算，
     保证"同一份原文永远得到同一份结果"。

---

## 8. 交付声明

* 使用的虚拟环境：**`c:/Users/Administrator/Documents/Test/Prompt_Phase_Shifting/.venv`**
* Python 3.13.12 · `deepseek-tokenizer==0.3.0`
* 全部编码分析、占位符实测、算法调试、测试运行均在该 `.venv` 内完成；
  未向全局 Python 环境安装或卸载任何包。
* 详细实现规范（含 tokenizer 行为实测数据、完整流程图、测试用例设计表）见 [`docs/SPEC.md`](docs/SPEC.md)。
