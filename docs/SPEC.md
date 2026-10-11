# 前置相位偏移 Demo — 实现规范与分阶段计划

> 目标：把用户输入中的**关键 token** 通过"动态插入无害 Token"物理推进到 DeepSeek 的
> 注意力偏置位置（`token_index % 4 == 0`），且全程只在 `./.venv` 虚拟环境内运行。

---

## 1. 术语

| 术语 | 含义 |
| --- | --- |
| 相位（phase） | token 在序列中的下标 `i` 对 `period`（默认 4）取模的结果，即 `i % 4` |
| 目标相位 | 本 Demo 约定 `i % period == 0`，即下标能被 `period` 整除的位置 |
| 关键 token | 需要被对齐的 token，默认判定为"非空白、非纯标点"，可由关键词/权重阈值自定义 |
| 占位 token（filler） | 插入的不可见字符，在 tokenizer 下可稳定编码、不引入语义噪声 |
| 物理推进 | 不修改模型，只改文本，使关键 token 的真实下标发生位移 |

## 2. 虚拟环境（硬约束）

- 唯一允许的解释器：`./.venv/Scripts/python.exe`（Python 3.13.12）。
- 已预装：`deepseek-tokenizer==0.3.0`（纯 Python 实现，BPE + ByteLevel 预分词）。
- 任何安装/卸载必须形如：
  `./.venv/Scripts/python.exe -m pip install <pkg>`，**严禁触碰全局环境**。
- 当前 Demo 运行**零第三方依赖**（仅标准库 + deepseek-tokenizer），测试同样用标准库 `unittest`。

## 3. Tokenizer 行为结论（来自实测探针）

DeepSeek tokenizer 的预分词链为：`Digits → CJK → Pattern3 → ByteLevel(add_prefix_space)`，
ByteLevel 使用 GPT-2 字节映射，因此：

1. **零宽字符有专属单 token**：`U+200B ZWSP → id 35020`，重复 N 次编码为 N 个独立 token
   （`[35020, 35020, ...]`，线性、无合并、无拆分）→ **可作为 1:1 的相位推进单位**。
2. **`U+FEFF BOM`** 同样是 1 字符 1 token（`id 15251`），线性稳定，是 ZWSP 的备选。
3. **不可用作推进单位的字符**（必须排除）：
   - `LF`/`SP`/`TAB`/`NBSP`：连续 N 个会被合并成 **1 个** token（`\n\n` → 单个 token 271），
     位移量与字符数不成线性关系；
   - `ZWNJ`/`WJ`/`MONGOLIAN_VS`：1 个字符炸成 **2 个** token，粒度是 2，无法表达奇数位移；
   - `ZWJ`/`IDEOGRAPHIC_SP`：合并行为非线性（1,1,2,2,3...），不可控。
4. **零宽字符是 Unicode `Cf`（format）类**，不落在 `L*`/`M*`/`P*`/`S*`/`N*`/空白分支，
   因此预分词器既不与相邻字母合并，也不吞掉后续 `Ġ`（前导空格）token——
   插入后相邻词的 token 形态保持原样（探针已验证：`['Hello','Ġworld']` 结构不变）。
5. **幂等性**：对已对齐文本重复运行时，引擎先做对齐判定，已全部对齐即零干预返回；
   仅当仍有未对齐项时才剥离历史占位字符并从干净原文重新计算，
   因此同一份原文永远得到同一份结果，不会累积不可见字符。
   亦可显式调用 `python -m phase_shifting restore aligned.txt` 一键剥离占位字符。

## 4. 功能规格

### 4.1 输入来源（三选一，可组合优先级：`--file` > `--text` > stdin）

| 来源 | 参数 |
| --- | --- |
| 命令行参数 | `--text "..."` 或位置参数 |
| 标准输入 | 管道/重定向，`-` 表示显式读 stdin |
| 指定文件 | `--file path`（utf-8） |

### 4.2 关键 token 判定

| 模式 | 参数 | 规则 |
| --- | --- | --- |
| 默认 `--key-mode auto` | — | token 文本去空白/纯标点后非空即为关键 token |
| 关键词 | `--keywords a,b,c` | token 文本含任一关键词（大小写不敏感） |
| 权重阈值 | `--weight-threshold 0.5` | token 的"语义权重" ≥ 阈值即为关键 token |

语义权重打分（`phase_shifting/weights.py`）：
`w = 0.5·len_norm + 0.3·is_word + 0.2·rarity`，其中 `len_norm = min(len(tok),8)/8`，
`is_word` 为"是否字母/汉字/CJK 连续块"，`rarity = 1 - min(id, 100000)/vocab_size` 的归一化
近似（token id 越大通常越罕见）。该公式仅用于演示"可插拔权重"机制。

### 4.3 插入策略（最小干预）

对每个关键 token，若其下标 `i % period != 0`：

1. 计算所需位移 `delta = (period - i % period) % period`，若 `delta == 0` 则跳过；
2. `delta ∈ {1,2,3}`，**每个占位字符 = 1 个 token**（用 ZWSP/BOM 这类 1:1 字符），
   因此"插入 k 个字符"即可精确推进 k 位；
3. 占位符**插入到该 token 所属的"文档段落/片段"起始处**（见 4.4），保证原始段落结构不被打散；
4. 插入后**立即重新 tokenize 全文做动态校验**（见 4.5）；
5. 若未对齐（例如该处预分词导致占位符合并），则切换占位符种类或调整数量重试，
   直到成功或达到 `--max-retries`（默认 6）。

占位符种类（`phase_shifting/fillers.py`，按优先级排序）：

| 名称 | 字符 | 码点 | 每字符 token 数 |
| --- | --- | --- | --- |
| `zwsp` | 零宽空格 | U+200B | 1 |
| `bom` | 零宽不换行空格(BOM) | U+FEFF | 1 |
| `zwsp+bom` | 交替混合 | — | 1（每个字符） |
| `word_joiner` | 零宽连接符 | U+2060 | 2（仅兜底） |
| `newline` | 换行 | U+000A | 1（会改变段落结构，兜底） |

> 探针结论：只有前三种满足"1 字符 = 1 token 且线性、可独立、不与邻居合并"。
> `word_joiner` / `newline` 只在重试阶段作为兜底，并在报告中标注"结构有干扰"。

### 4.4 插入点定位

按空行（`\n\s*\n`）切分为段落，用于报告展示与段落号标注。**插入点紧贴目标 token**，
由 `segments.token_start_offsets()` 通过 ByteLevel 字节前缀累加精确推导：

```
raw   = b"".join(token_bytes(t) for t in tokens)   # 全部 token 的原始字节
shift = 0 或 ±1                                    # add_prefix_space 的整体位移
offs[k] = byte_to_char_map(text)[pos_k + shift]    # 字节偏移 → 字符下标
```

由此保证：

- 插入点落在 token 的前导空格（`Ġ`）之前，token 形态不被破坏；
- 插入点永不落在多字节字符内部，文本不会被劈坏；
- **前缀零破坏**：`encode(text[:offs[k]])` 等于 `ids[:k]`，已对齐 token 的相位
  永不受扰动 —— 这是循环收敛的根本保证。

> 设计约束：插入点若落在目标 token 所属**段落头部**，会推动该段落内所有 token 的
> 下标，造成"修好一个、推歪另一个"的级联抖动，无法保证收敛。
> `--insert-mode segment` 仅作为兼容模式保留。

候选插入点为 `base + {0, ±1, ±2, ±3}`，用于吸收多字节字符被 BPE 切到字节中间时
插入点只能回退到字符起点所带来的偏差。

### 4.5 动态校验（每次插入后）

```
candidate = working[:offset] + payload + working[offset:]
ids2      = tokenizer.encode(candidate)        # 立即重新 tokenize 全文
# 条件 A：前缀零破坏
if ids2[:t] != ids[:t]:        -> 拒绝该候选
# 条件 B：目标 token 归位
j = ids2 中目标 id 首次出现且 >= t 的位置
if j is None or j % period != 0: -> 换候选重试
```

定位方式：以"关键 token 的原始 id 序列"为锚点，在新 token 流中做**带偏移的滑窗匹配**，
因此即使插入导致 BPE 边界变化，也能判断关键 token 是否仍然完整存在。

数量候选为 `delta + {0, 1, 2, -1}`，用于吸收 BPE 合并/拆分导致的 token 增量偏差。

### 4.6 输出

- 对齐前后 token index 对比表（原文 index → 新 index → 是否命中目标相位）；
- 插入占位符详情：字符（可读化）、Unicode 码点、字符数、对应 token 数、插入位置（段落号/字符偏移）；
- 整体结论：`ALIGNED` / `PARTIAL` / `FAILED`，及失败原因；
- 可选 `--diff`：给出插入前后的可读差异视图（用可见记号 `<ZWSP>` 标注不可见字符）。

## 5. 边界与健壮性场景（必须全部覆盖）

| # | 场景 | 预期行为 |
| --- | --- | --- |
| E1 | 空输入 | 直接返回 `ALIGNED`（0 token，无需插入），不抛异常 |
| E2 | 纯空白输入 | 同上，`ALIGNED`，无插入 |
| E3 | 已天然对齐（所有关键 token 下标 % 4 == 0） | 零插入，输出"无需插入" |
| E4 | 多个连续 token 均未对齐 | 逐个处理，前缀零破坏保证已对齐 token 不受扰动，最终全部对齐 |
| E5 | 超长输入 | `--max-chars` 上限保护（超出抛 `ValueError`）；测试用数百段文本验证不超时 |
| E6 | 占位符被 tokenizer 意外合并/拆分 | 双条件验收捕获 → 切换插入点/数量/种类重试 → 仍失败则记入 `failures` |
| E7 | 重复运行（幂等性） | 已对齐文本二次运行零插入、结果一致；提供 `restore` 子命令还原 |
| E8 | 关键词/权重模式无匹配 | 明确提示"未识别到关键 token"，不插入 |
| E9 | 单段落 / 无段落分隔 | 退化为单段处理，插入点仍紧贴目标 token |
| E10 | 目标 token 定位丢失（插入后 BPE 边界变化） | 记入 `failures`，状态降为 `PARTIAL`/`FAILED`，已成功的插入仍保留 |
| E11 | 达到重试上限 | 停止并报告 `PARTIAL`，列出未对齐的 token |
| E12 | 多字节字符（CJK / emoji）被 BPE 切到字节中间 | 插入点回退到字符起点 + 候选插入点重试，文本不被劈坏 |

## 6. 模块划分

```
phase_shifting/
├── __init__.py        # 包导出与版本
├── __main__.py        # python -m phase_shifting 入口
├── cli.py             # argparse CLI 入口、输入源解析、报告渲染
├── tokenizer_api.py   # deepseek-tokenizer 封装（encode/decode/token 字符串/健康检查）
├── fillers.py         # 无害占位符注册表（字符、码点、token 成本、稳定性探针）
├── segments.py        # 段落切分、token→段落归属、插入点计算
├── keytokens.py       # 关键 token 判定（auto / keywords / weight）
├── weights.py         # 语义权重打分
├── aligner.py         # 对齐引擎（最小干预算法 + 重试策略）
├── validator.py       # 动态校验器（重新 tokenize + 锚点滑窗定位）
├── diffview.py        # 插入前后可读差异视图
├── report.py          # 结果数据类与渲染
└── logsetup.py        # 日志模块（文件 + 控制台，可调级别）
```

## 7. 核心算法流程

```
                  ┌──────────────────────────────┐
                  │ 读取输入(file/text/stdin)     │
                  └──────────────┬───────────────┘
                                  ▼
                  ┌──────────────────────────────┐
                  │ tokenizer.encode(text)        │
                  │ → ids[] / token_strs[]        │
                  └──────────────┬───────────────┘
                                  ▼
                  ┌──────────────────────────────┐
                  │ 判定关键 token（auto/kw/weight）│
                  └──────────────┬───────────────┘
                                  ▼
         ┌────────────────► 存在未对齐关键 token ? ──否──► ALIGNED(零插入)
         │                      是
         │                       ▼
         │      ┌───────────────────────────────┐
         │      │ 取第一个未对齐 token t          │
         │      │ delta = (period - i%p) % p     │
         │      │ base  = token_start_offsets(t) │
         │      └───────────────┬───────────────┘
         │                      ▼
         │      ┌───────────────────────────────┐
         │      │ 在 base±0/1/2/3 处尝试插入      │
         │      │ delta+0/1/2 个占位字符          │
         │      └───────────────┬───────────────┘
         │                      ▼
         │      ┌───────────────────────────────┐
         │      │ 动态校验：重新 encode 全文       │
         │      │ A. ids2[:t] == ids[:t]        │
         │      │ B. 目标 id 归位到相位 0         │
         │      └───────┬───────────────┬───────┘
         │          A∧B 均满足        A 或 B 失败
         │              │               │
         │              ▼               ▼
         │        记录插入明细    切换插入点/数量/占位种类
         │              │          （重试 ≤ max_retries）
         └──────────────┴──────────────┘
                        ▼
         ┌──────────────────────────────┐
         │ 输出 index 对比 / 占位符详情 /  │
         │ 差异视图 / 最终结论             │
         └──────────────────────────────┘
```

## 8. 测试用例设计（`tests/`，标准库 unittest）

| 文件 | 覆盖 |
| --- | --- |
| `test_tokenizer_api.py` | 封装正确性、encode/decode 往返、token 字符串可见化、`locate_index_nth`、异常输入 |
| `test_fillers.py` | 每个候选占位符的 token 成本实测、1:1 线性、不与邻居合并、优先级链 |
| `test_segments.py` | 段落切分（空行/单段/CRLF）、token→段落归属、`token_start_offsets` 前缀不变量、字节↔字符映射 |
| `test_keytokens.py` | auto 判定（空白/纯标点/中英混排）、关键词模式、权重阈值模式、filler token 永不被误判 |
| `test_aligner.py` | 天然对齐、单 token 未对齐、多个连续未对齐、超长输入、重试上限、幂等性、语义完整性（换行/段落保留） |
| `test_validator.py` | 锚点滑窗定位、定位丢失、容忍带、合并/拆分场景下的校验失败 |
| `test_diffview.py` | 差异视图渲染（不可见字符转 `<ZWSP>` 记号） |
| `test_report.py` | 报告渲染（文本 / JSON / index 对比表） |
| `test_e2e.py` | CLI 端到端：`--file`、stdin、`--text`、`--diff`、`restore` 子命令、退出码、幂等性 |

## 9. 分阶段执行计划

| 阶段 | 内容 | 完成判据 | 状态 |
| --- | --- | --- | --- |
| P0 | 环境探查 + 探针实验 | 已确认 ZWSP/BOM 为 1:1 token | ✅ |
| P1 | 规范文档（本文件） | 本文档落地 | ✅ |
| P2 | 基础设施模块 | `tokenizer_api` / `fillers` / `logsetup` 可独立运行 | ✅ |
| P3 | 结构模块 | `segments` / `keytokens` / `weights` 单测通过 | ✅ |
| P4 | 核心引擎 | `aligner` / `validator` 单测通过 | ✅ |
| P5 | CLI 与报告 | `python -m phase_shifting --text ...` 输出完整报告 | ✅ |
| P6 | 测试全套 + README | `unittest discover` 166 项全绿，README 说明安装/运行/调试 | ✅ |

## 10. 交付声明

- 虚拟环境路径：`c:/Users/Administrator/Documents/Test/Prompt_Phase_Shifting/.venv`
- 运行方式：`.venv/Scripts/python.exe -m phase_shifting ...`
- 调试方式：`.venv/Scripts/python.exe -m unittest discover -s tests -v`
- 依赖变更：无（仅使用已预装的 `deepseek-tokenizer==0.3.0`，未污染全局环境）
