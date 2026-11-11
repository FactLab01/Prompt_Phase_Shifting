# DEVELOPMENT.md — 设计说明

本文件回答"**为什么这么设计**"。面向改动者与想理解实现原理的读者。

- 想知道"是什么、怎么跑" → 看 [`README.md`](README.md) 与[使用教程（GitHub Wiki）](https://github.com/FactLab01/Prompt_Phase_Shifting/wiki/使用教程)
- 想知道"代码怎么调、返回什么" → 看 `demo/` 下的 Demo 与源码 docstring
- 想知道"为什么这样设计" → 看本文件

---

## 1. 问题定义

大语言模型对**不同绝对位置的 token** 关注度并不均等。本项目针对的现象是：
当关键信息所在的 token 落在 `token_index % slice_length == 0` 这类"注意力偏置位置"时，
模型对该信息的遵循度显著提升；落在其他位置时则可能被忽略。

目标：**不改变提示词的可见内容**，只把关键 token 物理推进到偏置位置。

约束：不能改动模型、不能改动推理参数，只能在**输入文本的 token 序列**上做文章。

---

## 2. 核心思路：用不可见字符做位移单位

关键观察来自对 tokenizer 的实测探针（结论记录在 [`docs/SPEC.md`](docs/SPEC.md) §3）：

- 零宽字符（U+200B ZWSP、U+FEFF BOM）是 **1 字符 = 1 token**，重复 N 次编码为 N 个独立 token，
  线性、无合并、无拆分 → 可作为 **1:1 的相位推进单位**；
- 换行/空格/TAB 连续 N 个会被合并成 1 个 token，位移量与字符数不成线性；
- ZWNJ/ZWJ 一类 1 字符炸成 2 token 或非线性合并，粒度不可控。

因此选 ZWSP 作首选占位符，BOM 作备选，其余字符在实测中被否决。

**为什么是"零宽"而不是普通空格**：占位符必须在语义上不可见，
否则会污染用户可见的提示词内容，也会改变模型的语义理解。零宽字符属于 Unicode `Cf`（format）类，
不落在 `L*`/`M*`/`P*`/`S*`/`N*`/空白分支，预分词器既不与相邻字母合并，也不吞掉后续 `Ġ` token。

---

## 3. 三种 Token 判定模式的设计意图

"哪些 token 需要被推进"没有唯一答案，取决于调用方知道多少信息。因此提供三档，
让**精度与调用成本可权衡**：

| 模式 | 判定依据 | 设计意图 |
| --- | --- | --- |
| `auto` | 去空白与纯标点后仍非空 | 零配置兜底。调用方不想或不能提供任何线索时使用 |
| `keywords` | token 文本包含任一关键词 | 调用方明确知道要推什么（如密钥、专有名词），精度最高、干扰最小 |
| `weight` | 语义权重 ≥ 阈值 | 调用方只有模糊意图（"重要的词"），需要按重要性梯度筛选 |

### 3.1 为什么 weight 模式用这个公式

[`phase_shifting/weights.py`](phase_shifting/weights.py) 中的权重是**可插拔的近似**：

```
w = 0.5 * len_norm + 0.3 * is_word + 0.2 * rarity
```

- `len_norm`：越长的 token 通常携带越多语义；
- `is_word`：字母连续块 / CJK 块判 1.0，数字串判 0.5（数字常是编号而非语义主体）；
- `rarity`：token id 越大通常越罕见，罕见 token 更可能是信息载体。

**这是一个启发式，不是语义模型**。它的定位是"演示自定义阈值能力 + 提供无需关键词的中间档"，
不追求与真实语义重要性对齐。若需更高精度，调用方应改用 `keywords` 模式或自行判定后构造输入。

### 3.2 三种模式为什么结果差异巨大

`auto` 会把几乎所有实词判为关键 token，插入量最大、干预最强；
`keywords` 只命中显式指定的词，插入量通常最小；
`weight` 落在两者之间，且随阈值单调变化。

这个差异是**特性而非缺陷**：调用方用干预强度换取"不需要提供线索"的便利。
具体数值对比见 `demo/demo2_key_modes.py` 的实跑输出与用户文档的对比表。

### 3.3 为什么 weight 模式需要宏观窗口（Peak Window）

原始 `weight` 模式有一个结构性缺陷：**判定是孤立的微观打分**。
`score_token` 只看单个 token 自身的长度、词形与罕见度，完全没有上下文视野。
后果是：

- 一段无关背景里的普通名词（"配置""参数""结果"）与核心结论里的关键词拿到的分数量级相同；
- 于是"重要的一段话"被拆成几十个孤立高分 token 分散在全文，插入量随文本长度线性膨胀；
- 调用方真正想问的"哪一段最重要"这个问题，逐 token 判定在原理上就无法回答。

宏观窗口补的正是这块视野。做法分五步：

1. 保留 `score_tokens` 对全流做单点打分，得到基础权重数组 `W`（长度 N）；
2. 取定长窗口 `window_size`，在 `W` 上滑动，每个窗口算一个分数，得到窗口分数序列 `S`；
3. 窗口聚合方式二选一：`sum`（权重总和）或 `count`（≥ 阈值的 token 个数）；
4. 取 `S` 的最大值所在窗口为 **Peak Window**；多个同分窗口取最靠前者，保证确定性；
5. 只有落在峰窗内的 token 才可能被判为关键 token，峰窗外一律判否。

**为什么窗口长度建议取模型步长的整数倍**：本项目的一切都围绕
`index % slice_length == 0` 这个相位周期。若窗口长度是步长的整数倍，
峰窗在物理 token 流上的起始相位与推进后的相位可以对齐到同一周期网格，
"把峰窗第一个 token 推入 Phase 0"才不会把峰窗后半截甩出偏置区。
窗口长度与步长互质时仍能工作，但峰窗内后续 token 的相位分布会更散。

**为什么是"峰窗首 token"而不是"峰窗内所有权重 token"**：对齐引擎的插入是
逐 token 迭代的，每次插入都会改变其后所有 token 的下标。若把峰窗内十几个
token 全列为目标，前几次插入就会把后面目标的相位推离偏置区，导致反复重算。
锚定单个首 token 后，只需一次精确推进即可让整个峰窗起点落入 Phase 0，
后续 token 顺带获得连续的偏置相位——这与引擎"前缀零破坏"的不变量天然契合。

**为什么不引入 TF-IDF / 真实语义模型**：本项目刻意保持零外部模型依赖，
tokenizer 之外不引入任何推理组件。TF-IDF 需要语料库统计，语义模型需要额外权重文件，
都会破坏"单文件可安装、离线可跑"的定位。当前的窗口聚合是**统计密度的粗粒度代理**，
它回答的是"哪一段的高分 token 最密集"，而不是"哪一段语义最重要"。
若需后者，调用方应自行判定后构造输入，或改用 `keywords` 模式。

**为什么窗口化判定必须绕过判定缓存**：`KeyTokenSelector` 原有的
`_verdict_cache` 以 `(mode, token_id, weight_threshold)` 为键，
前提是"同一 token 的关键性与位置无关"。窗口化判定直接违反这个前提——
同一个 token 在峰窗内是关键、在峰窗外不是。因此 `select()` 在检测到
`mode="weight" and window_size > 0` 时走独立分支 `_select_windowed()`，
完全不访问缓存。这是有意为之的性能/正确性取舍：窗口化路径每轮都要重算
峰窗，代价 O(N)，但换来的是判定结果与全文结构一致。

**为什么全文短于一个窗口时退化为全流判定**：此时没有任何窗口可选，
`peak_window` 返回 `None`。若改为"报错"或"返回空结果"，短输入（一句话、
一个密钥）就连最基本的偏移都做不了，这与 `weight` 模式的存在意义相悖。
退化路径下 `peak_window` 为 `None`，调用方可据此判断"这次没有峰窗信息"。

---

## 4. 算法不变量（改动引擎前必读）

这五条是算法正确性的根基，每条都有测试守护。破坏任何一条都会导致不收敛或结果不确定。

1. **前缀零破坏**：插入后 `ids2[:t] == ids[:t]`。
   已对齐 token 的相位永不被扰动，这是迭代主循环必然收敛的根本保证。
2. **插入点紧贴目标 token**：插入位置必须是目标 token 正前方。
   改为"段落头部插入"会推动该段落内所有 token 的下标，造成"修好一个、推歪另一个"
   的级联抖动，循环无法收敛。
3. **双条件验收**：每次插入后必须真正重新 tokenize 全文，同时校验前缀零破坏 + 目标归位。
   不接受"预测新下标"的捷径——BPE 的合并行为无法离线预测。
4. **LF 不参与还原剥离**：`lf`（换行）是结构可见占位符，与原文合法换行无法区分。
   一旦纳入 `_strippable`，`restore` 会连带删掉原文换行，造成不可逆的段落破坏。
5. **幂等是两段式**：先对**原始输入**编码判定，已全部对齐就直接零干预返回；
   仍有未对齐项才剥离历史占位符、从干净原文重算。
   合并成"总是先剥离"会让第二次运行对已对齐文本也产生插入。

---

## 5. 分层与依赖方向

```
cli.py  ──▶  api.py  ──▶  aligner.py / validator.py / keytokens.py / ...
              config.py        （引擎内部实现）
              constants.py
              errors.py
primitives.py ──▶ constants.py / tokenizer_api.py
                  （函数体内惰性导入 api.py，避免环状依赖）
```

- `api.py` 是外部调用方唯一应依赖的门面，不暴露引擎内部类型；
- `cli.py` 只做"参数解析 → 调 SDK → 渲染结果"，不持有对齐算法；
- `report.py` / `diffview.py` 属于展示层，只被 CLI 调用。

**为什么 CLI 是薄壳**：CLI 与 SDK 必须共用同一引擎，否则两条路径会出现行为漂移。
CLI 只承担参数翻译与结果渲染，所有对齐逻辑都在 `api.py` 之后的层级。

---

## 6. 性能设计

每次插入后都要整篇重新 tokenize 做动态校验，基础成本是 **O(关键 token 数 × 文本长度)**：
插入次数与文本长度同时线性增长，耗时随二者乘积上升。已做的优化：

- 候选枚举顺序改为"插入点优先"（`off_delta=0` 最常见），减少无效候选的全量编码次数；
- `KeyTokenSelector` 按 `(mode, token_id, weight_threshold)` 缓存判定结论——
  同一 token 的关键性与位置无关，可跨轮次复用；
- `token_start_offsets` / `byte_to_char_map` 支持限量计算，只为目标 token 求前缀偏移；
- 段落映射仅在 `insert_mode != "inline"` 时才计算（默认模式不需要）。

**加功能时的性能守则**：不要在主循环里新增 O(文本长度) 的全量计算；能增量就增量，能限量就限量。

---

## 7. 为什么切片长度必须参数化

`slice_length` 是目标模型一次前向处理的 token 数，**不同模型取值不同**。
把 4 写死在实现里会让 SDK 只对单一模型有效，因此：

- 唯一允许出现该字面量的位置是 `phase_shifting/constants.py` 的 `DEFAULT_SLICE_LENGTH`；
- 公共 API 一律以 `slice_length` 暴露，接受任意正整数；
- 该约束由 `tests/test_packaging.py` 的 AST 扫描守护。

---

## 8. 异常体系

[`phase_shifting/errors.py`](phase_shifting/errors.py) 采用**多继承**设计，
让调用方既能按项目语义捕获，也能按 Python 内置类型捕获：

- `ConfigurationError(PhaseShiftingError, ValueError)` — 配置不合法；
- `UnknownFillerError(ConfigurationError, KeyError)` — 引用了不存在的占位符名；
- `TokenizerError(PhaseShiftingError, RuntimeError)` — tokenizer 不可用；
- `AlignmentError(PhaseShiftingError, RuntimeError)` — 对齐过程错误；
- `AnchorLostError(AlignmentError)` — 插入后目标 token 锚点丢失；
- `RetryExhaustedError(AlignmentError)` — 所有候选均未通过校验。

调用方可选 `except PhaseShiftingError`（统一处理）或 `except ValueError`（融入既有代码）。

---

## 9. 源码卫生约束

源码会随 `pip install` 分发给外部使用者，因此注释、docstring、报错信息里**不能**出现：

- 绝对盘符路径、某用户的家目录；
- 某个具体解释器的相对路径（如 `.venv/Scripts/python.exe`）；
- 某个开发者机器的目录名或用户名。

报错信息要给出跨平台通用的修复指引。该约束由
`tests/test_packaging.py::test_no_machine_specific_paths_in_source` 守护。

---

## 10. 文档分工

| 文档 | 回答什么 | 面向 |
| --- | --- | --- |
| [`README.md`](README.md) | 是什么、怎么跑 | 所有访客 |
| [使用教程（GitHub Wiki）](https://github.com/FactLab01/Prompt_Phase_Shifting/wiki/使用教程) | 怎么装、怎么调、怎么跑 | 使用者 |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | 为什么这么设计 | 改动者 |
| [`docs/SPEC.md`](docs/SPEC.md) | 实现规范与实测数据 | 两者 |
| [`CHANGELOG.md`](CHANGELOG.md) | 每个版本改了什么 | 使用者 |

**代码注释只回答"怎么用"**：这段代码做什么、参数含义、返回值、典型用法。
设计思路、架构理念、实现原因一律写进本文件，不散落在注释里。
