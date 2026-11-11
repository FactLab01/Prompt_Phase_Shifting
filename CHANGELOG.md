# CHANGELOG

本项目的版本变更记录，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

## [1.0.0] - 2026-11-11

首个发布版：以可直接安装、导入、调用的 Python SDK 形式交付"前置相位偏移"能力。

### Features

- SDK 门面 `PhaseShifter` 与函数式入口 `align_text` / `plan_text` / `restore_text`，
  配置对象 `PhaseShiftConfig` 支持 `to_dict` / `from_dict` 无损往返。
- 三种关键 token 判定模式：`auto` / `keywords` / `weight`，后者支持叠加宏观窗口
  （`window_size` / `window_agg`）定位 Peak Window。
- 每次插入后立即重新 tokenize 做双条件动态校验（前缀零破坏 + 目标归位），
  失败时自动切换插入点、数量与占位种类重试。
- 占位符注册表在运行时实测每个候选的 token 成本，与声明不符自动校正。
- CLI 前端 `python -m phase_shifting`（安装后为 `phase-shifting`）：文本 / 表格 /
  JSON 报告、可读差异视图、`restore` 子命令。
- 面向未来 C-ABI 绑定的 `phase_shifting.primitives` 原语层（仅位置参数、仅基础类型）。
- `slice_length` 作为显式参数暴露，默认取 `DEFAULT_SLICE_LENGTH`，不写死在实现里。
- `py.typed` 随包分发（PEP 561）。
