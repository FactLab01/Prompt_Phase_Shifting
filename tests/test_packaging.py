# -*- coding: utf-8 -*-
"""打包与架构守护测试。

覆盖三类不可回归的约束：

1. **切片长度不得硬编码**：``4`` 只允许出现在常量定义处，SDK 其余位置必须
   通过 ``DEFAULT_SLICE_LENGTH`` 或显式参数引用；
2. **CLI 不得承载业务逻辑**：CLI 只做参数解析与渲染；
3. **打包元数据完整**：``py.typed`` 存在、版本号一致、公共 API 可导入。
"""

from __future__ import annotations

import ast
import os
import re
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "phase_shifting"
CONSTANTS_FILE = PACKAGE_ROOT / "constants.py"

# 允许出现字面量 4 的文件（常量唯一定义处）
ALLOWED_LITERAL_FILES = {CONSTANTS_FILE.name}

# 与切片长度语义无关、可安全出现字面量 4 的正则场景（例如 Unicode 转义、字节数）
_ALLOWED_PATTERNS = (
    re.compile(r"\\u[0-9a-fA-F]{4}"),  # Unicode 转义
    re.compile(r"\\x[0-9a-fA-F]{2}"),  # 字节转义
)


def _iter_source_files():
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        yield path


class NoHardcodedSliceLengthTest(unittest.TestCase):
    """切片长度字面量必须收敛到常量定义处。"""

    def test_constants_file_declares_default(self):
        source = CONSTANTS_FILE.read_text(encoding="utf-8")
        self.assertIn("DEFAULT_SLICE_LENGTH", source)
        self.assertRegex(source, r"DEFAULT_SLICE_LENGTH\s*:\s*int\s*=\s*4\b")

    def test_no_bare_literal_in_other_modules(self):
        """除常量定义处外，不得出现与切片长度同值的裸字面量 4。

        判据是 AST 级的"值"，因此具名常量（如 ``_ROUND_FACTOR = 4``）也会被
        计入 —— 这些常量的存在正是为了让 4 不出现在表达式里，故允许其定义行。
        """
        offenders = []
        for path in _iter_source_files():
            if path.name in ALLOWED_LITERAL_FILES:
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            # 收集"模块级/类级具名常量的定义行"，这些行允许出现字面量
            definition_lines = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        node.targets
                        if isinstance(node, ast.Assign)
                        else ([node.target] if node.target else [])
                    )
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            definition_lines.add(node.lineno)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant):
                    continue
                if isinstance(node.value, bool) or node.value != 4:
                    continue
                if node.lineno in definition_lines:
                    continue
                offenders.append((path.name, node.lineno))
        self.assertEqual(
            offenders,
            [],
            "以下位置出现字面量 4，请改用 DEFAULT_SLICE_LENGTH 或具名常量：%r"
            % (offenders,),
        )

    def test_no_period_default_literal_in_signatures(self):
        """公共与内部 API 的默认值必须来自常量，而非字面量。"""
        for path in _iter_source_files():
            source = path.read_text(encoding="utf-8")
            for match in re.finditer(
                r"(period|slice_length)\s*:\s*int\s*=\s*(\d+)", source
            ):
                value = int(match.group(2))
                self.assertEqual(
                    value,
                    4,
                    "%s 中 %s 的默认值不是常量推导值（%d）"
                    % (path.name, match.group(1), value),
                )

    def test_no_modulo_four_expression(self):
        for path in _iter_source_files():
            if path.name in ALLOWED_LITERAL_FILES:
                continue
            source = path.read_text(encoding="utf-8")
            self.assertNotRegex(
                source,
                r"%\s*4\b",
                "%s 中出现 %% 4，应使用 %% slice_length" % path.name,
            )

    def test_default_slice_length_is_four(self):
        from phase_shifting.constants import DEFAULT_SLICE_LENGTH

        self.assertEqual(DEFAULT_SLICE_LENGTH, 4)

    def test_constant_is_single_source_of_truth(self):
        """常量文件中 DEFAULT_SLICE_LENGTH 只能被赋值一次。"""
        source = CONSTANTS_FILE.read_text(encoding="utf-8")
        assignments = re.findall(
            r"^DEFAULT_SLICE_LENGTH\s*(?::\s*int\s*)?=", source, re.MULTILINE
        )
        self.assertEqual(len(assignments), 1)


class PublicApiTypingTest(unittest.TestCase):
    """公共 API 必须完全类型标注并带文档字符串。"""

    PUBLIC_MODULES = ("api.py", "config.py", "constants.py", "errors.py")

    def test_public_functions_annotated_and_documented(self):
        for name in self.PUBLIC_MODULES:
            path = PACKAGE_ROOT / name
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("_"):
                        continue
                    self.assertIsNot(
                        node.returns,
                        None,
                        "%s:%d %s 缺少返回标注" % (name, node.lineno, node.name),
                    )
                    self.assertTrue(
                        ast.get_docstring(node),
                        "%s:%d %s 缺少文档字符串" % (name, node.lineno, node.name),
                    )

    def test_public_classes_documented(self):
        for name in self.PUBLIC_MODULES:
            path = PACKAGE_ROOT / name
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if node.name.startswith("_"):
                        continue
                    self.assertTrue(
                        ast.get_docstring(node),
                        "%s:%d 类 %s 缺少文档字符串"
                        % (name, node.lineno, node.name),
                    )

    def test_slice_length_exposed_in_public_api(self):
        import phase_shifting

        self.assertIn("PhaseShiftConfig", phase_shifting.__all__)
        self.assertIn("DEFAULT_SLICE_LENGTH", phase_shifting.__all__)


class PackagingMetadataTest(unittest.TestCase):
    """打包元数据必须支持 PEP 561 与可安装性。"""

    def test_py_typed_marker_exists(self):
        marker = PACKAGE_ROOT / "py.typed"
        self.assertTrue(marker.exists(), "缺少 PEP 561 标记文件 py.typed")

    def test_version_matches_pyproject(self):
        import phase_shifting

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        self.assertTrue(pyproject.exists(), "缺少 pyproject.toml")
        text = pyproject.read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        self.assertIsNotNone(match, "pyproject.toml 未声明 version")
        self.assertEqual(match.group(1), phase_shifting.__version__)

    def test_pyproject_declares_build_system(self):
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        self.assertIn("[build-system]", text)
        self.assertIn("setuptools", text)

    def test_pyproject_declares_dependency(self):
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        self.assertIn("deepseek-tokenizer", text)

    def test_pyproject_declares_console_script(self):
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        self.assertIn("[project.scripts]", text)
        self.assertIn("phase_shifting.cli:main", text)

    def test_package_data_includes_py_typed(self):
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        self.assertIn("py.typed", text)

    def test_main_entry_point_callable(self):
        from phase_shifting.cli import main

        self.assertTrue(callable(main))

    def test_module_entry_point_exists(self):
        self.assertTrue((PACKAGE_ROOT / "__main__.py").exists())


class CliThinShellTest(unittest.TestCase):
    """CLI 必须是薄壳：不含对齐算法。"""

    CLI_FILE = PACKAGE_ROOT / "cli.py"

    def test_cli_does_not_import_engine_internals(self):
        source = self.CLI_FILE.read_text(encoding="utf-8")
        # 报告渲染与 diff 视图属于展示层，允许；引擎配置/对齐器不允许
        self.assertNotIn("from .aligner import AlignConfig", source)
        self.assertNotIn("from .aligner import Aligner", source)
        self.assertNotIn("from .keytokens import KeyTokenSelector", source)

    def test_cli_delegates_to_sdk(self):
        source = self.CLI_FILE.read_text(encoding="utf-8")
        self.assertIn("PhaseShifter", source)
        self.assertIn(".align(", source)

    def test_cli_exposes_slice_length_flag(self):
        from phase_shifting.cli import build_parser

        args = build_parser().parse_args(["--slice-length", "8"])
        self.assertEqual(args.slice_length, 8)

    def test_cli_slice_length_default_from_constant(self):
        from phase_shifting.cli import build_parser
        from phase_shifting.constants import DEFAULT_SLICE_LENGTH

        args = build_parser().parse_args([])
        self.assertEqual(args.slice_length, DEFAULT_SLICE_LENGTH)

    def test_cli_accepts_short_flag(self):
        from phase_shifting.cli import build_parser

        args = build_parser().parse_args(["-p", "3"])
        self.assertEqual(args.slice_length, 3)


class SourceHygieneTest(unittest.TestCase):
    """源码卫生：无调试残留、无越界路径引用。"""

    def test_no_debug_prints_outside_cli(self):
        # print 只允许出现在 CLI 前端（它的职责就是往 stdout 写报告）；
        # SDK / 引擎层必须保持无副作用。文档字符串中的用法示例不计入。
        for path in _iter_source_files():
            if path.name == "cli.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name) and node.func.id == "print":
                    self.fail(
                        "%s:%d 不应有 print 调试残留" % (path.name, node.lineno)
                    )

    def test_cli_does_not_dump_objects(self):
        """CLI 的 print 只用于报告与错误，不得 dump 任意对象。"""
        source = (PACKAGE_ROOT / "cli.py").read_text(encoding="utf-8")
        self.assertNotIn("print(repr(", source)
        self.assertNotIn("print(vars(", source)
        self.assertNotIn("print(dir(", source)

    def test_no_tmp_path_references_in_package(self):
        for path in _iter_source_files():
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("/tmp/", source)
            self.assertNotRegex(source, r"(?<!\.)\btmp/", "%s 引用了 /tmp" % path.name)

    def test_no_global_environment_pollution(self):
        """不得出现直接执行的 pip install / 破坏性删除指令。"""
        for path in _iter_source_files():
            source = path.read_text(encoding="utf-8")
            self.assertNotRegex(source, r"rm\s+-rf", source)
            # pip install 只允许出现在两类文案里：
            #   1) 运行时依赖缺失的安装指引；
            #   2) 安装本项目自身的示例（pip install . / pip install -e .）。
            for match in re.finditer(r"pip install", source):
                line_start = source.rfind("\n", 0, match.start()) + 1
                line_end = source.find("\n", match.start())
                line = source[line_start : line_end if line_end > 0 else len(source)]
                self.assertRegex(
                    line,
                    r"python -m pip install|deepseek-tokenizer|pip install (-e )?\.",
                    "%s 中出现未受约束的 pip install：%s" % (path.name, line.strip()),
                )

    def test_no_machine_specific_paths_in_source(self):
        """源码（随包分发）不得夹带本地开发环境痕迹。

        外部使用者会直接安装这个包，因此源码注释、docstring、报错信息里
        不能出现绝对盘符路径、某用户的家目录、或特定解释器的相对路径。
        """
        offenders = []
        for path in _iter_source_files():
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                for pattern, label in (
                    (r"[A-Za-z]:[\\/]", "绝对盘符路径"),
                    (r"(?<![\w.])/Users/", "家目录路径"),
                    (r"(?<![\w.])/home/", "家目录路径"),
                    (r"\.venv[\\/]Scripts", "特定解释器相对路径"),
                ):
                    if re.search(pattern, line):
                        offenders.append((path.name, lineno, label, line.strip()[:80]))
        self.assertEqual(
            offenders,
            [],
            "源码中不得出现本地环境痕迹（请改用相对路径或跨平台示例）：%r" % (offenders,),
        )


if __name__ == "__main__":
    unittest.main()
