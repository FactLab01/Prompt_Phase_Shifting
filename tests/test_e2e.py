# -*- coding: utf-8 -*-
"""CLI 端到端测试：文件输入、stdin、--text、--diff、幂等性。"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

from phase_shifting.cli import build_parser, main, read_input

SAMPLE = "Hello world 关键信息 第二段落内容"


class ReadInputTest(unittest.TestCase):
    def test_text_arg(self):
        args = build_parser().parse_args(["--text", "abc"])
        text, source = read_input(args)
        self.assertEqual(text, "abc")
        self.assertEqual(source, "cli-arg")

    def test_positional_arg(self):
        args = build_parser().parse_args(["hello there"])
        text, source = read_input(args)
        self.assertEqual(text, "hello there")
        self.assertEqual(source, "cli-positional")

    def test_file_arg(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8", newline=""
        ) as fh:
            fh.write("文件内容 test")
            path = fh.name
        try:
            args = build_parser().parse_args(["--file", path])
            text, source = read_input(args)
            self.assertEqual(text, "文件内容 test")
            self.assertEqual(source, "file:%s" % path)
        finally:
            os.unlink(path)

    def test_file_arg_with_bom(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8-sig", newline=""
        ) as fh:
            fh.write("bom content")
            path = fh.name
        try:
            args = build_parser().parse_args(["--file", path])
            text, _ = read_input(args)
            self.assertEqual(text, "bom content")
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        args = build_parser().parse_args(["--file", "no_such_file_xyz.txt"])
        with self.assertRaises(FileNotFoundError):
            read_input(args)

    def test_empty_when_nothing(self):
        # 无 stdin 数据时返回空
        args = build_parser().parse_args([])
        old = sys.stdin
        sys.stdin = io.StringIO("")
        try:
            text, source = read_input(args)
            self.assertEqual(text, "")
            self.assertEqual(source, "empty")
        finally:
            sys.stdin = old


class CliMainTest(unittest.TestCase):
    def _run(self, argv, stdin_text=None):
        buf = io.StringIO()
        old_stdin = sys.stdin
        old_argv = sys.argv
        if stdin_text is not None:
            sys.stdin = io.StringIO(stdin_text)
        sys.argv = ["phase_shifting"] + argv
        try:
            with redirect_stdout(buf):
                code = main(argv)
        finally:
            sys.stdin = old_stdin
            sys.argv = old_argv
        return code, buf.getvalue()

    def test_text_arg_report(self):
        code, out = self._run(["--text", SAMPLE, "--quiet", "--log-file", None])
        self.assertEqual(code, 0)
        self.assertIn("ALIGNED", out)
        self.assertIn("U+200B", out)

    def test_table_mode(self):
        code, out = self._run(["--text", SAMPLE, "--quiet", "--table", "--log-file", None])
        self.assertEqual(code, 0)
        self.assertIn("before", out)

    def test_json_mode(self):
        code, out = self._run(["--text", SAMPLE, "--quiet", "--json", "--log-file", None])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["status"], "ALIGNED")

    def test_stdin_mode(self):
        code, out = self._run(["-", "--quiet", "--log-file", None], stdin_text=SAMPLE)
        self.assertEqual(code, 0)
        self.assertIn("ALIGNED", out)

    def test_diff_mode(self):
        code, out = self._run(["--text", SAMPLE, "--quiet", "--diff", "--log-file", None])
        self.assertEqual(code, 0)
        self.assertIn("差异视图", out)

    def test_side_by_side_mode(self):
        code, out = self._run(
            ["--text", SAMPLE, "--quiet", "--side-by-side", "--log-file", None]
        )
        self.assertEqual(code, 0)
        self.assertIn("并排视图", out)

    def test_list_fillers(self):
        code, out = self._run(["--list-fillers", "--log-file", None])
        self.assertEqual(code, 0)
        self.assertIn("zwsp", out)
        self.assertIn("U+200B", out)

    def test_file_input_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.txt")
            dst = os.path.join(tmp, "out.txt")
            with open(src, "w", encoding="utf-8", newline="") as fh:
                fh.write(SAMPLE)
            code, _ = self._run(
                ["--file", src, "--output", dst, "--quiet", "--log-file", None]
            )
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(dst))
            with open(dst, "r", encoding="utf-8", newline="") as fh:
                aligned = fh.read()
            self.assertIn("​", aligned)

    def test_missing_file_exit_code(self):
        code, out = self._run(["--file", "nope_xyz.txt", "--quiet"])
        self.assertEqual(code, 3)

    def test_bad_period_exit_code(self):
        code, out = self._run(["--text", "abc", "--period", "0"])
        self.assertEqual(code, 3)

    def test_keywords_mode(self):
        code, out = self._run(
            ["--text", SAMPLE, "--quiet", "--key-mode", "keywords",
             "--keywords", "Hello,关键", "--log-file", None]
        )
        self.assertEqual(code, 0)
        self.assertIn("ALIGNED", out)

    def test_weight_mode(self):
        code, out = self._run(
            ["--text", SAMPLE, "--quiet", "--key-mode", "weight",
             "--weight-threshold", "0.4", "--log-file", None]
        )
        self.assertEqual(code, 0)

    def test_empty_text(self):
        code, out = self._run(["--text", "", "--quiet", "--log-file", None])
        self.assertEqual(code, 0)
        self.assertIn("ALIGNED", out)

    def test_verbose(self):
        code, out = self._run(["--text", SAMPLE, "--quiet", "--verbose", "--log-file", None])
        self.assertEqual(code, 0)


class CliIdempotencyTest(unittest.TestCase):
    def _run(self, argv, stdin_text=None):
        buf = io.StringIO()
        old_stdin = sys.stdin
        if stdin_text is not None:
            sys.stdin = io.StringIO(stdin_text)
        try:
            with redirect_stdout(buf):
                code = main(argv)
        finally:
            sys.stdin = old_stdin
        return code, buf.getvalue()

    def test_double_run_same_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.txt")
            o1 = os.path.join(tmp, "o1.txt")
            o2 = os.path.join(tmp, "o2.txt")
            with open(src, "w", encoding="utf-8", newline="") as fh:
                fh.write(SAMPLE)
            self._run(["--file", src, "--output", o1, "--quiet", "--log-file", None])
            self._run(["--file", o1, "--output", o2, "--quiet", "--log-file", None])
            with open(o1, "r", encoding="utf-8", newline="") as fh:
                first = fh.read()
            with open(o2, "r", encoding="utf-8", newline="") as fh:
                second = fh.read()
            # 幂等：第二次运行结果与第一次完全一致
            self.assertEqual(first, second)


class RestoreSubcommandTest(unittest.TestCase):
    def test_restore_from_stdin(self):
        from phase_shifting.aligner import AlignConfig, Aligner
        from phase_shifting.fillers import FillerRegistry
        from phase_shifting.keytokens import KeyTokenSelector
        from phase_shifting.tokenizer_api import get_tokenizer

        tk = get_tokenizer()
        reg = FillerRegistry(tk)
        al = Aligner(
            tk,
            KeyTokenSelector(tokenizer=tk, filler_chars=[reg.get(n).char for n in reg.names()]),
            reg,
            AlignConfig(),
        )
        out = al.align(SAMPLE)

        buf = io.StringIO()
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        sys.stdin = io.StringIO(out.aligned_text)
        try:
            with redirect_stdout(buf):
                code = main(["restore", "--quiet"])
        finally:
            sys.stdin = old_stdin
        self.assertEqual(code, 0)
        restored = buf.getvalue()
        self.assertNotIn("​", restored)
        self.assertEqual(restored, SAMPLE)


if __name__ == "__main__":
    unittest.main()
