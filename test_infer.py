#!/usr/bin/env python3
"""Unit tests for infer — tests logic without hitting Ollama."""

import json, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import importlib.machinery, importlib.util
_infer_path = Path(__file__).with_name("infer")
_loader = importlib.machinery.SourceFileLoader("infer", str(_infer_path))
_spec = importlib.util.spec_from_loader("infer", _loader)
infer_mod = importlib.util.module_from_spec(_spec)
sys.modules["infer"] = infer_mod
_loader.exec_module(infer_mod)


class TestValidateShape(unittest.TestCase):
    def test_string_match(self):
        self.assertIsNone(infer_mod.validate_shape("hello", ""))

    def test_string_mismatch(self):
        self.assertIsNotNone(infer_mod.validate_shape(123, ""))

    def test_number_match(self):
        self.assertIsNone(infer_mod.validate_shape(42, 0))
        self.assertIsNone(infer_mod.validate_shape(3.14, 0.0))

    def test_number_mismatch(self):
        self.assertIsNotNone(infer_mod.validate_shape("42", 0))

    def test_object_match(self):
        self.assertIsNone(infer_mod.validate_shape({"name": "alice", "age": 30}, {"name": "", "age": 0}))

    def test_object_missing_key(self):
        err = infer_mod.validate_shape({"name": "alice"}, {"name": "", "age": 0})
        self.assertIn("age", err)

    def test_object_wrong_type(self):
        err = infer_mod.validate_shape({"name": 123}, {"name": ""})
        self.assertIn("name", err)

    def test_array_match(self):
        self.assertIsNone(infer_mod.validate_shape(["a", "b", "c"], [""]))

    def test_array_mismatch(self):
        err = infer_mod.validate_shape(["a", 2, "c"], [""])
        self.assertIsNotNone(err)

    def test_array_not_array(self):
        err = infer_mod.validate_shape("not an array", [""])
        self.assertIn("expected array", err)

    def test_nested_object(self):
        shape = {"user": {"name": "", "score": 0}}
        self.assertIsNone(infer_mod.validate_shape({"user": {"name": "bob", "score": 99}}, shape))
        err = infer_mod.validate_shape({"user": {"name": "bob"}}, shape)
        self.assertIn("score", err)


class TestLoadRole(unittest.TestCase):
    def test_load_existing_role(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            roles_dir = Path(tmpdir)
            role_file = roles_dir / "coder.md"
            role_file.write_text("You are a coder.")
            with patch.object(infer_mod, "GLOBAL_ROLES", roles_dir):
                result = infer_mod.load_role("coder")
            self.assertEqual(result, "You are a coder.")

    def test_missing_role_exits(self):
        with patch.object(infer_mod, "GLOBAL_ROLES", Path("/nonexistent")):
            with self.assertRaises(SystemExit):
                infer_mod.load_role("ghost")


class TestLoadConfig(unittest.TestCase):
    def test_defaults(self):
        with patch.object(infer_mod, "GLOBAL_CONFIG", Path("/nonexistent")), \
             patch.object(infer_mod, "GLOBAL_SYSTEM", Path("/nonexistent")), \
             patch.object(infer_mod, "LOCAL_CONFIG",  Path("/nonexistent")), \
             patch.object(infer_mod, "LOCAL_SYSTEM",  Path("/nonexistent")):
            cfg = infer_mod.load_config()
        self.assertEqual(cfg["url"], infer_mod.DEFAULTS["url"])
        self.assertEqual(cfg["model"], infer_mod.DEFAULTS["model"])
        self.assertEqual(cfg["system"], infer_mod.DEFAULT_SYSTEM)

    def test_global_config_override(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model": "qwen3:latest"}, f)
            config_path = Path(f.name)
        with patch.object(infer_mod, "GLOBAL_CONFIG", config_path), \
             patch.object(infer_mod, "GLOBAL_SYSTEM", Path("/nonexistent")), \
             patch.object(infer_mod, "LOCAL_CONFIG",  Path("/nonexistent")), \
             patch.object(infer_mod, "LOCAL_SYSTEM",  Path("/nonexistent")):
            cfg = infer_mod.load_config()
        self.assertEqual(cfg["model"], "qwen3:latest")
        config_path.unlink()

    def test_system_prompt_layering(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("global instructions")
            global_path = Path(f.name)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("local instructions")
            local_path = Path(f.name)
        with patch.object(infer_mod, "GLOBAL_CONFIG", Path("/nonexistent")), \
             patch.object(infer_mod, "GLOBAL_SYSTEM", global_path), \
             patch.object(infer_mod, "LOCAL_CONFIG",  Path("/nonexistent")), \
             patch.object(infer_mod, "LOCAL_SYSTEM",  local_path):
            cfg = infer_mod.load_config()
        self.assertIn("global instructions", cfg["system"])
        self.assertIn("local instructions", cfg["system"])
        global_path.unlink()
        local_path.unlink()


class TestRun(unittest.TestCase):
    def _make_response(self, content=None, tool_calls=None):
        msg = MagicMock()
        msg.content = content or ""
        msg.tool_calls = tool_calls or []
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage.completion_tokens = 10
        resp.usage.prompt_tokens = 20
        return resp

    def _make_tool_call(self, cmd):
        call = MagicMock()
        call.id = "call_test"
        call.function.name = "bash"
        call.function.arguments = json.dumps({"command": cmd})
        return call

    @patch("infer.subprocess.run")
    @patch("infer.OpenAI")
    def test_simple_response(self, mock_client_cls, mock_subproc):
        client = MagicMock()
        mock_client_cls.return_value = client
        client.chat.completions.create.return_value = self._make_response(content="Paris")
        with patch("builtins.print") as mock_print:
            code = infer_mod.run("capital of France", url="http://x", model="m", api_key="x", system="s", verbose=False)
        self.assertEqual(code, 0)
        mock_print.assert_called_with("Paris")

    @patch("infer.subprocess.run")
    @patch("infer.OpenAI")
    def test_tool_call_then_answer(self, mock_client_cls, mock_subproc):
        client = MagicMock()
        mock_client_cls.return_value = client
        mock_subproc.return_value = MagicMock(stdout="today\n", stderr="")
        client.chat.completions.create.side_effect = [
            self._make_response(tool_calls=[self._make_tool_call("date")]),
            self._make_response(content="today"),
        ]
        with patch("builtins.print") as mock_print:
            code = infer_mod.run("what day is it", url="http://x", model="m", api_key="x", system="s", verbose=False)
        self.assertEqual(code, 0)
        mock_subproc.assert_called_once()

    @patch("infer.subprocess.run")
    @patch("infer.OpenAI")
    def test_json_valid(self, mock_client_cls, mock_subproc):
        client = MagicMock()
        mock_client_cls.return_value = client
        client.chat.completions.create.return_value = self._make_response(content='{"name": "alice"}')
        with patch("builtins.print") as mock_print:
            code = infer_mod.run("get user", url="http://x", model="m", api_key="x", system="s", verbose=False, json_mode=True)
        self.assertEqual(code, 0)

    @patch("infer.subprocess.run")
    @patch("infer.OpenAI")
    def test_json_invalid_retries(self, mock_client_cls, mock_subproc):
        client = MagicMock()
        mock_client_cls.return_value = client
        client.chat.completions.create.side_effect = [
            self._make_response(content="not json"),
            self._make_response(content='{"fixed": true}'),
        ]
        with patch("builtins.print"), patch("sys.stderr"):
            code = infer_mod.run("get data", url="http://x", model="m", api_key="x", system="s", verbose=False, json_mode=True)
        self.assertEqual(code, 0)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    @patch("infer.subprocess.run")
    @patch("infer.OpenAI")
    def test_shape_mismatch_retries(self, mock_client_cls, mock_subproc):
        client = MagicMock()
        mock_client_cls.return_value = client
        client.chat.completions.create.side_effect = [
            self._make_response(content='{"name": 123}'),
            self._make_response(content='{"name": "alice"}'),
        ]
        with patch("builtins.print"), patch("sys.stderr"):
            code = infer_mod.run("get user", url="http://x", model="m", api_key="x", system="s", verbose=False,
                                 json_mode='{"name": ""}')
        self.assertEqual(code, 0)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    @patch("infer.subprocess.run")
    @patch("infer.OpenAI")
    def test_max_steps_returns_1(self, mock_client_cls, mock_subproc):
        client = MagicMock()
        mock_client_cls.return_value = client
        mock_subproc.return_value = MagicMock(stdout="out", stderr="")
        client.chat.completions.create.return_value = self._make_response(tool_calls=[self._make_tool_call("echo hi")])
        with patch("sys.stderr"):
            code = infer_mod.run("loop forever", url="http://x", model="m", api_key="x", system="s", verbose=False, max_steps=3)
        self.assertEqual(code, 1)
        self.assertEqual(client.chat.completions.create.call_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
