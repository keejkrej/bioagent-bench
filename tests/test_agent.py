import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from bioagent_bench.agent import execute_tool, run_agent


class WorkspaceToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name) / "workspace"
        self.workspace.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_write_read_and_list_round_trip(self):
        written = execute_tool(self.workspace, "write_file", {"path": "pipeline/predict.sh", "content": "echo ok\n"})
        self.assertTrue(written["ok"])
        read = execute_tool(self.workspace, "read_file", {"path": "pipeline/predict.sh"})
        self.assertIn("1: echo ok", read["content"])
        listed = execute_tool(self.workspace, "list_files", {"path": "."})
        self.assertEqual([item["path"] for item in listed["files"]], ["pipeline/predict.sh"])

    def test_absolute_and_parent_paths_are_rejected(self):
        for path in ("/etc/passwd", "../outside", "folder/../../outside"):
            with self.subTest(path=path):
                result = execute_tool(self.workspace, "write_file", {"path": path, "content": "bad"})
                self.assertFalse(result["ok"])
                self.assertIn("Path", result["error"])

    def test_command_does_not_receive_api_credentials(self):
        with patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "secret-value", "MY_TEST_TOKEN": "also-secret"}):
            result = execute_tool(self.workspace, "run_command", {"command": "python3 -c 'import os; print(os.getenv(\"AI_GATEWAY_API_KEY\"), os.getenv(\"MY_TEST_TOKEN\"))'"})
        self.assertTrue(result["ok"], result)
        self.assertIn("None None", result["stdout_tail"])
        self.assertNotIn("secret-value", json.dumps(result))
        self.assertNotIn("also-secret", json.dumps(result))

    def test_command_uses_virtualenv_python(self):
        result = execute_tool(self.workspace, "run_command", {"command": "python3 -c 'import numpy, pandas, scipy, sklearn; print(\"scientific stack ok\")'"})
        self.assertTrue(result["ok"], result)
        self.assertIn("scientific stack ok", result["stdout_tail"])

    def test_environment_is_allowlisted(self):
        injected = {
            "DATABASE_URL": "postgres://user:secret@example/db",
            "SSH_AUTH_SOCK": "/private/secret-agent.sock",
            "ALL_PROXY": "http://user:secret@proxy.example",
            "BASH_ENV": "/private/secret-startup.sh",
            "ENV": "/private/secret-startup.sh",
        }
        with patch.dict(os.environ, injected):
            result = execute_tool(self.workspace, "run_command", {"command": "python3 -c 'import os; print(*(os.getenv(k) for k in (\"DATABASE_URL\", \"SSH_AUTH_SOCK\", \"BASH_ENV\", \"ENV\")))'"})
        self.assertTrue(result["ok"], result)
        self.assertIn("None None None None", result["stdout_tail"])
        for secret in injected.values():
            self.assertNotIn(secret, json.dumps(result))

    def test_command_timeout_is_reported(self):
        result = execute_tool(self.workspace, "run_command", {"command": "sleep 10", "timeout_seconds": 1})
        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])

    @patch("bioagent_bench.agent.GatewayClient")
    def test_provider_refusal_is_recorded(self, client_class):
        client_class.return_value.chat.return_value = {
            "choices": [{"finish_reason": "content_filter", "message": {"role": "assistant", "content": None, "refusal": "blocked"}}],
            "usage": {"total_tokens": 2},
        }
        result = run_agent(self.workspace, Path(self.temp.name) / "logs", "test task", max_steps=3)
        self.assertEqual(result["status"], "provider_blocked")
        self.assertEqual(client_class.return_value.chat.call_count, 1)
        self.assertEqual(result["steps"], 1)

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_credentials_become_a_run_result(self):
        log_dir = Path(self.temp.name) / "missing-key-logs"
        result = run_agent(self.workspace, log_dir, "test task")
        self.assertEqual(result["status"], "agent_error")
        self.assertIn("AI_GATEWAY_API_KEY", result["error"])
        self.assertEqual(json.loads((log_dir / "agent_result.json").read_text())["status"], "agent_error")

    @patch("bioagent_bench.agent.GatewayClient")
    def test_expired_deadline_does_not_call_model(self, client_class):
        result = run_agent(self.workspace, Path(self.temp.name) / "deadline-logs", "test task", time_limit_seconds=0)
        self.assertEqual(result["status"], "time_limit")
        self.assertEqual(result["steps"], 0)
        client_class.assert_not_called()

    @patch("bioagent_bench.agent.GatewayClient")
    def test_deadline_caps_tool_timeout(self, client_class):
        client_class.return_value.chat.return_value = {
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "type": "function", "function": {
                    "name": "run_command",
                    "arguments": json.dumps({"command": "sleep 10", "timeout_seconds": 600}),
                }}],
            }}],
            "usage": {},
        }
        started = time.monotonic()
        result = run_agent(self.workspace, Path(self.temp.name) / "tool-deadline-logs", "test task", max_steps=3, time_limit_seconds=2)
        self.assertEqual(result["status"], "time_limit")
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(client_class.return_value.chat.call_count, 1)


if __name__ == "__main__":
    unittest.main()
