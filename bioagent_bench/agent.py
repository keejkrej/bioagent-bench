"""A small, auditable tool-using agent for one benchmark task."""

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .gateway import DEFAULT_MODEL, GatewayClient


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files below a workspace directory, with sizes.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a bounded range of lines from a UTF-8 workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1, "default": 1},
                    "max_lines": {"type": "integer", "minimum": 1, "maximum": 500, "default": 250},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or replace a UTF-8 workspace file (maximum 300000 characters).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a Bash command in the workspace. No credentials are provided; do not use the network.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "default": "."},
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600, "default": 120},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_PROMPT = """You are an autonomous computational-biology ML engineer being evaluated.
Work only inside the supplied workspace. Do not use the network, ask a user questions, or try to locate hidden labels.
Read BRIEF.md and inspect the public data before choosing a method. Build a valid, deterministic pipeline/predict.sh with exactly the interface in the brief. You may use Python's standard library and the installed numpy, pandas, scipy, and scikit-learn packages. Keep runtime and memory modest.
Use validation that respects the disclosed train/test grouping or temporal shift. Do not overfit the unlabeled development input. Write clear, portable code and make the script independent of your interactive shell state.
Test your complete command locally, inspect its JSON output, and run it twice to check determinism before finishing. The grader will copy only pipeline/ and run it in fresh directories on sealed input; therefore everything needed at prediction time must live under pipeline/. Do not rely on files elsewhere in the workspace except INPUT_DIR files supplied by the contract.
A short final report of your method and validation is welcome, but the required deliverable is the working pipeline."""


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _safe_path(workspace: Path, relative: str) -> Path:
    if os.path.isabs(relative):
        raise ValueError("Path must be relative; absolute paths are not allowed")
    root = workspace.resolve()
    candidate = (root / relative).resolve()
    if os.path.commonpath([str(root), str(candidate)]) != str(root):
        raise ValueError("Path leaves the task workspace")
    return candidate


def _clean_environment(workspace: Path) -> Dict[str, str]:
    """Use a small allowlist: inherited variables can contain live credentials."""
    home = workspace / ".home"
    temporary = workspace / ".tmp"
    home.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env[key] = "http://127.0.0.1:9"
    env["NO_PROXY"] = env["no_proxy"] = ""
    return env


def _tail(text: str, limit: int = 12000) -> str:
    return text if len(text) <= limit else "...[truncated]...\n" + text[-limit:]


def execute_tool(workspace: Path, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if name == "list_files":
            base = _safe_path(workspace, arguments.get("path") or ".")
            if not base.exists():
                raise ValueError("Path does not exist")
            entries = []
            if base.is_file():
                entries.append({"path": base.name, "bytes": base.stat().st_size})
            else:
                for path in sorted(base.rglob("*")):
                    if path.is_file() and ".home" not in path.parts:
                        entries.append({"path": str(path.relative_to(workspace.resolve())), "bytes": path.stat().st_size})
                    if len(entries) >= 500:
                        break
            return {"ok": True, "files": entries, "truncated": len(entries) >= 500}

        if name == "read_file":
            path = _safe_path(workspace, arguments["path"])
            start = max(1, int(arguments.get("start_line", 1)))
            maximum = min(500, max(1, int(arguments.get("max_lines", 250))))
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            selected = lines[start - 1 : start - 1 + maximum]
            content = "\n".join("%d: %s" % (start + index, line) for index, line in enumerate(selected))
            if len(content) > 60000:
                content = content[:60000] + "\n...[truncated]"
            return {"ok": True, "path": arguments["path"], "total_lines": len(lines), "content": content}

        if name == "write_file":
            content = arguments["content"]
            if len(content) > 300000:
                raise ValueError("write_file content exceeds 300000 characters")
            path = _safe_path(workspace, arguments["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return {"ok": True, "path": arguments["path"], "characters": len(content)}

        if name == "run_command":
            cwd = _safe_path(workspace, arguments.get("cwd") or ".")
            timeout = min(600, max(1, int(arguments.get("timeout_seconds", 120))))
            command = arguments["command"]
            started = time.monotonic()
            process = subprocess.Popen(
                ["/bin/bash", "-c", command],
                cwd=str(cwd),
                env=_clean_environment(workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                start_new_session=True,
            )
            timed_out = False
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            return {
                "ok": process.returncode == 0 and not timed_out,
                "returncode": process.returncode,
                "timed_out": timed_out,
                "duration_seconds": round(time.monotonic() - started, 3),
                "stdout_tail": _tail(stdout),
                "stderr_tail": _tail(stderr),
            }
        raise ValueError("Unknown tool: " + name)
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}


def run_agent(
    workspace: Path,
    log_dir: Path,
    task_title: str,
    model: str = DEFAULT_MODEL,
    max_steps: int = 60,
    time_limit_seconds: int = 1800,
    max_output_tokens: int = 8192,
) -> Dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Solve task %s. BRIEF.md and public data/ are in your workspace. Begin by reading the brief." % task_title,
        },
    ]
    events: List[Dict[str, Any]] = []
    started = time.monotonic()
    status = "max_steps"
    error = None
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "upstream_inference_cost_usd": 0.0,
        "market_cost_usd": 0.0,
    }
    steps = 0
    client = None
    deadline = started + max(0, time_limit_seconds)

    try:
        for step in range(1, max_steps + 1):
            if time.monotonic() >= deadline:
                status = "time_limit"
                break
            steps = step
            if client is None:
                client = GatewayClient()
            response = client.chat(
                model=model,
                messages=messages,
                tools=TOOLS,
                max_tokens=max_output_tokens,
                deadline=deadline,
            )
            choice = response["choices"][0]
            raw_message = choice.get("message") or {}
            message = {
                "role": "assistant",
                "content": raw_message.get("content"),
            }
            if raw_message.get("tool_calls"):
                message["tool_calls"] = raw_message["tool_calls"]
            messages.append(message)
            response_usage = response.get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += int(response_usage.get(key, 0) or 0)
            cost_details = response_usage.get("cost_details") or {}
            usage["upstream_inference_cost_usd"] += float(cost_details.get("upstream_inference_cost", 0) or 0)
            usage["market_cost_usd"] += float(response_usage.get("market_cost", 0) or 0)
            event: Dict[str, Any] = {
                "step": step,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "finish_reason": choice.get("finish_reason"),
                "message": raw_message,
                "usage": response_usage,
            }
            events.append(event)
            _json_dump(log_dir / "transcript.json", {"events": events})

            tool_calls = raw_message.get("tool_calls") or []
            if raw_message.get("refusal") or choice.get("finish_reason") == "content_filter":
                status = "provider_blocked"
                break
            if not tool_calls:
                if (workspace / "pipeline" / "predict.sh").is_file():
                    status = "completed"
                    break
                messages.append({
                    "role": "user",
                    "content": "pipeline/predict.sh is still missing. Continue working and create and test the required submission before your final response.",
                })
                continue

            for call in tool_calls:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "time_limit"
                    break
                function = call.get("function") or {}
                name = function.get("name", "")
                arguments_text = function.get("arguments") or "{}"
                try:
                    arguments = json.loads(arguments_text)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                    if name == "run_command":
                        requested_timeout = int(arguments.get("timeout_seconds", 120))
                        arguments["timeout_seconds"] = min(requested_timeout, max(1, math.ceil(remaining)))
                    result = execute_tool(workspace, name, arguments)
                except Exception as exc:
                    result = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
                tool_message = {"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(result, allow_nan=False)}
                messages.append(tool_message)
                events.append({"step": step, "tool": name, "arguments": arguments_text, "result": result})
                _json_dump(log_dir / "transcript.json", {"events": events})
            if status == "time_limit":
                break
        else:
            status = "max_steps"
    except Exception as exc:
        status = "time_limit" if time.monotonic() >= deadline else "agent_error"
        error = "%s: %s" % (type(exc).__name__, exc)
    finally:
        result = {
            "status": status,
            "error": error,
            "model": model,
            "steps": steps,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "usage": usage,
            "pipeline_present": (workspace / "pipeline" / "predict.sh").is_file(),
        }
        _json_dump(log_dir / "agent_result.json", result)
        _json_dump(log_dir / "transcript.json", {"events": events})
    return result
