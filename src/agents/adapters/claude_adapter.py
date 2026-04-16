"""
adapters/claude_adapter.py
Claude CLI adapter — calls `claude -p "prompt"` as subprocess.
Requires: `claude auth login` has been run and persists.
"""
import subprocess
import time
import shlex
from typing import Optional

from .base_adapter import BaseAdapter, AdapterResponse


class ClaudeAdapter(BaseAdapter):
    """
    Judge backend using Claude CLI subprocess.
    Invokes: claude -p "prompt" [--model model-name]
    Model: claude-sonnet-4-6 (default), claude-opus-4, claude-haiku-4
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    CLI_COMMAND = "claude"

    def __init__(
        self,
        agent_id: str = "claude-judge",
        model: Optional[str] = None,
        timeout: int = 120,
    ):
        super().__init__(agent_id, model or self.DEFAULT_MODEL)
        self.timeout = timeout

    def model_name(self) -> str:
        return self.model

    def health_check(self) -> bool:
        """Check if claude CLI is installed and authed."""
        try:
            result = subprocess.run(
                [self.CLI_COMMAND, "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return "authenticated" in result.stdout.lower() or result.returncode == 0
        except Exception:
            return False

    def ping(self) -> Optional[float]:
        """Ping Claude CLI with a minimal prompt — returns latency ms or None."""
        import time
        start = time.monotonic()
        try:
            result = subprocess.run(
                [self.CLI_COMMAND, "-p", "hi"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                return (time.monotonic() - start) * 1000
            return None
        except Exception:
            return None

    def call(self, prompt: str, **kwargs) -> AdapterResponse:
        start = time.monotonic()
        try:
            cmd = [self.CLI_COMMAND, "-p", prompt]
            # If a specific model is requested and differs from default, pass --model
            model_arg = kwargs.get("model")
            if model_arg and model_arg != self.DEFAULT_MODEL:
                cmd.insert(2, "--model")
                cmd.insert(3, model_arg)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=kwargs.get("timeout", self.timeout),
                cwd=kwargs.get("cwd"),
            )
            latency = (time.monotonic() - start) * 1000

            if result.returncode != 0:
                return AdapterResponse(
                    text="",
                    latency_ms=latency,
                    success=False,
                    error=result.stderr or f"exit {result.returncode}",
                )

            text = result.stdout.strip()
            return AdapterResponse(text=text, latency_ms=latency, success=True)

        except subprocess.TimeoutExpired:
            latency = (time.monotonic() - start) * 1000
            return AdapterResponse(
                text="", latency_ms=latency, success=False, error="timeout"
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return AdapterResponse(text="", latency_ms=latency, success=False, error=str(e))

    def call_json(self, prompt: str, **kwargs) -> AdapterResponse:
        """
        Call Claude and parse response as JSON.
        Strips any markdown code fences before parsing.
        """
        resp = self.call(prompt, **kwargs)
        if not resp.success:
            return resp

        text = resp.text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])  # drop first line (```json)
            if text.endswith("```"):
                text = text[:-3]
        text = text.strip()

        # Re-wrap in AdapterResponse with parsed text (caller parses json.loads)
        return AdapterResponse(text=text, latency_ms=resp.latency_ms, success=True)