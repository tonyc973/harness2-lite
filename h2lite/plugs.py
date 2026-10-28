"""Ready-made plugs: an OpenAI-compatible LLM and a wrapper that turns any program into an agent."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from .core import save_harness


def openai_chat(model=None, *, base_url=None, api_key=None, temperature=None, timeout=300):
    """Return `llm(prompt, system="") -> str` for any OpenAI-compatible endpoint.

    Works with OpenAI, OpenRouter, Ollama (base_url="http://localhost:11434/v1"), vLLM, LM Studio...
    Defaults come from $H2LITE_MODEL, $OPENAI_BASE_URL and $OPENAI_API_KEY.
    """
    model = model or os.environ.get("H2LITE_MODEL") or "gpt-4o-mini"
    base = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")

    def llm(prompt: str, system: str = "") -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        body = {"model": model, "messages": messages}
        if temperature is not None:  # some reasoning models only accept the default
            body["temperature"] = temperature
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(f"{base}/chat/completions", json.dumps(body).encode(), headers)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.load(resp)["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                if e.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f"{model}: HTTP {e.code}: {e.read()[:300]!r}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 2:
                    raise RuntimeError(f"{model}: {e}") from None
            time.sleep(5 * (attempt + 1))
        raise AssertionError("unreachable")

    llm.cache_key = f"{base}|{model}"  # type: ignore[attr-defined]  # a new model must not reuse cached answers
    return llm


def shell_agent(command: str, timeout: int = 600):
    """Wrap ANY program (Go, Node, a Python script...) as an agent.

    For each run: the harness files are written to a temp folder whose path is in $H2LITE_HARNESS,
    the task arrives as JSON on stdin, stdout is the output and stderr is the trace.
    """
    def run(harness, task):
        with tempfile.TemporaryDirectory(prefix="h2lite-") as d:
            save_harness(harness, d)
            p = subprocess.run(command, shell=True, input=json.dumps(task, default=str),
                               capture_output=True, text=True, timeout=timeout,
                               env={**os.environ, "H2LITE_HARNESS": d})
        if p.returncode:
            raise RuntimeError(f"exit code {p.returncode}: {p.stderr[-800:]}")
        return p.stdout, p.stderr
    return run
