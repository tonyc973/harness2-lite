"""h2lite: make an AI agent better by letting an LLM rewrite its instruction files, from its own runs.

A small, dependency-free take on Google's Harness² (arXiv 2610.14224).
"""

from .core import Result, Run, as_prompt, improve, load_harness, save_harness
from .plugs import openai_chat, shell_agent

__all__ = ["improve", "Result", "Run", "load_harness", "save_harness", "as_prompt",
           "openai_chat", "shell_agent"]
__version__ = "0.1.0"
