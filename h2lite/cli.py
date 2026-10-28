"""h2lite AGENT_FILE --harness DIR: improve an agent from the command line.

AGENT_FILE is a Python file that defines:
  run(harness, task)    -> output, or (output, trace)          required
  TASKS                 list of tasks the editor learns from    required
  score(task, output)   -> float in [0, 1]                      optional (else a blind LLM judge)
  TEST_TASKS            held-out tasks, scored before/after     optional
  EDITOR                callable(prompt) -> str                 optional (else openai_chat(--model))
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .core import improve
from .plugs import openai_chat


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="h2lite", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agent", help="Python file with run(), TASKS, optional score(), TEST_TASKS, EDITOR")
    p.add_argument("--harness", required=True, help="folder with the agent's instruction files")
    p.add_argument("--out", default="h2lite_out", help="results folder (default: h2lite_out)")
    p.add_argument("--rounds", type=int, default=1, help="improvement rounds (default: 1)")
    p.add_argument("--repeats", type=int, default=1, help="runs per task, to average out luck (default: 1)")
    p.add_argument("--model", help="editor/judge model (default: $H2LITE_MODEL or gpt-4o-mini)")
    p.add_argument("--lessons", help="lessons file to read and append (default: OUT/lessons.md)")
    p.add_argument("--focus", default="process,knowledge",
                   help="comma list of editor focuses; any free text works too (default: process,knowledge)")
    p.add_argument("--no-compose", action="store_true", help="skip the merge step")
    p.add_argument("--workers", type=int, default=4, help="parallel agent runs (default: 4)")
    a = p.parse_args(argv)

    path = Path(a.agent).resolve()
    spec = importlib.util.spec_from_file_location("h2lite_agent", path)
    if not path.is_file() or spec is None or spec.loader is None:
        p.error(f"cannot load {a.agent}: give the path to a .py file")
    sys.path.insert(0, str(path.parent))  # so the agent file can import its neighbours
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("run", "TASKS"):
        if not hasattr(mod, name):
            p.error(f"{a.agent} must define {name}")

    improve(a.harness, mod.TASKS, mod.run,
            llm=getattr(mod, "EDITOR", None) or openai_chat(a.model),
            score=getattr(mod, "score", None), test_tasks=getattr(mod, "TEST_TASKS", ()),
            rounds=a.rounds, repeats=a.repeats, out=a.out, lessons=a.lessons,
            focuses=[f.strip() for f in a.focus.split(",") if f.strip()],
            compose=not a.no_compose, workers=a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
