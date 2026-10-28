"""Offline demo: the full h2lite loop with NO API key.

The "agent" is a tiny function that obeys its harness: it looks for a VAT rate in the
instruction files and falls back to 19%, the old Romanian rate. Since 1 Aug 2025 the
standard rate is 21%, so the starting harness is wrong. The "editor" (EDITOR below) is
scripted: it reads the runs it is shown and, like a real LLM should, only edits when a run
shows the problem.

    h2lite examples/offline_demo/agent.py --harness examples/offline_demo/harness --out runs/demo
"""
import json
import re
from decimal import ROUND_HALF_UP, Decimal

import h2lite

TASKS = [["100.00", "250.50"], ["1200.00", "-200.00", "33.33"], ["19.99"]]   # net amounts per invoice
TEST_TASKS = [["10.00", "5.00"]]                                               # held out: never shown to the editor


def total(amounts, rate):
    return (sum(map(Decimal, amounts)) * (1 + Decimal(rate) / 100)).quantize(Decimal("0.01"), ROUND_HALF_UP)


def run(harness, task):
    m = re.search(r"VAT[^0-9]*?(\d+)%", h2lite.as_prompt(harness))
    rate = int(m.group(1)) if m else 19
    return str(total(task, rate)), f"VAT used: {rate}% (rate found in the harness: {bool(m)})"


def score(task, output):
    return float(output == str(total(task, 21)))   # the answer key lives HERE, never inside a task


def EDITOR(prompt):
    if "## Focus: knowledge" in prompt and "VAT used: 19%" in prompt:
        return json.dumps({
            "files": {"memory.md": "Standard VAT in Romania: 21% (since 1 Aug 2025; 19% before).\n"},
            "notes": "Every run printed 'VAT used: 19%' and scored 0: the agent does not know the current rate.",
        })
    return json.dumps({"files": {}, "notes": "No change: the runs show no problem in this focus."})
