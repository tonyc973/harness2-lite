"""Offline tests: fake agents and fake LLMs, no network. Run: python tests/test_h2lite.py (or pytest)."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h2lite import as_prompt, improve, shell_agent  # noqa: E402

QUIET = dict(log=lambda *a: None, workers=2)


def _demo():
    spec = importlib.util.spec_from_file_location("demo_agent", ROOT / "examples/offline_demo/agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_llm(replies):
    """An LLM that returns the queued replies in order and remembers the prompts it saw."""
    def llm(prompt):
        llm.prompts.append(prompt)
        return replies.pop(0) if replies else json.dumps({"files": {}})
    llm.prompts = []
    return llm


def test_offline_demo_improves_and_resumes_for_free():
    demo = _demo()
    with tempfile.TemporaryDirectory() as out:
        kw = dict(llm=demo.EDITOR, score=demo.score, test_tasks=demo.TEST_TASKS, out=out, **QUIET)
        res = improve(ROOT / "examples/offline_demo/harness", demo.TASKS, demo.run, **kw)
        assert (res.train_before, res.train_after, res.test_before, res.test_after) == (0.0, 1.0, 0.0, 1.0)
        assert "21%" in (Path(out) / "harness/memory.md").read_text()
        assert "[kept, +1.000] (knowledge)" in (Path(out) / "lessons.md").read_text()
        assert "+Standard VAT" in (Path(out) / "report.md").read_text()
        calls = []
        again = improve(ROOT / "examples/offline_demo/harness", demo.TASKS,
                        lambda h, t: calls.append(t) or demo.run(h, t), **kw)
        assert again.train_after == 1.0 and calls == []  # everything came from the cache


def test_invalid_reply_gets_one_retry_with_the_error():
    good = json.dumps({"files": {"system.md": "Answer: 42"}, "notes": "run 1 answered 41"})
    llm = _fake_llm(["not json at all", good])
    with tempfile.TemporaryDirectory() as out:
        res = improve({"system.md": "Answer."}, ["q"], lambda h, t: h["system.md"],
                      llm=llm, score=lambda t, o: float("42" in o), focuses=["process"], out=out, **QUIET)
    assert res.harness["system.md"] == "Answer: 42"
    assert "YOUR PREVIOUS REPLY WAS REJECTED" in llm.prompts[1] and "not a JSON object" in llm.prompts[1]


def test_bad_file_names_are_rejected_and_nothing_changes():
    bad = json.dumps({"files": {"../escape.sh": "rm -rf /"}})
    llm = _fake_llm([bad, bad])
    with tempfile.TemporaryDirectory() as out:
        res = improve({"system.md": "x"}, ["q"], lambda h, t: "o", llm=llm,
                      score=lambda t, o: 0.0, focuses=["process"], out=out, **QUIET)
        assert res.harness == {"system.md": "x"} and not (Path(out) / "escape.sh").exists()
    assert "not allowed" in llm.prompts[1]


def test_size_cap_is_enforced():
    big = json.dumps({"files": {"system.md": "x" * 500}})
    with tempfile.TemporaryDirectory() as out:
        res = improve({"system.md": "x"}, ["q"], lambda h, t: "o", llm=_fake_llm([big, big]),
                      score=lambda t, o: 0.0, focuses=["process"], max_chars=100, out=out, **QUIET)
    assert res.harness == {"system.md": "x"}


def test_blind_judge_without_a_score_function():
    """No scorer: a judge LLM compares answers it cannot attribute. It must still pick the right one."""
    def agent(h, task):
        return "polite answer" if "polite" in h["system.md"] else "rude answer"

    def llm(prompt):
        if prompt.startswith("Two answers"):
            a = prompt.split("## Answer A")[1].split("## Answer B")[0]
            return "A" if "polite" in a else "B"
        return json.dumps({"files": {"system.md": "Be polite."}, "notes": "run 1 was rude"})

    with tempfile.TemporaryDirectory() as out:
        res = improve({"system.md": "Answer."}, ["t1", "t2", "t3", "t4"], agent, llm=llm,
                      focuses=["process"], out=out, **QUIET)
    assert res.harness["system.md"] == "Be polite."
    assert res.history[0]["candidates"][0]["advantage"] == 0.5  # won every comparison


def test_a_crashing_agent_is_recorded_not_raised():
    def agent(h, task):
        if "safe" not in h["system.md"]:
            raise RuntimeError("boom")
        return "ok"

    fix = json.dumps({"files": {"system.md": "Be safe."}, "notes": "every run crashed with boom"})
    llm = _fake_llm([fix])
    with tempfile.TemporaryDirectory() as out:
        res = improve({"system.md": "Go."}, ["t"], agent, llm=llm, score=lambda t, o: float(o == "ok"),
                      focuses=["process"], out=out, **QUIET)
    assert "ERROR: RuntimeError: boom" in llm.prompts[0]
    assert res.train_after == 1.0


def test_compose_merges_two_candidates():
    def agent(h, task):
        text = as_prompt(h)
        return f"{'A' if 'rule A' in text else '-'}{'B' if 'rule B' in text else '-'}"

    def llm(prompt):
        if "## Focus: process" in prompt:
            return json.dumps({"files": {"a.md": "rule A"}})
        if "## Focus: knowledge" in prompt:
            return json.dumps({"files": {"b.md": "rule B"}})
        return json.dumps({"files": {"a.md": "rule A", "b.md": "rule B"}, "notes": "both helped"})

    with tempfile.TemporaryDirectory() as out:
        res = improve({"system.md": "Go."}, ["t"], agent, llm=llm, score=lambda t, o: o.count("-") * -0.5 + 1,
                      out=out, **QUIET)
    assert res.history[0]["kept"] == "composed" and res.train_after == 1.0


def test_shared_lessons_are_read_and_not_duplicated():
    with tempfile.TemporaryDirectory() as d:
        shared = Path(d) / "lessons.md"
        shared.write_text("- an old lesson from another project\n")
        kw = dict(score=lambda t, o: 0.0, focuses=["process"], lessons=shared, **QUIET)
        llm = _fake_llm([json.dumps({"files": {"system.md": "y"}, "notes": "tried y"})])
        improve({"system.md": "x"}, ["q"], lambda h, t: "o", llm=llm, out=Path(d) / "r1", **kw)
        assert "an old lesson from another project" in llm.prompts[0]
        improve({"system.md": "x"}, ["q"], lambda h, t: "o",
                llm=_fake_llm([json.dumps({"files": {"system.md": "y"}, "notes": "tried y"})]),
                out=Path(d) / "r2", **kw)
        lines = shared.read_text().splitlines()
        assert len(lines) == 2 and "[not kept" in lines[1]


def test_shell_agent_runs_any_program():
    run = shell_agent(f'{sys.executable} -c "import json,os,sys; t=json.load(sys.stdin); '
                      f'print(open(os.environ[\'H2LITE_HARNESS\']+\'/system.md\').read().strip(), t)"')
    out, trace = run({"system.md": "hello"}, "world")
    assert out.strip() == "hello world" and trace == ""


def test_openai_chat_speaks_the_openai_protocol():
    """A local fake server: checks the request shape, the auth header and the retry on HTTP 429."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from unittest import mock

    from h2lite import openai_chat

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers.get("Authorization"), body))
            if len(seen) == 1:  # first call: rate limited
                self.send_response(429)
                self.end_headers()
                return
            reply = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        llm = openai_chat("m1", base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="k")
        with mock.patch("h2lite.plugs.time.sleep"):  # skip the back-off wait
            assert llm("hello", system="be brief") == "hi"
    finally:
        server.shutdown()
    path, auth, body = seen[-1]
    assert len(seen) == 2 and path == "/v1/chat/completions" and auth == "Bearer k"
    assert body == {"model": "m1", "messages": [{"role": "system", "content": "be brief"},
                                                {"role": "user", "content": "hello"}]}
    assert llm.cache_key.endswith("|m1")


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
