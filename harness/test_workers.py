import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness import workers


MODEL = "claude-fable-5-2"
SECRET = "synthetic-private-value"
CONTEXT = {
    "language": "cuda", "parent_sha": "abc123", "parent_source": "// 父版本\n",
    "findings": [{"provenance": "proxy"}], "cases": [{"N": 256}, {"N": 4096}],
    "contract": {"entry": "solve"},
}
CARDS = [
    {"family": "layout", "change": "align loads", "falsifier": "same transactions", "target_cases": [4096]},
    {"family": "launch", "change": "fewer blocks", "falsifier": "same latency", "target_cases": [256]},
]
USAGE = {MODEL: {"inputTokens": 11, "outputTokens": 23, "costUSD": 0.01}}
SOURCE = {"language": "cuda", "source": "// 完整文件\nvoid solve() {}\n", "summary": "调整访存"}
CODEX_THREAD = "01234567-89ab-4def-8123-456789abcdef"
CODEX_USAGE = {"input_tokens": 120, "cached_input_tokens": 80, "output_tokens": 14}


def envelope(**changes):
    return {"type": "result", "subtype": "success", "is_error": False, "modelUsage": USAGE,
            "result": "unused", "structured_output": {"cards": CARDS}, "session_id": CODEX_THREAD, **changes}


class WorkerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="test-workers-", dir="/tmp")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.exe = self.root / "mock-cli"
        self.private = self.root / "private"
        self.private.mkdir()
        self.marker = self.private / "private-credential"
        self.marker.write_text(SECRET)
        env = patch.dict(os.environ, {
            "PATH": os.defpath, "HOME": str(self.private), "CODEX_HOME": str(self.private),
            "CODEX_API_KEY": "codex-test-key", "UNRELATED_TOKEN": SECRET, "ANTHROPIC_API_KEY": SECRET,
        }, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.gateway = Mock()
        self.gateway.resolve_claude.return_value = str(self.exe)
        self.gateway.claude_env.return_value = {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:1", "ANTHROPIC_API_KEY": "gateway-test-key",
            "HOME": str(self.private), "CLAUDE_CONFIG_DIR": str(self.private),
        }
        modules = patch.dict(sys.modules, {"harness.gateway": self.gateway})
        modules.start()
        self.addCleanup(modules.stop)

    def executable(self, body):
        self.exe.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body), encoding="utf-8")
        self.exe.chmod(0o700)

    def develop(self):
        return workers.develop(CONTEXT, CARDS[0], codex_bin=self.exe, timeout=5)

    def test_hinge_requests_fast_and_rejects_missing_fast_catalog(self):
        cfg = dict(version="v2", hypothesis="减少网格发射", falsifier="最大档耗时没有下降", next_mechanism="")
        catalog = {"models": [{"slug": "gpt-6.2-astra", "service_tiers": [{"id": "priority", "name": "Fast"}]}]}
        cache = self.private / "models_cache.json"
        cache.write_text(json.dumps(catalog))
        def reply(name, argv, *args, **kwargs):
            self.assertEqual(name, "Codex Fast")
            self.assertIn('service_tier="priority"', argv)
            path = json.loads(next(v.split("=", 1)[1] for v in argv if v.startswith("model_catalog_json=")))
            self.assertEqual(json.loads(Path(path).read_bytes()), catalog)
            Path(argv[argv.index("--output-last-message") + 1]).write_text(json.dumps(cfg))
        with (patch.dict(os.environ, {"CODEX_HOME": str(self.private)}),
              patch.object(workers, "_codex_sandbox", return_value=[str(self.exe)]),
              patch.object(workers.shutil, "which", return_value=str(self.exe)),
              patch.object(workers, "_run", side_effect=reply) as call):
            context = {"options": {"v2": "限网格"}, "language": "CUDA C++"}
            self.assertEqual(workers.propose_hinge(context), cfg)
            cache.write_text('{"models":[]}')
            with self.assertRaisesRegex(workers.WorkerError, "Fast"):
                workers.propose_hinge(context)
            call.assert_called_once()
        self.gateway.claude_env.assert_not_called()

    def test_claude_argv_stdin_environment_and_structured_receipt(self):
        self.executable(f"""
            import json, os, sys
            from pathlib import Path
            args = sys.argv[1:]
            def opt(name): return args[args.index(name) + 1]
            assert {{'-p','--bare','--safe-mode','--restricted','--strict-mcp-config',
                    '--disable-slash-commands','--no-session-persistence'}} <= set(args)
            assert opt('--tools') == opt('--setting-sources') == ''
            assert opt('--model') == {MODEL!r} and opt('--max-budget-usd') == '1'
            assert opt('--permission-mode') == 'dontAsk' and opt('--permission-prompts') == 'none'
            assert opt('--output-format') == 'json' and '--fallback-model' not in args
            schema = json.loads(opt('--json-schema'))['properties']['cards']
            assert schema['minItems'] == schema['maxItems'] == 2
            assert set(schema['items']['required']) == set({list(CARDS[0])!r})
            assert not list(Path.cwd().iterdir())
            assert Path(os.environ['CLAUDE_CONFIG_DIR']).parent == Path(os.environ['HOME'])
            assert Path(os.environ['HOME']).parent == Path.cwd().parent
            assert 'UNRELATED_TOKEN' not in os.environ and 'CODEX_API_KEY' not in os.environ
            assert os.environ['ANTHROPIC_API_KEY'] == 'gateway-test-key'
            payload = json.load(sys.stdin)
            assert payload['context'] == {CONTEXT!r}
            assert all(word in payload['task'] for word in ('两张','可归因','实测','推测','实际 N'))
            assert payload['context']['parent_source'] not in str(args)
            print(json.dumps({envelope()!r}))
        """)
        with patch.object(workers, "_run", wraps=workers._run) as run:
            result = workers.plan(CONTEXT, timeout=5)
        self.assertEqual(result, [dict(card, modelUsage=USAGE) for card in CARDS])
        self.assertFalse(run.call_args.args[3].exists())
        self.gateway.resolve_claude.assert_called_once_with()
        self.gateway.claude_env.assert_called_once_with()

    def test_claude_usage_counts_rejected_result_once_in_configured_directory(self):
        settings = self.private / ".cc-switch"
        settings.mkdir()
        (settings / "settings.json").write_text(json.dumps({"claudeConfigDir": "~/claude-audit"}))
        archive = self.private / "claude-audit" / "projects" / "cc-switch-worker"
        bad = envelope(result=SECRET, structured_output={"cards": []},
                       unrelated=SECRET, modelUsage={MODEL: {
                           "inputTokens": 11, "outputTokens": 23, "cacheReadInputTokens": 89,
                           "cacheCreationInputTokens": 17, "costUSD": 0.01, "unrelated": SECRET,
                       }})
        self.executable(f"""
            import json
            print(json.dumps({bad!r}))
        """)
        for attempt in range(2):
            with self.assertRaisesRegex(workers.WorkerError, "两张"):
                workers.plan(CONTEXT)
        receipt, = archive.iterdir()
        self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)
        raw = receipt.read_text()
        self.assertNotIn(SECRET, raw)
        record = json.loads(raw)
        self.assertEqual(record["sessionId"], CODEX_THREAD)
        self.assertEqual(record["message"]["model"], MODEL)
        self.assertEqual(record["message"]["usage"], {"input_tokens": 11, "output_tokens": 23,
                         "cache_read_input_tokens": 89, "cache_creation_input_tokens": 17})
        self.assertFalse((self.private / ".claude").exists())

    def test_codex_timeout_preserves_complete_usage_already_reported(self):
        lines = [{"type": "thread.started", "thread_id": CODEX_THREAD},
                 {"type": "turn.completed", "usage": CODEX_USAGE}]
        raw = ("\n".join(map(json.dumps, lines)) + "\n").encode()
        with (patch.object(workers.subprocess, "Popen") as launch,
              patch.object(workers.os, "killpg"),
              self.assertRaisesRegex(workers.WorkerError, "超时")):
            proc = launch.return_value.__enter__.return_value
            proc.communicate.side_effect = subprocess.TimeoutExpired("codex", 1, output=raw)
            workers._run("Codex", ["codex"], b"", self.root, {}, 1)
        receipt, = (self.private / ".codex" / "archived_sessions").iterdir()
        self.assertIn(CODEX_THREAD, receipt.name)

    def test_explicit_continuous_mode_has_no_claude_dollar_cap(self):
        result = envelope()
        result["modelUsage"] = {MODEL: {**USAGE[MODEL], "costUSD": 2.5}}
        result["total_cost_usd"] = 2.5
        with patch.object(workers, "_run", return_value=json.dumps(result)) as call:
            cards = workers.plan(CONTEXT, max_budget_usd=None)
        self.assertNotIn("--max-budget-usd", call.call_args.args[1])
        self.assertEqual(cards[0]["modelUsage"][MODEL]["costUSD"], 2.5)
        with self.assertRaises(workers.WorkerError):
            workers._fable_usage(result)

    def test_claude_final_json_and_case_representations(self):
        result = envelope(result=json.dumps({"cards": CARDS}))
        del result["structured_output"]
        for cases in ([256, 4096], [{"n": "256"}, {"n": 4096}], {"256": {}, "4096": {}}, {"1": 256, "2": 4096}):
            with self.subTest(cases=cases), patch.object(workers, "_run", return_value=json.dumps(result)):
                self.assertEqual(len(workers.plan(dict(CONTEXT, cases=cases))), 2)

    def test_claude_rejects_failures_downgrade_and_invalid_cards(self):
        bad = [
            SECRET, json.dumps(envelope(subtype="error_max_budget_usd")), json.dumps(envelope(is_error=True)),
            json.dumps(envelope(type="assistant")), json.dumps(envelope(modelUsage={})),
            json.dumps(envelope(modelUsage={"other": USAGE[MODEL]})),
            json.dumps(envelope(modelUsage={**USAGE, "other": {"outputTokens": 1}})),
            json.dumps(envelope(modelUsage={MODEL: {"outputTokens": 0}})),
            json.dumps(envelope(modelUsage={MODEL: {"outputTokens": True}})),
            json.dumps(envelope(modelUsage={MODEL: dict(USAGE[MODEL], costUSD=1.01)})),
            json.dumps(envelope(total_cost_usd=1.01)),
            json.dumps(envelope(modelUsage={MODEL: dict(USAGE[MODEL], inputTokens=0.5)})),
            json.dumps(envelope(structured_output=None, result=json.dumps({"cards": CARDS}))),
        ]
        for cards in ([], CARDS[:1], CARDS * 2, [CARDS[0]] * 2,
                      [CARDS[0], dict(CARDS[1], family=" Layout ")],
                      [CARDS[0], dict(CARDS[1], change=CARDS[0]["change"])],
                      [CARDS[0], dict(CARDS[1], falsifier="")]):
            bad.append(json.dumps(envelope(structured_output={"cards": cards})))
        for target in ([1, 2], [True], [256.0], ["256"], [], 256):
            bad.append(json.dumps(envelope(structured_output={"cards": [dict(CARDS[0], target_cases=target), CARDS[1]]})))
        for index, raw in enumerate(bad):
            with self.subTest(index=index), patch.object(workers, "_run", return_value=raw):
                with self.assertRaises(workers.WorkerError) as error:
                    workers.plan(CONTEXT)
                self.assertNotIn(SECRET, str(error.exception))

    def test_gateway_fails_closed_without_leaking_the_cause(self):
        self.gateway.claude_env.return_value = {"ANTHROPIC_API_KEY": SECRET}
        with patch.object(workers, "_run") as run:
            with self.assertRaises(workers.WorkerError):
                workers.plan(CONTEXT)
            self.gateway.claude_env.side_effect = RuntimeError(SECRET)
            try:
                workers.plan(CONTEXT)
            except workers.WorkerError:
                self.assertNotIn(SECRET, traceback.format_exc())
            else:
                self.fail("缺失网关应失败")
            run.assert_not_called()

    @unittest.skipUnless(shutil.which("bwrap", path=os.defpath), "需要本机 bwrap")
    def test_codex_argv_sandbox_and_final_file(self):
        self.executable(f"""
            import json, os, sys
            from pathlib import Path
            args = sys.argv[1:]
            def opt(name): return args[args.index(name) + 1]
            assert {{'--no-daemon','exec','--json','--ephemeral','--ignore-rules','--ignore-user-config',
                    '--skip-git-repo-check','--strict-config'}} <= set(args)
            assert opt('--ask-for-approval') == 'never' and opt('--sandbox') == 'read-only'
            assert opt('--model') == 'gpt-6.2-astra' and args[-1] == '-'
            off = {{args[i+1] for i, arg in enumerate(args) if arg == '--disable'}}
            assert {{'shell_tool','unified_exec','view_image','hooks','apps','plugins','multi_agent',
                    'js_repl','browser_use','computer_use','goals'}} <= off
            assert 'mcp_servers={{}}' in args and 'web_search="disabled"' in args
            assert not list(Path.cwd().iterdir())
            assert not Path({str(self.marker)!r}).exists()
            assert not Path({str(Path(__file__).absolute())!r}).exists()
            assert 'UNRELATED_TOKEN' not in os.environ and 'ANTHROPIC_API_KEY' not in os.environ
            assert os.environ['CODEX_API_KEY'] == 'codex-test-key'
            assert Path(os.environ['CODEX_HOME']).parent == Path(os.environ['HOME'])
            payload = json.load(sys.stdin)
            assert payload['context'] == {CONTEXT!r} and payload['card'] == {CARDS[0]!r}
            assert payload['context']['parent_source'] not in str(args)
            schema = json.loads(Path(opt('--output-schema')).read_text())
            assert set(schema['required']) == {{'language','summary','source'}}
            assert schema['properties']['language']['enum'] == ['cuda']
            print(json.dumps({{'type': 'thread.started', 'thread_id': {CODEX_THREAD!r}}}))
            print(json.dumps({{'type': 'item.completed', 'item': {{'text': {SECRET!r}}}}}))
            print(json.dumps({{'type': 'turn.completed', 'usage': dict({CODEX_USAGE!r}, private={SECRET!r})}}))
            assert Path(opt('--output-last-message')).parent == Path.cwd().parent
            Path(opt('--output-last-message')).write_text(json.dumps({SOURCE!r}))
        """)
        with patch.object(workers, "_run", wraps=workers._run) as run:
            self.assertEqual(self.develop(), SOURCE)
        self.assertFalse(run.call_args.args[3].exists())
        archive = self.private / ".codex" / "archived_sessions"
        receipt, = list(archive.iterdir())
        self.assertEqual(receipt.name, f"rollout-cc-switch-worker-{CODEX_THREAD}.jsonl")
        self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)
        raw = receipt.read_text()
        self.assertNotIn(SECRET, raw)
        self.assertEqual([(event["type"], event["payload"]) for event in map(json.loads, raw.splitlines())], [
            ("session_meta", {"id": CODEX_THREAD}),
            ("turn_context", {"model": "gpt-6.2-astra"}),
            ("event_msg", {"type": "token_count", "info": {"total_token_usage": CODEX_USAGE}}),
        ])
        self.assertEqual(self.develop(), SOURCE)
        self.assertEqual(list(archive.iterdir()), [receipt])
        self.assertEqual(receipt.read_text(), raw)
        self.gateway.claude_env.assert_not_called()

    def test_codex_usage_rejects_invalid_counts_without_writing(self):
        lines = [{"type": "thread.started", "thread_id": CODEX_THREAD},
                 {"type": "turn.completed", "usage": dict(CODEX_USAGE, cached_input_tokens=121)}]
        with self.assertLogs("harness.workers", "WARNING"):
            workers._record_codex_usage("\n".join(map(json.dumps, lines)).encode())
        self.assertFalse((self.private / ".codex").exists())

    @unittest.skipUnless(shutil.which("bwrap", path=os.defpath), "需要本机 bwrap")
    def test_codex_file_login_is_read_only_and_other_files_are_hidden(self):
        del os.environ["CODEX_API_KEY"]
        (self.private / "auth.json").write_text("dummy-codex-auth")
        (self.private / "config.toml").write_text("dummy-private-config")
        self.executable(f"""
            import json, os, sys
            from pathlib import Path
            home = Path(os.environ['CODEX_HOME'])
            auth = home / 'auth.json'
            assert auth.read_text() == 'dummy-codex-auth' and not (home / 'config.toml').exists()
            assert not Path({str(self.private)!r}).exists()
            try: auth.write_text('changed')
            except OSError: pass
            else: sys.exit(4)
            Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps({SOURCE!r}))
        """)
        self.assertEqual(self.develop(), SOURCE)
        self.assertEqual((self.private / "auth.json").read_text(), "dummy-codex-auth")

    def test_codex_rejects_missing_ambiguous_or_unsafe_final_output(self):
        private_json = self.private / "private.json"
        private_json.write_text(json.dumps(dict(SOURCE, source=SECRET)))
        failures = [None, "missing", "symlink", "fifo", SECRET, json.dumps(SOURCE) * 2,
                    '{"source":"first","source":"second","language":"cuda","summary":"duplicate"}',
                    json.dumps(dict(SOURCE, other_source="second file")),
                    json.dumps(dict(SOURCE, language="python")), json.dumps(dict(SOURCE, source="   ")),
                    json.dumps(dict(SOURCE, summary="")), json.dumps(dict(SOURCE, source="~~~c\ncode\n~~~")),
                    json.dumps(dict(SOURCE, source="--- a.cu\n+++ b.cu\n@@ -1 +1 @@"))]
        for failure in failures:
            def finish(name, argv, *args, **kwargs):
                target = Path(argv[argv.index("--output-last-message") + 1])
                if failure == "symlink":
                    target.symlink_to(private_json)
                elif failure == "fifo":
                    os.mkfifo(target)
                elif failure is None:
                    target.write_text(json.dumps(SOURCE))
                    raise workers.WorkerError("Codex 执行失败（退出码 7）")
                elif failure != "missing":
                    target.write_text(failure)

            with (self.subTest(failure=failure), patch.object(workers, "_codex_sandbox", return_value=[str(self.exe)]),
                  patch.object(workers, "_run", side_effect=finish) as run):
                with self.assertRaises(workers.WorkerError) as error:
                    self.develop()
                self.assertNotIn(SECRET, str(error.exception))
                run.assert_called_once()

    def test_codex_does_not_fall_back_without_isolation_or_identity(self):
        self.exe.touch()
        with patch.object(workers, "_run") as run:
            with patch.object(shutil, "which", return_value=None), self.assertRaisesRegex(workers.WorkerError, "bwrap"):
                self.develop()
            del os.environ["CODEX_API_KEY"]
            with patch.object(shutil, "which", return_value="/usr/bin/bwrap"), self.assertRaisesRegex(workers.WorkerError, "登录"):
                self.develop()
            run.assert_not_called()

    def test_bad_inputs_do_not_start_any_cli(self):
        with patch.object(workers, "_run") as run:
            for timeout in (0, -1, True, float("inf"), float("nan")):
                with self.assertRaises(workers.WorkerError):
                    workers.plan(CONTEXT, timeout=timeout)
            for cases in ([], [{"size": "missing N"}], [True]):
                with self.assertRaises(workers.WorkerError):
                    workers.plan(dict(CONTEXT, cases=cases))
            with self.assertRaises(workers.WorkerError):
                workers.develop(CONTEXT, dict(CARDS[0], target_cases=[1]))
            run.assert_not_called()
        self.gateway.resolve_claude.assert_not_called()

    def test_exit_status_is_sanitized(self):
        self.executable(f"""
            import sys
            print({SECRET!r}, file=sys.stderr)
            print({SECRET!r})
            sys.exit(17)
        """)
        with self.assertRaisesRegex(workers.WorkerError, "17") as error:
            workers.plan(CONTEXT)
        self.assertNotIn(SECRET, str(error.exception))
        self.exe.unlink()
        with self.assertRaises(workers.WorkerError):
            workers.plan(CONTEXT)

    def test_timeout_kills_descendant_holding_pipe_after_parent_exits(self):
        lockfile = self.root / "child.lock"
        self.executable(f"""
            import fcntl, os, subprocess, sys
            fd = os.open({str(lockfile)!r}, os.O_WRONLY | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(8)'], pass_fds=(fd,))
            os._exit(0)
        """)
        start = time.monotonic()
        with self.assertRaisesRegex(workers.WorkerError, "超时"):
            workers.plan(CONTEXT, timeout=0.5)
        self.assertLess(time.monotonic() - start, 4)
        with lockfile.open() as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() - start > 3:
                        self.fail("后代进程仍持有文件锁")
                    time.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
