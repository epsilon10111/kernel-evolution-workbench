"""三席经过真实 campaign、原始报告重算和独立 GPU 消费入口的离线端到端回归。"""

import json
import threading
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from harness import local_eval, local_loop, workers
from harness.local_campaign import AGENTS, Campaign


def simulated_gpu(request):
    report = local_eval.template(request["problem"], request["source_sha256"], local_eval.KINDS["candidate"])
    report["environment"] = dict(gpu="NVIDIA H20", gpu_uuid="GPU-simulated", driver_version="sim", torch="sim",
                                 cuda="sim", nvcc="sim", compile_flags=local_eval.FLAGS, runner_sha256="0" * 64)
    report["prewarm"] = local_eval.prewarm(local_eval.warm_cases(report["shapes"]), lambda *_: None,
                                         iter([0.0, local_eval.PREWARM_SECONDS + 1.0]).__next__)
    for shape in report["shapes"]:
        for case in shape["cases"]:
            case.update(status="ok", correct=True, gpu_samples_us=[5.0] * 9, e2e_samples_us=[7.0] * 9,
                        median_us=5.0, e2e_median_us=7.0)
        shape["mean_us"] = 5.0
    report.update(status="ok", rankable=True, total_us=5.0 * len(report["shapes"]))
    return report


class AutoLoopTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        base = root / "base.cu"
        base.write_text("void baseline() {}\n")
        self.c = Campaign(root / "campaign")
        self.c.init("hinge", base)
        for _ in range(3):
            self.assertEqual(self.c.run_next(simulated_gpu)["status"], "completed")
        self.calls = []
        self.lock = threading.Lock()

    def model(self, schema, prompt, timeout, *args, **settings):
        self.assertTrue(settings["max_effort"])
        ctx = json.loads(prompt)["context"]
        self.assertNotIn("all_reports", ctx)
        self.assertTrue(ctx["measurements_by_source"])
        agent = ctx["agent"]
        number = max([r["round"] for r in ctx["rounds"] if r["agent"] == agent] or [0])
        if "source" in schema["properties"]:
            phase = "develop"
            output = dict(language=ctx["language"], source=f"void generated_{agent}_{number}() {{}}\n", summary="独立源码")
        elif "conclusion" in schema["properties"]:
            phase = "review"
            self.assertTrue(ctx["recent_raw_reports"])
            mine = next(r for r in ctx["rounds"] if r["id"] == ctx["current_round"])
            self.assertEqual(mine["decision"], "indistinguishable")
            output = dict(conclusion="两组收益均在波动内", next_step="检验另一种访存机制",
                          entry_id="result", experience=f"{agent} 第{number}轮全套未区分。", waiting_reason="")
        else:
            phase = "plan"
            if number == 1:
                self.assertIn("第1轮", ctx["experience"])
            output = dict(action="develop", parent_sha=ctx["eligible_parents"][-1] if number else ctx["parent_sha"],
                          family=f"approach-{agent}-{number + 1}", change=f"路线 {agent} 的第 {number + 1} 个机制",
                          falsifier="全部形状无已复现收益", target_cases=[ctx["cases"][0]["shape_id"]],
                          borrowed_from="已通过的候选" if number else "", changed_premise="",
                          parameter_range=[], parameter="", waiting_reason="")
        with self.lock:
            self.calls.append((agent, phase))
        return output

    def test_three_fixed_model_seats_two_rounds_resume_and_review(self):
        done = threading.Event()
        def consumer():
            while not done.is_set():
                result = self.c.run_next(simulated_gpu)
                if result["status"] == "idle":
                    done.wait(0.01)
                else:
                    self.assertEqual(result["status"], "completed", result)
        with ThreadPoolExecutor(max_workers=1) as card, patch.object(workers, "_codex_json", side_effect=self.model):
            drainer = card.submit(consumer)
            try:
                with patch.object(self.c, "begin", side_effect=OSError("模拟模型返回后主控重启")):
                    with self.assertRaisesRegex(RuntimeError, "主控重启"):
                        local_loop.run(self.c, 1, poll=0.01)
                for rounds in (1, 1, 2):
                    result = local_loop.run(Campaign(self.c.root), rounds, poll=0.01)
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(set(result["rounds"].values()), {rounds})
            finally:
                done.set()
                drainer.result(timeout=3)
        counts = Counter(self.calls)
        for agent in AGENTS:
            for phase in ("plan", "develop", "review"):
                self.assertEqual(counts[agent, phase], 2)
        state = self.c.snapshot()
        self.assertEqual(len(state["rounds"]), 6)
        self.assertTrue(all(r["review"] and r["worker"]["reasoning_effort"] == "max" for r in state["rounds"]))
        self.assertTrue(all(Path(r["round_report"]).is_file() for r in state["rounds"]))
        self.assertEqual(Path(state["experience_path"]).read_text().count("全套未区分"), 6)

    def test_wait_for_evidence_without_busy_model_retries(self):
        def wait_for_data(schema, prompt, timeout, **kwargs):
            ctx = json.loads(prompt)["context"]
            return dict(action="wait", parent_sha=ctx["parent_sha"], family="", change="", falsifier="", target_cases=[],
                        borrowed_from="", changed_premise="", parameter_range=[], parameter="",
                        waiting_reason="缺少新的形状诊断")
        with patch.object(workers, "_codex_json", side_effect=wait_for_data) as model:
            first = local_loop.run(self.c, 2, poll=0.01)
            self.assertEqual(first["status"], "waiting_evidence")
            self.assertEqual(set(first["waiting"]), set(AGENTS))
            local_loop.run(self.c, 2, poll=0.01)
            self.assertEqual(model.call_count, 3)
            path = Path(self.c.snapshot()["experience_path"])
            path.write_text(path.read_text() + "\n新的诊断数据已到。\n")
            local_loop.run(self.c, 2, poll=0.01)
            self.assertEqual(model.call_count, 6)
            report = self.c.root.parent / "REPORT.md"
            report.write_text("Profile exposed a new bottleneck.")
            state = self.c.snapshot()
            for _ in range(2):
                self.c.attach_profile(state["baseline_sha"], state["shape_ids"][0], report)
                local_loop.run(self.c, 2, poll=0.01)
                self.assertEqual(model.call_count, 9)


if __name__ == "__main__":
    unittest.main()
