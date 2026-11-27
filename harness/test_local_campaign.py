"""用具有可控收益和波动的 CPU 回执检查调度；真实逐例计时另由 local_eval 验证。"""

import hashlib
import json
import sys
import tempfile
import types
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from harness import local_campaign as campaign
from harness import workers


SHAPES = ("n100", "h2_w3_b64")
MANIFEST = dict(problem="fixture", dataset_version="set1", manifest_sha256="e" * 64,
                shapes=[dict(shape_id=s, parameters={}) for s in SHAPES], cases=[{"case_id": "mock"}])
CARD = dict(family="vector", change="减少访存指令", falsifier="大图延迟维持原值", target_cases=[SHAPES[0]])


def mock_recompute(raw):
    # 保证调度读取重算时间。回执顶层的伪造总时间始终是 -1。
    values = raw["fixture_shape_means"]
    valid = raw.get("fixture_prewarm") and values.keys() == set(SHAPES)
    return dict(rankable=bool(valid), total_us=sum(values.values()) if valid else None,
                shape_means_us=values if valid else {}, errors=[] if valid else ["缺少预热或形状"])


class CampaignTests(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.dirname = Path(td.name)
        self.c = campaign.Campaign(self.dirname / "campaign")
        stub = types.ModuleType("harness.local_eval")
        stub.manifest = lambda _: MANIFEST
        stub.PROTOCOL_VERSION = "warmed2"
        stub.recompute = mock_recompute
        self.mock_eval = stub
        module = patch.dict(sys.modules, {"harness.local_eval": stub})
        module.start()
        self.addCleanup(module.stop)
        self.tmp_source = self.dirname / "input.cu"
        self.tmp_source.write_text("void base() {}\n")
        self.baseline = self.c.init("fixture", self.tmp_source, contract="fixture ABI")
        self.times = {self.baseline: (10.0, 10.0)}

    def report(self, req):
        timings = self.times[req["source_sha256"]]
        noise = int(req["group_id"][-1]) / 10 if req["kind"] == "calibrate" and req["side"] == "A" else 0
        raw = {k: req[k] for k in ("problem", "source_sha256", "manifest_sha256", "dataset_version", "protocol_version")}
        env = dict.fromkeys(campaign.ENV_KEYS, "mock")
        env["compile_flags"] = ["-O3"]
        return dict(raw, run_id=str(uuid.uuid4()), kind="local_cuda_evaluation",
                    status="ok", rankable=True, total_us=-1, protocol={"warmups": 5}, environment=env,
                    fixture_prewarm=True, fixture_shape_means={s: x + noise for s, x in zip(SHAPES, timings)})

    def drain(self, fn=None):
        for _ in range(20):
            res = self.c.run_next(evaluate_group=fn or (lambda reqs: map(self.report, reqs)))
            if res["status"] == "idle":
                inode = self.c.path.stat().st_ino
                self.assertEqual(self.c.run_next(evaluate_group=lambda reqs: self.fail("空队列不应调用 GPU"))["status"], "idle")
                self.assertEqual(self.c.path.stat().st_ino, inode)
                break
            self.assertEqual(res["status"], "completed", res)
        else:
            self.fail("队列未清空")

    def submit(self, seat, code, times, *, card=None, **kwargs):
        row = self.c.begin(seat, card or CARD, **kwargs)
        self.tmp_source.write_text(f"void {code}() {{}}\n")
        item = self.c.submit(seat, self.tmp_source, summary=code, round_id=row["id"])
        self.times[item["candidate_sha"]] = times
        self.assertEqual(Path(item["source"]).read_bytes(), (Path(row["workspace"]) / "solution.cu").read_bytes())
        return item["candidate_sha"]

    def test_counterbalanced_pairs_do_not_promote_a_position_effect(self):
        self.drain()
        sha = self.submit("a", "same_speed", (10, 10))
        orders = []

        def position_effect(requests):
            orders.append("".join(r["side"] for r in requests))
            for req in requests:
                raw = self.report(req)
                if req["leg_index"] in (1, 2):
                    raw["fixture_shape_means"] = dict.fromkeys(SHAPES, 9.0)
                yield raw

        self.drain(position_effect)
        state = self.c.snapshot()
        self.assertEqual(orders, ["ABBA", "BAAB"])
        self.assertEqual(state["rounds"][0]["comparisons"][0]["gains_us"]["total_us"], [2.0, -2.0])
        self.assertEqual(state["rounds"][0]["decision"], "indistinguishable")
        self.assertEqual(state["best"]["source_sha256"], self.baseline)
        brief = self.c.context("b", compact=True)
        self.assertEqual(brief["measurements_by_source"][sha],
                         dict(rankable_legs=4, median_total_us=19.0,
                              median_shape_means_us=dict.fromkeys(SHAPES, 9.5), failures=[]))
        self.assertEqual(brief["recent_raw_reports"], self.c.context("b")["recent_raw_reports"])

    def test_previous_campaign_keeps_original_order_on_resume(self):
        with self.c._edit() as state:
            del state["comparison_orders"]
            for group in state["groups"].values():
                del group["order"]
        self.c.init("fixture", self.tmp_source, contract="fixture ABI")
        seen = []
        def observe(reqs):
            seen.append("".join(r["side"] for r in reqs))
            return map(self.report, reqs)
        self.drain(observe)
        self.submit("a", "faster", (8, 8))
        self.drain(observe)
        self.assertEqual(seen, ["ABBA"] * 5)
        self.assertEqual(self.c.snapshot()["rounds"][0]["decision"], "promoted")

    def test_queued_peer_keeps_last_measured_source_and_raw_report_in_handoff(self):
        self.drain()
        previous = self.submit("b", "earlier", (11, 9))
        self.drain()
        self.c.review("b", "保留已经测过的结果")
        queued = self.submit("b", "next_kernel", (10, 10))
        context = self.c.context("a", compact=True)
        self.assertTrue({previous, queued} <= context["related_sources"].keys())
        measured = {r["source_sha256"] for r in context["recent_raw_reports"].values()}
        self.assertIn(previous, measured)
        self.assertNotIn(queued, measured)

    def test_profile_binds_to_archived_source_and_shape_without_changing_ranking(self):
        self.drain()
        before = self.c.snapshot()
        report = self.dirname / "REPORT.md"
        report.write_text("NCU diagnostic on the baseline.", encoding="utf-8")
        for sha, shape in [("f" * 64, SHAPES[0]), (self.baseline, "no-such-shape")]:
            with self.assertRaises(ValueError):
                self.c.attach_profile(sha, shape, report)
        entry = self.c.attach_profile(self.baseline, SHAPES[0], report)
        self.assertEqual(self.c.attach_profile(self.baseline, SHAPES[0], report), entry)
        self.assertEqual(self.c.context("a", compact=True)["profiles"], [dict(entry, markdown=report.read_text())])
        after = self.c.snapshot()
        for key in ("reports", "best", "noise_band", "groups"):
            self.assertEqual(after[key], before[key])
        archive = self.c.root / "profile" / entry["report_sha256"] / "REPORT.md"
        archive.write_text("edited")
        with self.assertRaisesRegex(ValueError, "指纹"):
            self.c.context("a", compact=True)

    def test_initializer_pinning_first_baseline_and_recovery(self):
        self.assertEqual([a["agent"] for a in self.c.snapshot()["agents"]], ["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "全套通过"):
            self.c.begin("a", CARD)
        self.assertEqual(self.c.init("fixture", self.tmp_source, contract="fixture ABI"), self.baseline)
        with self.assertRaisesRegex(ValueError, "另一配置"):
            self.c.init("fixture", self.tmp_source, contract="changed")

        def interrupted(reqs):
            yield self.report(reqs[0])
            self.assertEqual(self.c.run_next(self.report)["status"], "busy")
            self.assertEqual(self.c.snapshot()["device"]["state"], "busy")
            self.assertEqual(self.c.begin("a", dict(CARD, target_cases=[100]))["card"]["target_cases"], ["n100"])
            raise campaign.DeviceUnavailable("离线")

        self.assertEqual(self.c.run_next(evaluate_group=interrupted)["status"], "blocked")
        self.c = campaign.Campaign(self.c.root)
        self.assertEqual(self.c.snapshot()["device"]["reason"], "离线")
        self.drain()
        state = self.c.snapshot()
        self.assertEqual(len(state["reports"]), 13)
        first = state["groups"][0]
        self.assertEqual(first["attempt"], 2)
        self.assertTrue(all("-a002-" in x for x in first["report_ids"]))
        self.assertAlmostEqual(state["noise_band"]["total_us"], 0.6)
        self.assertEqual(state["agents"][0]["state"], "developing")
        self.assertEqual(state["best"]["source_sha256"], self.baseline)
        self.assertTrue(all(r["report_sha256"] == hashlib.sha256(Path(r["report"]).read_bytes()).hexdigest()
                            for r in state["reports"]))

    def test_whole_shape_only_stall_and_concurrent_shared_review(self):
        self.drain()
        first = self.submit("a", "faster_total", (7, 9), worker_id="tool-A")
        shape = self.submit("b", "faster_shape", (7, 14), borrowed_from="a")
        span = ["threads128", "threads256"]
        probe = dict(CARD, parameter_range=span, parameter=span[0])
        self.submit("c", "param128", (9.99, 10), card=probe)
        self.drain()
        state = self.c.snapshot()
        self.assertEqual([r["decision"] for r in state["rounds"]], ["promoted", "shape_only", "indistinguishable"])
        self.assertEqual(state["best"]["source_sha256"], first)
        self.assertEqual(state["best"]["total_us"], 16)
        self.assertTrue(any(r["source_sha256"] == shape and r["shape_id"] == SHAPES[0] and
                            r["control_sha"] == self.baseline for r in state["shape_routes"]))
        self.assertEqual({r["parent_sha"] for r in state["agents"]}, {self.baseline})
        self.assertEqual(state["rounds"][0]["worker"]["reasoning_effort"], "max")
        exp = dict(entry_id="memory", experience="记录分布影响。", next_step="更换 tile")
        with ThreadPoolExecutor(max_workers=3) as pool:
            for fut in [pool.submit(self.c.review, agent, "完成本轮", **exp) for agent in ("a", "b", "c")]:
                fut.result()
        for agent in ("a", "b", "c"):
            self.c.review(agent, "完成本轮", **exp)
        self.assertEqual(Path(state["experience_path"]).read_text().count("记录分布影响。"), 3)
        self.assertEqual(self.c.snapshot()["agents"][2]["state"], "idle")
        self.submit("c", "param256", (10.01, 10), card=dict(probe, parameter=span[1]))
        self.drain()
        self.c.review("c", "范围已测完")
        self.assertEqual(self.c.snapshot()["agents"][2]["state"], "waiting_evidence")
        with self.assertRaisesRegex(ValueError, "已停滞"):
            self.c.begin("c", CARD)
        row = self.c.begin("c", CARD, changed_premise="新内存分析显示对齐不同")
        context = self.c.context("c")
        self.assertEqual(row["parent_sha"], self.baseline)
        self.assertIn(shape, context["related_sources"])
        self.assertTrue(context["recent_raw_reports"])
        self.assertIn("记录分布影响", context["experience"])
        other = self.c.begin("b", CARD, parent_sha=first)
        self.assertEqual(other["parent_sha"], first)

    def test_racing_best_and_shared_pair(self):
        self.drain()
        a = self.submit("a", "first", (8, 8))
        b = self.submit("b", "second", (7.8, 8))
        same = self.submit("c", "second", (7.8, 8))
        self.assertEqual(b, same)
        self.drain()
        state = self.c.snapshot()
        self.assertEqual(state["best"]["source_sha256"], a)
        for row in state["rounds"][1:]:
            self.assertEqual([c["control_sha"] for c in row["comparisons"]], [self.baseline, a])
            self.assertTrue(row["comparisons"][0]["whole_win"])
            self.assertFalse(row["comparisons"][1]["whole_win"])
        self.assertEqual(len(state["groups"]), 9)
        self.assertEqual(state["rounds"][1]["comparisons"], state["rounds"][2]["comparisons"])
        self.assertEqual(state["rounds"][1]["report"], state["rounds"][2]["report"])

    def test_unrankable_partial_failure_and_missing_competitor_parent(self):
        self.drain()
        broken = self.submit("a", "missing_case", (1, 1))
        def incomplete(reqs):
            yield self.report(reqs[0])
            raw = self.report(reqs[1])
            raw["fixture_shape_means"].pop(SHAPES[0])
            yield raw
        result = self.c.run_next(evaluate_group=incomplete)
        self.assertEqual(result["outcome"], "failed")
        self.assertEqual(self.c.run_next(self.report)["status"], "idle")
        state = self.c.snapshot()
        self.assertEqual(state["rounds"][0]["decision"], "invalid")
        self.assertEqual(state["best"]["source_sha256"], self.baseline)
        self.assertEqual(state["reports"][-1]["error"], "缺少预热或形状")
        with self.assertRaisesRegex(ValueError, "全套通过"):
            self.c.begin("b", CARD, parent_sha=broken)

    def test_mismatched_device_replayed_run_id_and_outdated_protocol(self):
        self.c.run_next(self.report)
        orig = self.c.snapshot()["reports"][0]
        def duplicate(reqs):
            yield dict(self.report(reqs[0]), run_id=orig["run_id"])
        self.assertIn("重复的测量", self.c.run_next(evaluate_group=duplicate)["reason"])
        def swapped(reqs):
            raw = self.report(reqs[0])
            raw["environment"]["gpu_uuid"] = "other"
            yield raw
        self.assertIn("设备", self.c.run_next(evaluate_group=swapped)["reason"])
        with patch.object(self.mock_eval, "PROTOCOL_VERSION", "future"):
            result = self.c.run_next(lambda _: self.fail("协议升级前不应调用 GPU"))
        self.assertIn("协议已升级", result["reason"])
        self.drain()

    def test_real_worker_entry_pins_effort_and_keeps_retryable_error(self):
        self.drain()
        self.c.begin("c", CARD)
        with patch.object(workers, "develop", side_effect=workers.WorkerError("固定档位不可用")) as fn:
            with self.assertRaisesRegex(workers.WorkerError, "固定档位"):
                self.c.develop("c")
            self.assertIs(fn.call_args.kwargs["max_effort"], True)
        self.assertEqual(self.c.snapshot()["agents"][2]["state"], "developing")
        with patch.object(workers, "develop", return_value=dict(source="void candidate() {}\n", summary="调整访存")):
            row = self.c.develop("c")
        self.assertEqual(row["worker"]["model"], campaign.MODEL)
        self.assertEqual(row["worker"]["reasoning_effort"], "max")
        self.assertEqual(self.c.snapshot()["agents"][2]["state"], "queued")

    def test_older_environment_schema_rebuilt_only_from_intact_matching_reports(self):
        stamp = dict(image_digest="digest", image_id="image", requested_at=12345)
        def report(req):
            raw = self.report(req)
            raw["environment"]["host_attestation"] = dict(stamp)
            return raw
        self.c.run_next(report)
        with self.c._edit() as state:
            state["conditions"].pop("host")
        stamp["requested_at"] += 1
        self.assertEqual(self.c.run_next(report)["status"], "completed")
        self.assertEqual(self.c.snapshot()["conditions"]["host"]["image_digest"], "digest")
        stamp["image_id"] = "changed-image"
        self.assertEqual(self.c.run_next(report)["status"], "blocked")
        stamp["image_id"] = "image"
        with self.c._edit() as state:
            state["conditions"].pop("host")
            old = Path(state["reports"][0]["report"])
            old.write_bytes(old.read_bytes() + b" ")
        self.assertEqual(self.c.run_next(report)["status"], "blocked")
        self.assertNotIn("host", self.c.snapshot()["conditions"])

    def test_cli_model_catalog_and_native_shape_cards(self):
        content = {"models": [{"slug": campaign.MODEL, "supported_reasoning_levels": [{"effort": "max"}]}]}
        cache = self.dirname / "models_cache.json"
        cache.write_text(json.dumps(content))
        def run(name, argv, *rest):
            self.assertEqual(name, "Codex max")
            self.assertEqual(argv[argv.index("--model") + 1], campaign.MODEL)
            self.assertIn('model_reasoning_effort="max"', argv)
            Path(argv[argv.index("--output-last-message") + 1]).write_text(json.dumps(
                {"language": "CUDA C++", "source": "void candidate() {}\n", "summary": "调整访存"}))
        context = dict(language="CUDA C++", cases=MANIFEST["shapes"])
        with (patch.dict("os.environ", {"CODEX_HOME": str(self.dirname)}),
              patch.object(workers, "_codex_sandbox", return_value=[]), patch.object(workers, "_run", side_effect=run) as call):
            workers.develop(context, CARD, codex_bin="/mock/codex", max_effort=True)
            cache.write_text('{"models":[]}')
            with self.assertRaisesRegex(workers.WorkerError, "max"):
                workers.develop(context, CARD, codex_bin="/mock/codex", max_effort=True)
            call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
