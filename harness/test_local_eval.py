"""本地协议的 CPU 回归。GPU 样本和 C++ 调用在这些测试中均为替身。"""

import ctypes
import io
import json
import os
import shutil
import subprocess
import unittest
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from harness import local_eval as bench


def passed_case(case, center):
    case.update(status="ok", correct=True, gpu_samples_us=[center * k / 5 for k in (7, 1, 5, 8, 2, 9, 4, 3, 6)],
                e2e_samples_us=[center + 2] * 9, median_us=center, e2e_median_us=center + 2)


def prewarm_evidence(shapes):
    return bench.prewarm(bench.warm_cases(shapes), lambda *_: None, iter([2, 2.6]).__next__)


def mock_env():
    return dict(gpu="NVIDIA H20", gpu_uuid="GPU-00000000-1111-2222-3333-444444444444",
                driver_version="test-driver", torch="test", cuda="test", nvcc="test",
                compile_flags=bench.FLAGS.copy(), runner_sha256="a" * 64, host_attestation=None)


def complete(problem="leaky_relu"):
    r = bench.template(problem, "1" * 64, bench.KINDS["candidate"])
    for shape in r["shapes"]:
        for i, case in enumerate(shape["cases"], 1):
            passed_case(case, i)
        shape["mean_us"] = 5.5
    r.update(status="ok", rankable=True, total_us=5.5 * len(r["shapes"]), environment=mock_env(),
             prewarm=prewarm_evidence(r["shapes"]))
    return r


class ProtocolTests(unittest.TestCase):
    def test_manifest_shapes_case_ids_and_dtypes(self):
        for problem, count in (("hinge", 4), ("histogram", 9), ("leaky_relu", 8)):
            with self.subTest(problem=problem):
                m = bench.manifest(problem)
                self.assertEqual(len(m["shapes"]), count)
                self.assertEqual([c["case_id"] for c in m["cases"]], list(bench.CASE_IDS))
                self.assertEqual(len({seed for s in m["shapes"] for seed in s["seeds"].values()}), count * 10)
                self.assertEqual(bench.manifest(problem), m)
                for shape in m["shapes"]:
                    for t in shape["tensors"].values():
                        self.assertEqual((t["dtype"], t["layout"]), ("float32", "contiguous"))
        leaky = bench.manifest("leaky_relu")
        self.assertEqual([s["parameters"] for s in leaky["shapes"]],
                         [dict(n=n, m=m, alpha=a) for n, m in ((257, 513), (2049, 4097)) for a in (.02, .125, .25, .5)])
        self.assertEqual(bench.manifest("histogram")["shapes"][0]["tensors"]["image"]["shape"], [63, 65])
        self.assertEqual(bench.manifest("hinge")["shapes"][-1]["parameters"], {"n": 8_388_611})
        self.assertEqual([k["name"] for k in leaky["abi"]], ["input", "alpha", "output", "n", "m"])

    def test_exact_75_5_from_450_gpu_and_450_e2e_samples(self):
        expected = bench.manifest("leaky_relu")["shapes"][:5]
        rows = complete()["shapes"][:5]
        for value, row in zip((None, 8, 12, 20, 30), rows):
            if value:
                for case in row["cases"]:
                    passed_case(case, value)
                row["mean_us"] = value
        means, total, errors = bench.summarize_shapes(expected, rows)
        self.assertEqual((list(means.values()), total, errors), ([5.5, 8, 12, 20, 30], 75.5, []))
        self.assertEqual(len({(s["shape_id"], c["case_id"]) for s in rows for c in s["cases"]}), 50)

    def test_missing_wrong_duplicate_nonfinite_and_forged_fields_never_rank(self):
        source = complete()
        self.assertEqual(bench.recompute(source)["total_us"], 44.0)
        changes = [
            lambda r: r["shapes"].pop(),
            lambda r: r["shapes"][0]["cases"].pop(),
            lambda r: r["shapes"][0]["cases"].__setitem__(1, r["shapes"][0]["cases"][0]),
            lambda r: r["shapes"][0]["cases"][0].update(correct=False),
            lambda r: r["shapes"][0]["cases"][0].update(seed=0),
            lambda r: r["shapes"][0]["cases"][0]["gpu_samples_us"].pop(),
            lambda r: r["shapes"][0]["cases"][0]["e2e_samples_us"].__setitem__(0, float("nan")),
            lambda r: r["shapes"][0]["cases"][0]["gpu_samples_us"].__setitem__(0, False),
            lambda r: r["shapes"][0]["cases"][0].update(median_us=100),
            lambda r: r["shapes"][0].update(mean_us=100),
            lambda r: r.update(total_us=100),
            lambda r: r.update(protocol_version="old"),
            lambda r: r.update(status="running"),
            lambda r: r["prewarm"].update(elapsed_s=.49),
            lambda r: r["prewarm"].update(cases=[]),
            lambda r: r["prewarm"].update(synchronized=False),
        ]
        for change in changes:
            bad = deepcopy(source)
            change(bad)
            result = bench.recompute(bad)
            self.assertFalse(result["rankable"])
            self.assertIsNone(result["total_us"])
            self.assertTrue(result["errors"])

    def test_prewarm_full_cycles_before_clock_cutoff_and_copy_exclusion(self):
        rows = complete()["shapes"]
        selected = bench.warm_cases(rows)
        calls = []
        record = bench.prewarm(selected, lambda *a: calls.append(a), iter((10, 10.24, 10.54)).__next__)
        self.assertEqual(len(calls), 8)
        self.assertEqual(calls[:4], calls[4:])
        self.assertEqual(record["cycles"], 2)
        self.assertTrue(bench.verified_prewarm(record, rows))
        self.assertEqual([c["calls"] for c in record["cases"]], [2 * bench.PREWARM_CALLS] * 4)
        source = complete()
        source.update(kind=bench.KINDS["copy"], source_sha256=bench.sha256(bench.COPY_SOURCE), rankable=False)
        self.assertEqual(bench.recompute(source), dict(rankable=False, total_us=44.0,
                         shape_means_us={s["shape_id"]: 5.5 for s in rows}, errors=[]))

    def test_abi_order_pointer_width_and_library_identity(self):
        inputs = [SimpleNamespace(data_ptr=lambda: 0x100000020), SimpleNamespace(data_ptr=lambda: 0x200000010)]
        output = SimpleNamespace(data_ptr=lambda: 0x300000010)
        for problem, params, expected in (
                ("hinge", dict(n=2**34), [0x100000020, 0x200000010, 0x300000010, 2**34]),
                ("histogram", dict(num_bins=256, height=2**33, width=3), [0x100000020, 256, 0x300000010, 2**33, 3]),
                ("leaky_relu", dict(alpha=.1, n=4, m=2**34), [0x100000020, ctypes.c_float(.1).value, 0x300000010, 4, 2**34])):
            s = bench.PROBLEMS[problem]
            vals = []
            callback = ctypes.CFUNCTYPE(None, *(bench.CTYPE[t] for _, t in s.abi))(lambda *args: vals.extend(args))
            callback(*bench.abi_args(s, params, inputs, output))
            self.assertEqual(vals, expected)
            self.assertNotIn(b"ARG_", bench.timer_source(s))
        with TemporaryDirectory() as tmp, patch.object(bench, "compile_library", side_effect=[MagicMock(), MagicMock(), MagicMock()]) as compile_:
            timer = bench.NativeTimer(bench.PROBLEMS["leaky_relu"], Path(tmp))
            a, b = b"candidate a", b"candidate b"
            first = timer.load(bench.sha256(a), a)
            second = timer.load(bench.sha256(b), b)
            self.assertIs(first, timer.load(bench.sha256(a), a))
            self.assertIsNot(first, second)
            self.assertEqual(compile_.call_count, 3)
            self.assertNotEqual(compile_.call_args_list[1].args[1], compile_.call_args_list[2].args[1])

    @unittest.skipUnless(shutil.which("c++"), "需要宿主 C++ 编译器验证 ctypes ABI")
    def test_native_timer_runs_compiled_callback_with_exact_call_counts_and_no_buffer_reset(self):
        header = """
        #pragma once
        #include <cstddef>
        using cudaError_t = int;
        using cudaEvent_t = int*;
        constexpr int cudaSuccess = 0;
        inline int cudaEventCreate(int** p) { *p = new int(0); return 0; }
        inline int cudaEventDestroy(int* p) { delete p; return 0; }
        inline int cudaEventRecord(int*, int) { return 0; }
        inline int cudaEventElapsedTime(float* t, int*, int*) { *t = 0.015625f; return 0; }
        inline int cudaDeviceSynchronize() { return 0; }
        inline int cudaGetLastError() { return 0; }
        inline const char* cudaGetErrorString(int) { return "error"; }
        """
        counters = b"""
        #include <cstddef>
        extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {
            output[0] += 1;
            output[1] = input[0] * alpha;
            output[2] = float((n >> 32) + (m >> 32));
        }
        """
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cuda_runtime.h").write_text(header)

            def host_build(data, directory, *, candidate):
                directory.mkdir()
                source, library = directory / "source.cpp", directory / "host.so"
                source.write_bytes(data)
                subprocess.run(["c++", "-std=c++17", "-O2", "-I", str(root), "-fPIC", "-shared",
                                "-Wl,-Bsymbolic", str(source), "-o", str(library)], capture_output=True, check=True)
                return ctypes.CDLL(str(library), mode=os.RTLD_LOCAL | os.RTLD_NOW)

            with patch.object(bench, "compile_library", side_effect=host_build):
                timer = bench.NativeTimer(bench.PROBLEMS["leaky_relu"], root)
                entry = timer.load(bench.sha256(counters), counters)
                x, out = ctypes.c_float(-5), (ctypes.c_float * 3)(0, 0, 0)
                args = (ctypes.addressof(x), .2, ctypes.addressof(out), 2**33, 2**34)
                timer.warm(entry, args, 3)
                self.assertEqual(list(out), [3, -1, 6])
                case = {}
                timer.measure(entry, args, case)
                self.assertEqual(list(out), [3 + 5 + 9, -1, 6])
                self.assertEqual(case["gpu_samples_us"], [15.625] * 9)
                self.assertTrue(all(t > 0 for t in case["e2e_samples_us"]))
                self.assertEqual(case["median_us"], 15.625)

    def test_batch_all_legs_in_one_session_failed_leg_retains_evidence(self):
        import torch

        def case_result(timer, entry, sid, case):
            if entry == b"bad" and case["case_id"] == "c03":
                raise bench.WrongAnswer("同址改写未生效")
            passed_case(case, 2)

        dataset = SimpleNamespace(prewarm=lambda timer, entry, shapes: prewarm_evidence(shapes),
                                  measure_case=case_result)
        timer = SimpleNamespace(load=lambda digest, data: data)
        with TemporaryDirectory() as tmp, patch.object(bench, "device_environment", return_value=("dummy", mock_env())), \
                patch.object(torch.cuda, "default_stream"), patch.object(torch.cuda, "stream", return_value=nullcontext()), \
                patch.object(bench, "NativeTimer", return_value=timer), patch.object(bench, "Dataset", return_value=dataset) as prepare:
            path = Path(tmp) / "abba.json"
            r = bench.evaluate("leaky_relu", [b"good", b"bad"], [0, 0, 1, 0], path, batch=True)
            self.assertEqual(r, json.loads(path.read_bytes()))
            self.assertEqual((r["status"], r["failure_kind"]), ("failed", "candidate"))
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(len({leg["run_id"] for leg in r["reports"]}), 3)
            self.assertTrue(all(bench.recompute(leg)["rankable"] for leg in r["reports"][:2]))
            bad = r["reports"][2]
            self.assertFalse(bench.recompute(bad)["rankable"])
            self.assertIsNone(bad["total_us"])
            self.assertEqual(sum(c["status"] == "failed" for s in bad["shapes"] for c in s["cases"]), 8)
            self.assertEqual(sum(c["status"] == "ok" for s in bad["shapes"] for c in s["cases"]), 72)
            self.assertIn("未生效", bad["error"])

    def test_offline_cli_manifest_and_recompute(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(bench.main(["manifest", "--problem", "hinge"]), 0)
        self.assertEqual(len(json.loads(stdout.getvalue())["shapes"]), 4)
        with TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps(complete()))
            self.assertEqual(bench.main(["recompute", "--report", str(path)]), 0)


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def inputs(self, problem, params):
        s = bench.PROBLEMS[problem]
        result = []
        for i in range(1, 11):
            args = (self.torch, "cpu", params, i, 197 + i, 901)
            data = s.generate(*args)
            self.assertTrue(all(t.equal(repeat) for t, repeat in zip(data, s.generate(*args))))
            self.assertTrue(all(t.is_contiguous() and t.dtype == self.torch.float32 for t in data))
            result.append(data)
        return result

    def test_delayed_output_is_caught_even_if_device_sync_eventually_fixes_it(self):
        torch = self.torch
        outer = bench.Dataset.__new__(bench.Dataset)
        pending = []
        def sync(_device):
            while pending:
                pending.pop(0)()
        cuda = SimpleNamespace(Event=lambda: SimpleNamespace(record=lambda s: None, query=lambda: False),
                               stream=lambda s: nullcontext(), synchronize=sync, _sleep=lambda cycles: None)
        outer.torch = SimpleNamespace(cuda=cuda)
        outer.device, outer.default, outer.producer = "cpu", MagicMock(), MagicMock()
        src = torch.tensor([-2., 1., 0., 4.])
        expected = torch.tensor([-.2, 1., 0., 4.])
        active, out, observed = src.clone(), torch.empty_like(src), torch.empty_like(src)
        outer.data = {("s", "c01"): ((src,), expected)}
        outer.buffers = {"s": ((active,), out, observed, ())}
        def late():
            pending.append(lambda: out.copy_(expected))
        with self.assertRaisesRegex(bench.WrongAnswer, "立即消费"):
            outer.validate(late, "s", "c01")
        self.assertTrue(out.equal(expected))
        outer.validate(lambda: out.copy_(torch.where(active > 0, active, active * .1)), "s", "c01")

    def test_same_address_stale_output_and_post_timing_corruption_both_fail(self):
        torch = self.torch
        outer = bench.Dataset.__new__(bench.Dataset)
        cuda = SimpleNamespace(Event=lambda: SimpleNamespace(record=lambda s: None, query=lambda: False),
                               stream=lambda s: nullcontext(), synchronize=lambda d: None, _sleep=lambda cycles: None)
        outer.torch = SimpleNamespace(cuda=cuda)
        outer.device, outer.default, outer.producer = "cpu", MagicMock(), MagicMock()
        values = [torch.tensor(v) for v in ([-2., 1.], [2., 2.], [-1., -1.])]
        outer.data = {("s", f"c0{i}"): ((v,), v.clone()) for i, v in enumerate(values, 1)}
        active, out, observed = torch.empty(2), torch.empty(2), torch.empty(2)
        outer.buffers = {"s": ((active,), out, observed, ())}
        previous = None
        def skip_same_inputs():
            nonlocal previous
            if previous is None or not previous.equal(active):
                out.copy_(active)
                previous = active.clone()
        with self.assertRaisesRegex(bench.WrongAnswer, "立即消费"):
            outer.measure_case(MagicMock(), skip_same_inputs, "s", {"case_id": "c01"})
        timer = SimpleNamespace(measure=lambda *args: out.zero_())
        with self.assertRaisesRegex(bench.WrongAnswer, "连续计时后的完整输出"):
            outer.measure_case(timer, lambda: out.copy_(active), "s", {"case_id": "c01"})

    def test_hinge_margins_and_golden_reference(self):
        torch = self.torch
        p = {"n": 1027}
        data = self.inputs("hinge", p)
        self.assertTrue(all(((t == 1) | (t == -1)).all() for _, t in data))
        self.assertTrue(data[1][1].eq(1).all() and data[2][1].eq(-1).all())
        self.assertTrue(all(data[3][1].equal(t) for _, t in data[3:]))
        margins = [x * t for x, t in data]
        for m, want in zip(margins[1:6], [.5, 2, 1, 1 - 2**-24, 1 + 2**-23]):
            self.assertTrue(m.eq(want).all())
        self.assertTrue(margins[6].equal(torch.arange(p["n"]) % 2 * 2))
        self.assertTrue(margins[7].equal(torch.arange(p["n"]) // 32 % 2 * 2))
        self.assertEqual(int((margins[8] == 0).sum()), 1)
        self.assertEqual(int((margins[9] == 2).sum()), 1)
        self.assertEqual((margins[8][-1], margins[9][-1]), (0, 2))
        for predictions, targets in data:
            want = [max(0., 1. - x * y) for x, y in zip(predictions.tolist(), targets.tolist())]
            self.assertEqual(bench.hinge_reference(torch, (predictions, targets), p).tolist(), want)

    def test_histogram_all_patterns_and_python_counts(self):
        torch = self.torch
        p = {"height": 16, "width": 64, "num_bins": 16}
        images = [args[0] for args in self.inputs("histogram", p)]
        flat = [x.flatten() for x in images]
        self.assertTrue((flat[0] >= 0).all() and (flat[0] < 16).all())
        self.assertTrue(flat[1].eq(0).all() and flat[2].eq(15).all())
        self.assertTrue(flat[3].equal(torch.arange(1024) % 2 * 15))
        self.assertEqual((flat[4] != 0).sum().item(), 51)
        self.assertTrue(flat[5].equal(torch.arange(1024) % 16))
        self.assertTrue(flat[6].equal(torch.arange(1024) // 32 % 16))
        self.assertTrue(images[7].equal((torch.arange(64) // 4).expand(16, 64)))
        self.assertTrue((flat[8] % 8).eq(0).all())
        self.assertTrue(flat[9][:-1].eq(0).all() and flat[9][-1].item() == 15)
        for image in images:
            reference = bench.histogram_reference(torch, (image,), p)
            self.assertEqual(reference.tolist(), [image.flatten().tolist().count(i) for i in range(16)])

    def test_leaky_all_patterns_and_scalar_reference(self):
        torch = self.torch
        p = {"n": 8, "m": 17, "alpha": .05}
        inputs = self.inputs("leaky_relu", p)
        flat = [x.flatten() for x, in inputs]
        self.assertTrue((flat[0] < 0).any() and (flat[0] > 0).any())
        self.assertTrue(flat[1].gt(0).all() and flat[2].lt(0).all())
        self.assertTrue(flat[3].eq(0).all() and torch.signbit(flat[3][1::2]).all())
        self.assertEqual(flat[4][:5].tolist(), [-2**-40, -2**-20, 0, 2**-40, 2**-20])
        self.assertTrue(flat[5].equal(torch.arange(136) % 2 * 2 - 1))
        self.assertTrue(flat[6].equal(torch.arange(136) // 32 % 2 * 2 - 1))
        self.assertEqual(flat[7].eq(1).sum().item(), 7)
        self.assertTrue(flat[8][:-1].eq(1).all() and flat[8][-1].item() == -1)
        self.assertTrue(flat[9][:-1].eq(-1).all() and flat[9][-1].item() == 1)
        f32 = lambda v: ctypes.c_float(v).value
        for x, in inputs:
            want = [v if v > 0 else f32(f32(p["alpha"]) * v) for v in x.flatten().tolist()]
            self.assertEqual(bench.leaky_reference(torch, (x,), p).flatten().tolist(), want)


if __name__ == "__main__":
    unittest.main()
