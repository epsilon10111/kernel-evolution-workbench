"""整题 H20 本地评测器。manifest / recompute 可离线运行，run / batch / copy 在单卡锁内使用。"""

import argparse
import ctypes
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple


WARMUP, SAMPLES = 5, 9
CASE_IDS = tuple(f"c{i:02d}" for i in range(1, 11))
FLAGS = ["-O3", "-std=c++17", "-arch=sm_90", "--default-stream=legacy", "--cudart=shared",
         "--shared", "-Xcompiler=-fPIC", "-Xlinker=-Bsymbolic"]
PROTOCOL_VERSION = "h20-native-event-3"
PREWARM_SECONDS, PREWARM_CALLS = 0.5, 16
PROTOCOL = {
    "version": PROTOCOL_VERSION, "warmup_calls_per_case": WARMUP, "samples_per_case": SAMPLES,
    "unit": "us", "gpu_timer": "cudaEventElapsedTime on legacy default stream; 1 native call per pair",
    "e2e_timer": "C++ steady_clock from immediately before solution to cudaDeviceSynchronize after stop event",
    "buffers": "preallocated contiguous device tensors; fixed active addresses per shape; no L2 flush",
    "outside_timing": "input generation, reference, compilation, same-address probes with poisoned output and "
                      "immediate consumer; full output check after the nine samples",
    "output_initialization": "output reused without runner clearing during five warmups and nine timed calls; "
                             "any operator-required initialization is timed",
    "aggregation": "sum(shape arithmetic means of ten case medians); no rounding",
    "prewarm": {"min_seconds": PREWARM_SECONDS, "calls_per_case_per_cycle": PREWARM_CALLS,
                "selection": "first and last benchmark shape; first and seventh distribution at both shapes",
                "per_leg": True, "synchronized": True, "excluded_from_score": True},
}
KINDS = {"candidate": "local_cuda_evaluation", "copy": "local_cuda_copy_control"}
CTYPE = {"const float*": ctypes.c_void_p, "float*": ctypes.c_void_p,
         "float": ctypes.c_float, "int": ctypes.c_int, "size_t": ctypes.c_size_t}
POISON_DELAY_CYCLES = (4_000_000, 40_000_000)


class CandidateError(Exception):
    """候选的编译、ABI 或执行失败。"""


class WrongAnswer(CandidateError):
    """可以继续诊断其余用例的数值错误。"""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def seed_for(problem, shape_id, case_id):
    key = f"local-input-v1/{problem}/{shape_id}/{case_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & (2**63 - 1)


def generator(torch, device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def hinge_inputs(torch, device, p, case, seed, label_seed):
    n = p["n"]
    rng = generator(torch, device, seed)
    values = torch.empty(n, dtype=torch.float32, device=device)
    labels = torch.empty_like(values)
    if case == 1:
        values.uniform_(-2, 2, generator=rng)
        labels.bernoulli_(0.5, generator=rng).mul_(2).sub_(1)
    elif case in (2, 3):
        labels.fill_(1 if case == 2 else -1)
        values.fill_(0.5 if case == 2 else -2)
    else:
        labels.bernoulli_(0.5, generator=generator(torch, device, label_seed)).mul_(2).sub_(1)
        if case in (4, 5, 6):
            values.fill_({4: 1.0, 5: 1.0 - 2**-24, 6: 1.0 + 2**-23}[case])
        elif case in (7, 8):
            i = torch.arange(n, dtype=torch.int32, device=device)
            values.copy_(i.floor_divide_(1 if case == 7 else 32).remainder_(2).mul_(2))
        else:
            values.fill_(2 if case == 9 else 0)
            values[-1] = 0 if case == 9 else 2
        values.mul_(labels)
    return values, labels


def histogram_inputs(torch, device, p, case, seed, label_seed):
    h, w, bins = p["height"], p["width"], p["num_bins"]
    n = h * w
    rng = generator(torch, device, seed)
    image = torch.empty(n, dtype=torch.float32, device=device)
    if case == 1:
        image.random_(0, bins, generator=rng)
    elif case in (2, 3, 10):
        image.fill_(bins - 1 if case == 3 else 0)
        if case == 10:
            image[-1] = bins - 1
    elif case == 5:
        image.zero_()
        count = (n + 10) // 20
        positions = torch.randperm(n, device=device, generator=rng)[:count]
        image[positions] = torch.empty(count, device=device, dtype=image.dtype).random_(1, bins, generator=rng)
    elif case == 8:
        columns = torch.arange(w, dtype=torch.int32, device=device).mul_(bins).floor_divide_(w)
        image.view(h, w).copy_(columns.expand(h, w))
    elif case == 9:
        image.random_(0, (bins + 7) // 8, generator=rng).mul_(8)
    else:
        i = torch.arange(n, dtype=torch.int32, device=device)
        if case == 4:
            i.remainder_(2).mul_(bins - 1)
        else:
            i.floor_divide_(32 if case == 7 else 1).remainder_(bins)
        image.copy_(i)
    return (image.view(h, w),)


def leaky_inputs(torch, device, p, case, seed, label_seed):
    n, m = p["n"], p["m"]
    length = n * m
    x = torch.empty(length, dtype=torch.float32, device=device)
    rng = generator(torch, device, seed)
    if case <= 3:
        lo, hi = ((-2, 2), (0.125, 2), (-2, -0.125))[case - 1]
        x.uniform_(lo, hi, generator=rng)
    elif case in (4, 5):
        pattern = (0.0, -0.0) if case == 4 else (-2**-40, -2**-20, 0.0, 2**-40, 2**-20)
        for offset, value in enumerate(pattern):
            x[offset::len(pattern)].fill_(value)
    elif case in (6, 7):
        i = torch.arange(length, dtype=torch.int32, device=device)
        x.copy_(i.floor_divide_(32 if case == 7 else 1).remainder_(2).mul_(2).sub_(1))
    elif case == 8:
        x.fill_(-1)
        positions = torch.randperm(length, device=device, generator=rng)[:(length + 10) // 20]
        x[positions] = 1
    else:
        x.fill_(1 if case == 9 else -1)
        x[-1] = -1 if case == 9 else 1
    return (x.view(n, m),)


def hinge_reference(torch, inputs, p):
    predictions, targets = inputs
    return torch.clamp(1 - predictions * targets, min=0)


def histogram_reference(torch, inputs, p):
    bins = p["num_bins"]
    pixels = inputs[0].flatten().clamp(0, bins - 1).to(torch.int64)
    return torch.bincount(pixels, minlength=bins).to(torch.float32)


def leaky_reference(torch, inputs, p):
    x = inputs[0]
    return torch.where(x > 0, x, p["alpha"] * x)


class Problem(NamedTuple):
    abi: tuple
    dimensions: dict
    shapes: tuple
    cases: tuple
    generate: object
    reference: object
    reference_note: str


PROBLEMS = {
    "hinge": Problem(
        (("predictions", "const float*"), ("targets", "const float*"), ("output", "float*"), ("n", "size_t")),
        {"predictions": ("n",), "targets": ("n",), "output": ("n",)},
        tuple((f"n{n}", {"n": n}) for n in (4_093, 65_537, 1_048_579, 8_388_611)),
        (("随机预测与标签", "预测均匀取 [-2,2)，标签独立等概率为 ±1"),
         ("正标签与正损失", "标签 +1，间隔 0.5"), ("负标签与零损失", "标签 −1，间隔 2"),
         ("阈值相等", "间隔等于 1"), ("阈值前一浮点数", "间隔等于 1−2⁻²⁴"),
         ("阈值后一浮点数", "间隔等于 1+2⁻²³"), ("逐元素交替", "间隔按元素交替取 0、2"),
         ("每 warp 交替", "间隔每 32 个元素交替取 0、2"), ("仅末项有损失", "末间隔 0，其余 2"),
         ("仅末项零损失", "末间隔 2，其余 0")),
        hinge_inputs, hinge_reference, "torch.clamp(1 - predictions * targets, min=0)"),
    "histogram": Problem(
        (("image", "const float*"), ("num_bins", "int"), ("output", "float*"), ("height", "size_t"), ("width", "size_t")),
        {"image": ("height", "width"), "output": ("num_bins",)},
        tuple((f"h{h}_w{w}_b{b}", {"height": h, "width": w, "num_bins": b})
              for h, w in ((63, 65), (511, 769), (2049, 1537)) for b in (16, 64, 512)),
        (("均匀随机", "全桶独立均匀采样"), ("单桶热点 0", "所有像素为 0"),
         ("单桶热点末桶", "所有像素为 B−1"), ("两端逐点交替", "展平后交替取 0、B−1"),
         ("95% 热点", "随机位置放置最接近 5% 的非零像素，均匀取自 [1,B−1]"),
         ("逐点轮桶", "i mod B"), ("warp 内同桶", "floor(i/32) mod B"),
         ("连续列条带", "floor(列号×B/宽)"), ("仅八的倍数桶", "从编号为 8 的倍数的桶均匀采样"),
         ("仅末像素取末桶", "末像素 B−1，其余 0")),
        histogram_inputs, histogram_reference, "合法整数像素逐桶计数，PyTorch int64 bincount 后转 FP32"),
    "leaky_relu": Problem(
        (("input", "const float*"), ("alpha", "float"), ("output", "float*"), ("n", "size_t"), ("m", "size_t")),
        {"input": ("n", "m"), "output": ("n", "m")},
        tuple((f"n{n}_m{m}_a{a}", {"n": n, "m": m, "alpha": a})
              for n, m in ((257, 513), (2049, 4097)) for a in (0.02, 0.125, 0.25, 0.5)),
        (("均匀混合正负", "独立均匀取 [-2,2)"), ("全正", "均匀取 [0.125,2)"),
         ("全负", "均匀取 [-2,−0.125)"), ("带符号零", "逐点交替取 +0 和 −0"),
         ("零阈值邻域", "重复 −2⁻⁴⁰、−2⁻²⁰、0、2⁻⁴⁰、2⁻²⁰"),
         ("逐点正负交替", "逐点交替取 −1、1"), ("每 warp 同号", "每 32 个值交替取 −1、1"),
         ("稀疏正值", "−1 背景，随机位置放置最接近 5% 的 +1"),
         ("仅末项为负", "末值 −1，其余 +1"), ("仅末项为正", "末值 +1，其余 −1")),
        leaky_inputs, leaky_reference, "PyTorch torch.where(x > 0, x, alpha * x)"),
}


def manifest(problem):
    spec = PROBLEMS[problem]
    shapes = []
    for shape_id, params in spec.shapes:
        tensors = {name: {"shape": [params[k] for k in dims], "dtype": "float32", "layout": "contiguous"}
                   for name, dims in spec.dimensions.items()}
        shapes.append({"shape_id": shape_id, "parameters": dict(params), "tensors": tensors,
                       "seeds": {cid: seed_for(problem, shape_id, cid) for cid in CASE_IDS},
                       "label_seed": seed_for(problem, shape_id, "labels") if problem == "hinge" else None})
    data = {"problem": problem, "dataset_version": f"{problem}.v1",
            "abi": [{"name": name, "type": ctype} for name, ctype in spec.abi],
            "reference": spec.reference_note, "rtol": 0, "atol": 0, "generator": "PyTorch CUDA, per-case Generator",
            "shapes": shapes, "cases": [{"case_id": cid, "name": name, "rule": rule}
                                        for cid, (name, rule) in zip(CASE_IDS, spec.cases)]}
    data["manifest_sha256"] = sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    return data


def template(problem, source_sha, kind):
    catalog = manifest(problem)
    shapes = []
    for s in catalog["shapes"]:
        cases = [{"case_id": c["case_id"], "name": c["name"], "seed": s["seeds"][c["case_id"]],
                  "status": "pending", "correct": False, "gpu_samples_us": [], "e2e_samples_us": [],
                  "median_us": None, "e2e_median_us": None, "error": None} for c in catalog["cases"]]
        shapes.append({"shape_id": s["shape_id"], "parameters": s["parameters"], "cases": cases, "mean_us": None})
    return {**{key: catalog[key] for key in ("problem", "dataset_version", "manifest_sha256")},
            "schema_version": 1, "kind": kind, "run_id": str(uuid.uuid4()),
            "started_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source_sha,
            "protocol_version": PROTOCOL_VERSION, "protocol": PROTOCOL.copy(), "shapes": shapes,
            "status": "running", "rankable": False, "total_us": None, "error": None, "failure_kind": None,
            "environment": {}, "prewarm": {}, "stage": "environment"}


def warm_cases(shapes):
    return [(s["shape_id"], cid) for s in (shapes[0], shapes[-1]) for cid in ("c01", "c07")]


def prewarm(cases, invoke, clock=time.perf_counter):
    started = clock()
    elapsed, cycles = 0.0, 0
    while elapsed < PREWARM_SECONDS or not cycles:
        for sid, cid in cases:
            invoke(sid, cid, PREWARM_CALLS)  # 原生调用在每组末尾等待所有 GPU 流。
        cycles += 1
        elapsed = clock() - started
    return {"cases": [{"shape_id": s, "case_id": c, "calls": cycles * PREWARM_CALLS} for s, c in cases],
            "elapsed_s": elapsed, "cycles": cycles, "synchronized": True, "excluded_from_score": True, "completed": True}


def verified_prewarm(record, expected):
    if not isinstance(record, dict):
        return False
    cycles = record.get("cycles")
    return (type(cycles) is int and cycles > 0 and positive(record.get("elapsed_s"))
            and record["elapsed_s"] >= PREWARM_SECONDS and record.get("synchronized") is True
            and record.get("excluded_from_score") is True and record.get("completed") is True
            and record.get("cases") == [{"shape_id": s, "case_id": c, "calls": cycles * PREWARM_CALLS}
                                         for s, c in warm_cases(expected)])


def positive(value):
    return type(value) in (float, int) and 0 < value <= sys.float_info.max


def same_number(a, b):
    return positive(a) and positive(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def summarize_shapes(expected, rows):
    """只从未舍入的逐例样本计算。expected 可为任意同协议清单。"""
    errors, means = [], {}
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return {}, None, ["形状列表无效"]
    if [r.get("shape_id") for r in rows] != [s["shape_id"] for s in expected]:
        return {}, None, ["形状缺失、重复或顺序不符"]
    for spec, row in zip(expected, rows):
        sid, cases = spec["shape_id"], row.get("cases")
        if (row.get("parameters") != spec["parameters"] or not isinstance(cases, list)
                or any(not isinstance(c, dict) for c in cases) or [c.get("case_id") for c in cases] != list(CASE_IDS)):
            errors.append(f"{sid} 形状参数或十个用例身份不符")
            continue
        medians = []
        for case in cases:
            cid = case["case_id"]
            gpu, e2e = case.get("gpu_samples_us"), case.get("e2e_samples_us")
            if (case.get("status") != "ok" or case.get("correct") is not True or case.get("error") is not None
                    or case.get("seed") != spec["seeds"][cid]
                    or not all(isinstance(samples, list) and len(samples) == SAMPLES
                               and all(positive(x) for x in samples) for samples in (gpu, e2e))):
                errors.append(f"{sid}/{cid} 校验、种子或原始计时无效")
                continue
            median = statistics.median(gpu)
            if not (same_number(case.get("median_us"), median)
                    and same_number(case.get("e2e_median_us"), statistics.median(e2e))):
                errors.append(f"{sid}/{cid} 公布中位数与原样本不符")
                continue
            medians.append(median)
        if len(medians) == 10:
            mean = statistics.fmean(medians)
            if not same_number(row.get("mean_us"), mean):
                errors.append(f"{sid} 公布平均数与原样本不符")
            else:
                means[sid] = mean
    return means, None if errors else math.fsum(means.values()), errors


def recompute(report):
    """审查原始记录及公布值，失败/对照结果不能参与候选排名。"""
    if not isinstance(report, dict) or not isinstance(report.get("problem"), str) or report["problem"] not in PROBLEMS:
        return {"rankable": False, "total_us": None, "shape_means_us": {}, "errors": ["未知题目"]}
    catalog = manifest(report["problem"])
    means, total, errors = summarize_shapes(catalog["shapes"], report.get("shapes"))
    if (type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or report.get("kind") not in KINDS.values()
            or any(report.get(k) != catalog[k] for k in ("dataset_version", "manifest_sha256"))
            or report.get("protocol_version") != PROTOCOL_VERSION or report.get("protocol") != PROTOCOL):
        errors.append("题目、数据集或计时协议身份不符")
    if report.get("kind") == KINDS["copy"] and (report["problem"] != "leaky_relu"
            or report.get("source_sha256") != sha256(COPY_SOURCE)):
        errors.append("设备复制对照来源不符")
    try:
        digest = report["source_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError
        run_id = str(report.get("run_id"))
        if uuid.UUID(run_id).version != 4 or str(uuid.UUID(run_id)) != run_id:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        errors.append("源码或独立运行身份不符")
    env = report.get("environment")
    if (not isinstance(env, dict) or any(not isinstance(env.get(k), str) or not env[k]
                for k in ("gpu_uuid", "driver_version", "torch", "cuda", "nvcc", "runner_sha256"))
            or env.get("compile_flags") != FLAGS or env.get("gpu") != "NVIDIA H20"):
        errors.append("设备或工具链证据不完整")
    if (report.get("status") != "ok" or report.get("error") is not None or report.get("failure_kind") is not None
            or not same_number(total, report.get("total_us"))
            or report.get("rankable") is not (report.get("kind") == KINDS["candidate"])):
        errors.append("整题状态或公布总和与原样本不符")
    if not verified_prewarm(report.get("prewarm"), catalog["shapes"]):
        errors.append("正式计时前独立预热未完成或记录不符")
    return {"rankable": not errors and report["kind"] == KINDS["candidate"],
            "total_us": None if errors else total, "shape_means_us": means, "errors": errors}


# 计时调用直接从 C++ 进入对应的 .so，两个 CUDA Event 之间没有 Python/ctypes 调用。
TIMER = r"""
#include <cuda_runtime.h>
#include <chrono>
#include <cstddef>
using Fn = void (*)(ARG_TYPES);
using Clock = std::chrono::steady_clock;

struct Resources {
    cudaEvent_t begin{}, end{};
    ~Resources() {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
    }
};
#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) return int(e); } while (0)

extern "C" const char* error_text(int e) { return cudaGetErrorString(cudaError_t(e)); }

extern "C" int warm(Fn fn, ARG_DECL, int repeats) {
    for (int i = 0; i < repeats; ++i) fn(ARG_NAMES);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    return 0;
}

extern "C" int measure(Fn fn, ARG_DECL, double* gpu_us, double* e2e_us, int* measured) {
    Resources state;
    CHECK(cudaEventCreate(&state.begin));
    CHECK(cudaEventCreate(&state.end));
    for (int i = 0; i < WARMUP_COUNT; ++i) fn(ARG_NAMES);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        CHECK(cudaEventRecord(state.begin, 0));
        auto host_start = Clock::now();
        fn(ARG_NAMES);
        CHECK(cudaEventRecord(state.end, 0));
        CHECK(cudaDeviceSynchronize());
        auto host_end = Clock::now();
        CHECK(cudaGetLastError());
        float ms;
        CHECK(cudaEventElapsedTime(&ms, state.begin, state.end));
        gpu_us[i] = double(ms) * 1000.0;
        e2e_us[i] = std::chrono::duration<double, std::micro>(host_end - host_start).count();
        *measured = i + 1;
    }
    return 0;
}
"""

COPY_SOURCE = b"""#include <cuda_runtime.h>
extern "C" void solution(const float* input, float, float* output, size_t n, size_t m) {
    cudaMemcpyAsync(output, input, n * m * sizeof(float), cudaMemcpyDeviceToDevice, 0);
}
"""


def timer_source(spec):
    types = ", ".join(t for _, t in spec.abi)
    decl = ", ".join(f"{t} {n}" for n, t in spec.abi)
    names = ", ".join(n for n, _ in spec.abi)
    return (TIMER.replace("ARG_TYPES", types).replace("ARG_DECL", decl).replace("ARG_NAMES", names)
            .replace("WARMUP_COUNT", str(WARMUP)).replace("SAMPLE_COUNT", str(SAMPLES))).encode()


def device_environment(torch):
    if torch.cuda.device_count() != 1:
        raise RuntimeError("评测进程须只看见一张 H20")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if props.name != "NVIDIA H20" or torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError(f"当前设备 {props.name} 与 SM90 H20 不符")
    gpu_uuid = "GPU-" + str(props.uuid).removeprefix("GPU-").lower()
    values = subprocess.check_output(["nvidia-smi", "--id=" + gpu_uuid, "--query-gpu=uuid,driver_version",
                                      "--format=csv,noheader"], text=True, timeout=15).strip().split(",")
    if len(values) != 2 or values[0].strip().lower() != gpu_uuid.lower():
        raise RuntimeError("当前 CUDA 设备和 nvidia-smi UUID 不符")
    driver = values[1].strip()
    proof_path = Path("/run/kernel-workbench-gpu.json")
    proof = json.loads(proof_path.read_bytes()) if proof_path.is_file() else None
    if proof is not None and (not isinstance(proof, dict) or proof.get("remote_ssh") is not True
            or proof.get("gpu_uuid") != gpu_uuid
            or proof.get("gpu") != props.name or proof.get("driver_version") != driver
            or not proof.get("image_digest") or not proof.get("image_id")):
        raise RuntimeError("容器内 H20 身份、镜像或驱动与宿主证据不符")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("环境缺少 nvcc")
    return device, {"gpu": props.name, "gpu_uuid": gpu_uuid, "driver_version": driver,
                    "torch": str(torch.__version__), "cuda": str(torch.version.cuda),
                    "nvcc": subprocess.check_output([nvcc, "--version"], text=True, timeout=15).strip(),
                    "compile_flags": FLAGS.copy(), "runner_sha256": sha256(Path(__file__).read_bytes()),
                    "host_attestation": proof}


def compile_library(source, directory, *, candidate):
    directory.mkdir()
    path = directory / "source.cu"
    path.write_bytes(source)
    library = directory / "compiled.so"
    command = ["nvcc", *FLAGS, str(path), "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode:
        detail = (result.stderr + result.stdout).strip()[-4000:]
        infra_errors = ("no space left", "out of memory", "unsupported gpu architecture", "cannot find -lcudart",
                        "no supported gcc", "unsupported gnu version", "killed signal", "cannot execute",
                        "cuda_runtime.h: no such file", "failed to execute")
        failure = CandidateError if candidate and not any(e in detail.lower() for e in infra_errors) else RuntimeError
        raise failure(f"nvcc 退出 {result.returncode}：{detail}")
    return ctypes.CDLL(str(library), mode=os.RTLD_LOCAL | os.RTLD_NOW)


class NativeTimer:
    def __init__(self, spec, scratch):
        self.spec = spec
        self.scratch = scratch
        self.entries = {}
        self.argtypes = [CTYPE[t] for _, t in spec.abi]
        self.lib = compile_library(timer_source(spec), scratch / "timer", candidate=False)
        pdouble, pint = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
        self.lib.measure.argtypes = [ctypes.c_void_p, *self.argtypes, pdouble, pdouble, pint]
        self.lib.measure.restype = ctypes.c_int
        self.lib.warm.argtypes = [ctypes.c_void_p, *self.argtypes, ctypes.c_int]
        self.lib.warm.restype = ctypes.c_int
        self.lib.error_text.argtypes = [ctypes.c_int]
        self.lib.error_text.restype = ctypes.c_char_p

    def load(self, sha, data):
        if sha not in self.entries:
            lib = compile_library(data, self.scratch / sha, candidate=True)
            try:
                entry = lib.solution
            except AttributeError as exc:
                raise CandidateError("候选没有导出 solution 符号") from exc
            entry.argtypes, entry.restype = self.argtypes, None
            self.entries[sha] = (lib, entry)
        return self.entries[sha][1]

    def check_cuda(self, code):
        if code:
            detail = self.lib.error_text(code).decode(errors="replace")
            if any(s in detail.lower() for s in ("illegal memory access", "misaligned address",
                    "invalid configuration argument", "device-side assert", "launch out of resources",
                    "invalid device function", "unspecified launch failure")):
                raise CandidateError(detail)
            raise RuntimeError(detail)

    def warm(self, entry, args, count):
        self.check_cuda(self.lib.warm(entry, *args, count))

    def measure(self, entry, args, case):
        gpu, e2e = (ctypes.c_double * SAMPLES)(), (ctypes.c_double * SAMPLES)()
        measured = ctypes.c_int(0)
        code = self.lib.measure(entry, *args, gpu, e2e, ctypes.byref(measured))
        case["gpu_samples_us"] = list(gpu[:measured.value])
        case["e2e_samples_us"] = list(e2e[:measured.value])
        self.check_cuda(code)
        if (measured.value != SAMPLES or not all(positive(x) for x in (*gpu, *e2e))):
            raise RuntimeError("GPU 或端到端计时缺失或无效")
        case["median_us"], case["e2e_median_us"] = statistics.median(gpu), statistics.median(e2e)


def abi_args(spec, params, inputs, out):
    pointers = iter(t.data_ptr() for t in inputs)
    return tuple(next(pointers) if t == "const float*" else out.data_ptr() if t == "float*" else params[name]
                 for name, t in spec.abi)


def check_equal(actual, expected, stage):
    if not actual.equal(expected):
        # 只在错误路径定位首个差异，避免为大张量创建所有错误元素的索引。
        a, e = actual.view(-1), expected.view(-1)
        idx = (a != e).byte().argmax().item()
        raise WrongAnswer(f"{stage}：元素 {idx} 实际 {a[idx].item()}，期望 {e[idx].item()}")


class Dataset:
    def __init__(self, torch, device, problem, copy):
        self.torch, self.device = torch, device
        self.spec = PROBLEMS[problem]
        self.default = torch.cuda.default_stream(device)
        self.producer = torch.cuda.Stream(device=device)
        if self.default.cuda_stream != 0 or self.producer.cuda_stream == self.default.cuda_stream:
            raise RuntimeError("需要 legacy 默认调用流和独立生产流")
        self.buffers, self.data = {}, {}
        with torch.cuda.stream(self.default):
            for sid, params in self.spec.shapes:
                for i, cid in enumerate(CASE_IDS, 1):
                    inputs = self.spec.generate(torch, device, params, i, seed_for(problem, sid, cid),
                                                seed_for(problem, sid, "labels"))
                    expected = inputs[0] if copy else self.spec.reference(torch, inputs, params)
                    self.data[sid, cid] = (inputs, expected)
                active = tuple(t.clone() for t in self.data[sid, "c02"][0])
                out = torch.empty_like(expected)
                self.buffers[sid] = (active, out, torch.empty_like(out), abi_args(self.spec, params, active, out))
        torch.cuda.synchronize(device)

    def prewarm(self, timer, entry, shapes):
        cases = warm_cases(shapes)
        # 每次预热使用自己的可写缓冲；模板与参考始终只在驱动侧持有。
        keep, args = [], {}
        for sid, cid in cases:
            frozen, expected = self.data[sid, cid]
            inputs = tuple(t.clone() for t in frozen)
            out = self.torch.empty_like(expected)
            keep.append((inputs, out))
            args[sid, cid] = abi_args(self.spec, dict(self.spec.shapes)[sid], inputs, out)
        self.torch.cuda.synchronize(self.device)
        return prewarm(cases, lambda s, c, n: timer.warm(entry, args[s, c], n))

    def validate(self, entry, sid, cid):
        torch = self.torch
        inputs, expected = self.data[sid, cid]
        active, out, observed, args = self.buffers[sid]
        out.fill_(float("nan"))
        self.default.synchronize()
        for delay in POISON_DELAY_CYCLES:
            ready = torch.cuda.Event()
            with torch.cuda.stream(self.producer):
                torch.cuda._sleep(delay)
                for buf, src in zip(active, inputs):
                    buf.copy_(src)
                ready.record(self.producer)
            self.default.wait_event(ready)
            if not ready.query():
                break
        else:
            raise RuntimeError("生产流已提前完成，本次无法证明调用流等候未完成输入")
        entry(*args)
        observed.copy_(out)  # 调用流上的下一条 GPU 操作，先取得快照再全设备同步。
        torch.cuda.synchronize(self.device)
        check_equal(observed, expected, "调用流立即消费的完整输出")
        check_equal(out, expected, "同步后的完整输出")
        for buf, src in zip(active, inputs):
            check_equal(buf, src, "候选改写只读输入")

    def measure_case(self, timer, entry, sid, case):
        cid = case["case_id"]
        inputs, expected = self.data[sid, cid]
        alt = "c02" if expected.equal(self.data[sid, "c03"][1]) else "c03"
        if alt == "c02" and expected.equal(self.data[sid, alt][1]):
            raise RuntimeError("同址回归缺少输出不同的第二份输入")
        case["same_address_probe"] = alt
        self.validate(entry, sid, alt)
        self.validate(entry, sid, cid)
        self.validate(entry, sid, cid)  # 输入不变但输出已再次污染，覆盖跨调用复用输出。
        active, out, _, args = self.buffers[sid]
        timer.measure(entry, args, case)
        check_equal(out, expected, "连续计时后的完整输出")
        for buf, src in zip(active, inputs):
            check_equal(buf, src, "计时期间候选改写只读输入")
        case.update(status="ok", correct=True)


def write_json(path, data):
    staging = path.with_name(path.name + ".partial")
    staging.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n",
                       encoding="utf-8")
    os.replace(staging, path)


def fail(report, exc):
    candidate_cuda_errors = ("illegal memory access", "misaligned address", "invalid configuration argument",
                             "device-side assert", "launch out of resources", "invalid device function")
    candidate_error = isinstance(exc, CandidateError) or any(s in str(exc).lower() for s in candidate_cuda_errors)
    report.update(status="failed", rankable=False, total_us=None,
                  failure_kind="candidate" if candidate_error else "infrastructure",
                  error=f"{type(exc).__name__}: {exc}")


def run_leg(report, dataset, timer, entry, save):
    report["stage"] = "prewarm"
    save()
    report["prewarm"] = dataset.prewarm(timer, entry, report["shapes"])
    save()
    wrong = []
    for shape in report["shapes"]:
        sid = shape["shape_id"]
        for case in shape["cases"]:
            report["stage"] = f"{sid}/{case['case_id']}"
            case["status"] = "running"
            save()
            try:
                dataset.measure_case(timer, entry, sid, case)
            except Exception as exc:
                case.update(status="failed", correct=False, error=f"{type(exc).__name__}: {exc}")
                if not isinstance(exc, WrongAnswer):
                    raise
                wrong.append(f"{report['stage']} {exc}")
        if all(c["status"] == "ok" for c in shape["cases"]):
            shape["mean_us"] = statistics.fmean(c["median_us"] for c in shape["cases"])
        save()
    if wrong:
        raise WrongAnswer(f"{len(wrong)} 个用例出错，首处 {wrong[0]}")
    report.update(stage="complete", status="ok", rankable=report["kind"] == KINDS["candidate"],
                  total_us=math.fsum(s["mean_us"] for s in report["shapes"]))
    audit = recompute(report)
    if audit["errors"]:
        raise RuntimeError("原始数据复核未通过：" + "，".join(audit["errors"][:3]))


def evaluate(problem, sources, order, path, *, batch=False, copy=False):
    kind = KINDS["copy"] if copy else KINDS["candidate"]
    payloads = [(sha256(data), data) for data in sources]
    group = {"kind": "local_cuda_batch", "problem": problem, "order": order, "reports": [], "status": "running",
             "source_sha256s": [sha for sha, data in payloads], "error": None, "failure_kind": None}
    # 先写候选身份及用例表。进程在编译、CUDA 调用或外部超时处中止时，已完成的腿可单独归档。
    first = template(problem, payloads[order[0]][0], kind)
    group["reports"].append(first)

    def save():
        write_json(path, group if batch else group["reports"][-1])

    save()
    current = first
    try:
        import torch
        device, env = device_environment(torch)
        with TemporaryDirectory(dir=path.parent, prefix="native-eval-") as tmp, torch.no_grad(), torch.cuda.stream(torch.cuda.default_stream(device)):
            current["environment"] = env
            current["stage"] = "compile_timer"
            save()
            timer = NativeTimer(PROBLEMS[problem], Path(tmp))
            dataset = None
            for leg, index in enumerate(order):
                digest, data = payloads[index]
                if leg:
                    current = template(problem, digest, kind)
                    current["environment"] = env
                    group["reports"].append(current)
                current["stage"] = "compile_candidate"
                save()
                entry = timer.load(digest, data)
                if dataset is None:
                    current["stage"] = "prepare_all_inputs"
                    save()
                    dataset = Dataset(torch, device, problem, copy)
                run_leg(current, dataset, timer, entry, save)
                save()
        group["status"] = "ok"
    except Exception as exc:
        fail(current, exc)
        group.update(status="failed", error=current["error"], failure_kind=current["failure_kind"])
    save()
    return group if batch else current


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("manifest", "run", "batch", "copy"):
        p = sub.add_parser(name)
        p.add_argument("--problem", required=True, choices=("leaky_relu",) if name == "copy" else tuple(PROBLEMS))
        if name in ("run", "batch"):
            p.add_argument("--source", type=Path, required=True, action="append" if name == "batch" else "store")
        if name == "batch":
            p.add_argument("--order", required=True, help="按 0 起始的源码序号，如 0,1,1,0")
        if name != "manifest":
            p.add_argument("--output", type=Path, required=True)
    sub.add_parser("recompute").add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "manifest":
        print(json.dumps(manifest(args.problem), ensure_ascii=False, indent=2))
        return 0
    if args.command == "recompute":
        data = json.loads(args.report.read_bytes())
        result = recompute(data)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(bool(result["errors"]))
    if args.output.exists():
        parser.error("输出已存在，请为新测量指定独立文件")
    if args.command == "batch":
        try:
            order = [int(x) for x in args.order.split(",")]
        except ValueError:
            parser.error("--order 需要逗号分隔的源码序号")
        if any(i < 0 or i >= len(args.source) for i in order):
            parser.error("--order 引用了不存在的源码")
        files = args.source
    else:
        order, files = [0], [] if args.command == "copy" else [args.source]
    for source in files:
        if source.suffix != ".cu":
            parser.error("本地原生评测接受 CUDA C++ .cu 单文件")
    data = [f.read_bytes() for f in files] if files else [COPY_SOURCE]
    report = evaluate(args.problem, data, order, args.output.resolve(), batch=args.command == "batch", copy=args.command == "copy")
    return int(report["status"] != "ok")


if __name__ == "__main__":
    sys.exit(main())
