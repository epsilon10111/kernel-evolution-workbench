"""在指定 H20 的同一次独占租用中完成交错评测。入口只消费本地队列。"""

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from .local_campaign import DeviceUnavailable

GPU_UUID = r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
IMAGE_DIGEST = r"[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}"


def _connection():
    host = os.environ.get("KEW_SSH_HOST", "")
    gpu_uuid = os.environ.get("KEW_GPU_UUID", "")
    image = os.environ.get("KEW_CUDA_IMAGE", "")
    if not re.fullmatch(r"[\w.][\w.@:\-]*", host, re.ASCII):
        raise ValueError("Set KEW_SSH_HOST to a configured SSH alias or user@host")
    if not re.fullmatch(GPU_UUID, gpu_uuid):
        raise ValueError("Set KEW_GPU_UUID to the target GPU UUID from nvidia-smi -L")
    if not re.fullmatch(IMAGE_DIGEST, image):
        raise ValueError("Set KEW_CUDA_IMAGE to an image pinned by SHA-256 digest")
    return host, gpu_uuid, image


def _diagnostic(stderr):
    return stderr.decode("utf-8", errors="replace").strip()[-600:]


def _archive(files):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name, data in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            item.mode = 0o444
            archive.addfile(item, io.BytesIO(data))
    return payload.getvalue()


def _read_source(path, expected):
    path = Path(path)
    if path.suffix != ".cu" or not 0 < path.stat().st_size <= 512 * 1024:
        raise ValueError("需要 512 KiB 以内的完整 CUDA 源码")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("候选文件已变化，源码 SHA 与入队记录不同")
    return data


def _run_remote(problem, sources, order, *, copy=False):
    host, gpu_uuid, image = _connection()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", problem):
        raise ValueError("题目标识无效")
    files = {"local_eval.py": Path(__file__).with_name("local_eval.py").read_bytes()}
    files.update({f"candidate-{i}.cu": source for i, source in enumerate(sources)})
    if copy:
        args = ["copy"]
    else:
        if not 1 <= len(order) <= 4 or not all(0 <= i < len(sources) <= 4 for i in order):
            raise ValueError("一组独占评测需包含 1–4 腿")
        args = ["batch", str(len(sources)), ",".join(map(str, order))]
    script = Path(__file__).with_name("h20_gate.sh").read_text()
    command = ["ssh"]
    if config := os.environ.get("KEW_SSH_CONFIG"):
        command.extend(["-F", config])
    command += ["-o", "BatchMode=yes",
               "-o", "StrictHostKeyChecking=yes", "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes",
               "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3",
               "--", host, shlex.join(["bash", "-c", script, "kernel-workbench", gpu_uuid, image, problem, *args])]
    try:
        proc = subprocess.run(command, input=_archive(files), capture_output=True, timeout=950)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise DeviceUnavailable(f"H20 连接中断：{exc}") from exc
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        raise DeviceUnavailable(f"H20 返回 {proc.returncode}，{_diagnostic(proc.stderr)}")
    try:
        packet = json.loads(proc.stdout)
    except (ValueError, UnicodeError) as exc:
        raise DeviceUnavailable(f"H20 返回了不完整的报告，{_diagnostic(proc.stderr)}") from exc
    return packet


def _verify_device(report):
    _, gpu_uuid, image = _connection()
    env = report.get("environment")
    attestation = env.get("host_attestation") if isinstance(env, dict) else None
    if not (isinstance(attestation, dict) and attestation.get("remote_ssh") is True
            and env.get("gpu_uuid") == attestation.get("gpu_uuid") == gpu_uuid
            and env.get("driver_version") == attestation.get("driver_version")
            and attestation.get("gpu") == "NVIDIA H20" and attestation.get("image_digest") == image
            and attestation.get("image_id")):
        raise DeviceUnavailable("报告缺少本次 H20、CUDA 镜像或驱动的对应记录")


def evaluate_group(requests):
    """返回每腿独立报告。失去设备时交给 campaign 重试整组。"""
    if not requests:
        raise ValueError("空的评测组")
    problem = requests[0]["problem"]
    sources, positions, order = [], {}, []
    for req in requests:
        if req["problem"] != problem:
            raise ValueError("同一测量组含有不同题目")
        sha = req["source_sha256"]
        if sha not in positions:
            positions[sha] = len(sources)
            sources.append(_read_source(req["source_path"], sha))
        order.append(positions[sha])
    packet = _run_remote(problem, sources, order)
    if not isinstance(packet, dict):
        raise DeviceUnavailable("H20 批次报告格式无效")
    if packet.get("kind") != "local_cuda_batch" or packet.get("problem") != problem:
        raise DeviceUnavailable(f"H20 批次报告与请求不符：{str(packet.get('error') or '')[:500]}")
    reports = packet.get("reports", [])
    if len(reports) > len(requests):
        raise DeviceUnavailable("H20 返回了超出本组的测量")
    seen = set()
    for req, report in zip(requests, reports):
        if report.get("failure_kind") == "infrastructure":
            raise DeviceUnavailable(str(report.get("error") or "H20 基础设施失败")[:800])
        if (report.get("kind") != "local_cuda_evaluation"
                or report.get("problem") != problem or report.get("source_sha256") != req["source_sha256"]
                or not report.get("run_id") or report["run_id"] in seen):
            raise DeviceUnavailable("H20 的版本、题目或独立测量标识与本腿不符")
        _verify_device(report)
        seen.add(report["run_id"])
        yield report
        if not report.get("rankable"):
            return
    if len(reports) != len(requests):
        raise DeviceUnavailable(f"整组仅收回 {len(reports)} 腿：{str(packet.get('error') or '')[:500]}")


def copy_control(problem):
    report = _run_remote(problem, [], [], copy=True)
    if (report.get("kind") != "local_cuda_copy_control" or report.get("problem") != problem
            or report.get("rankable") or not report.get("total_us")):
        raise RuntimeError(f"拷贝对照未完成：{report.get('error')}")
    _verify_device(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("drain", help="依次消费已排队的本地交错测量组")
    worker.add_argument("--root", required=True, type=Path)
    worker.add_argument("--watch", action="store_true", help="空队列时也继续等待新候选")
    worker.add_argument("--poll", type=float, default=5.0)
    copy = commands.add_parser("copy", help="记录同条件 D2D 拷贝对照")
    copy.add_argument("--problem", default="leaky_relu")
    copy.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    _connection()
    if args.command == "copy":
        if args.output.exists():
            parser.error("对照报告已存在，请指定新的文件名")
        result = copy_control(args.problem)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, allow_nan=False)
        print(f"拷贝对照 {result['total_us']:.3f} µs")
        return
    if not 0 < args.poll <= 60:
        parser.error("轮询间隔须在 0 至 60 秒内")
    from .local_campaign import Campaign
    campaign = Campaign(args.root)
    last = None
    while True:
        result = campaign.run_next(evaluate_group=evaluate_group)
        state = result["status"]
        if state != last or state == "completed":
            print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
        if state == "idle" and not args.watch:
            return
        if state in {"blocked", "idle", "busy"}:
            time.sleep(args.poll)
        elif state != "completed":
            raise RuntimeError(f"未知队列状态：{state}")
        last = state


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        sys.exit(str(exc))
