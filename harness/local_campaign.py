"""单题三席开发和本地交错测量队列。运行目录包含可恢复状态、不可变源码与逐腿原始报告。"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median

from . import workers


AGENTS = ("a", "b", "c")
MODEL = "gpt-6.2-astra"
ENV_KEYS = ("gpu_uuid", "driver_version", "torch", "cuda", "nvcc", "compile_flags", "runner_sha256")


class DeviceUnavailable(RuntimeError):
    """设备、远端运行器或协议暂不可用。保留队首，下一次重新测量整组。"""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _atomic(path, data):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", delete=False) as file:
        tmp = Path(file.name)
        try:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
            os.replace(tmp, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            tmp.unlink(missing_ok=True)


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 需要非空文字")
    return value.strip()


def _source(path):
    if isinstance(path, bytes):
        data = path
    else:
        with Path(path).open("rb") as file:
            data = file.read(524289)
    if not 0 < len(data) <= 524288:
        raise ValueError("完整源码需要 1 至 524288 字节")
    code = data.decode("utf-8").strip()
    if (not code or "\0" in code or code.startswith(("\x60\x60\x60", "~~~", "diff --git ", "*** Begin Patch", "@@ "))
            or (code.startswith("--- ") and "\n+++ " in code)):
        raise ValueError("需要完整的 UTF-8 源文件")
    return data


def _proc_start(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, IndexError, PermissionError):
        return None


def _device(state, status, reason, *, running=False):
    state["device"] = dict(state=status, reason=reason, last_activity=_now(),
                           pid=os.getpid() if running else None, proc_start=_proc_start(os.getpid()) if running else None)


def _seat(state, agent, status, detail, **fields):
    slot = state["agents"][agent]
    slot.update(state=status, detail=detail, last_activity=_now(), model_phase=None, model_error=None, **fields)
    if status not in ("developing", "evaluating"):
        slot.update(pid=None, proc_start=None)


def _active(state, agent, round_id=None):
    if agent not in AGENTS:
        raise ValueError("未知 Agent")
    current = state["agents"][agent]["round_id"]
    if current is None or round_id is not None and round_id != current:
        raise ValueError("本操作与该席位当前轮次不符")
    return state["rounds"][current]


def _metrics(row):
    return dict(total_us=row["total_us"], **row["shape_means_us"])


def _conditions(raw):
    env = raw["environment"]
    signature = {k: env[k] for k in ENV_KEYS}
    signature["protocol"] = raw["protocol"]
    attestation = env.get("host_attestation")
    if isinstance(attestation, dict):
        signature["host"] = {k: attestation[k] for k in ("image_digest", "image_id") if k in attestation}
    return signature


def _archived_conditions_match(state, expected):
    from .local_eval import recompute
    old = state["conditions"]
    if "host" in old or "host" not in expected or old != {k: v for k, v in expected.items() if k != "host"}:
        return False
    accepted = [r for r in state["reports"] if r["rankable"]]
    for row in accepted:
        try:
            encoded = Path(row["report"]).read_bytes()
            if _sha(encoded) != row["report_sha256"]:
                return False
            raw = json.loads(encoded)
            if ((raw["run_id"], raw["source_sha256"]) != (row["run_id"], row["source_sha256"])
                    or not recompute(raw)["rankable"] or _conditions(raw) != expected):
                return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
    return bool(accepted)


def _result(row):
    return {key: row[key] for key in ("source_sha256", "source", "report", "total_us", "shape_means_us", "id")}


def _eligible_parents(state):
    failed = {row["candidate_sha"] for row in state["rounds"].values() if row["decision"] == "invalid"}
    return list(dict.fromkeys(r["source_sha256"] for r in state["reports"] if r["rankable"] and r["source_sha256"] not in failed))


def _source_measurements(reports):
    by_source = {}
    for report in reports:
        by_source.setdefault(report["source_sha256"], []).append(report)
    summary = {}
    for sha, rows in by_source.items():
        valid = [r for r in rows if r["rankable"]]
        summary[sha] = dict(
            rankable_legs=len(valid),
            median_total_us=median(r["total_us"] for r in valid) if valid else None,
            median_shape_means_us={shape: median(r["shape_means_us"][shape] for r in valid)
                                   for shape in valid[0]["shape_means_us"]} if valid else {},
            failures=[dict(id=r["id"], error=r["error"]) for r in rows if not r["rankable"]],
        )
    return summary


def _new_group(state, control, candidate, repetition, round_id=None):
    group_id = f"g{len(state['groups']) + 1:05d}"
    orders = state.get("comparison_orders", ["ABBA"])
    group = dict(id=group_id, control_sha=control, candidate_sha=candidate, repetition=repetition,
                 order=orders[(repetition - 1) % len(orders)],
                 round_id=round_id, kind="compare" if round_id else "calibrate", status="queued",
                 attempt=0, report_ids=[], gains_us=None, reason="")
    state["groups"][group_id] = group
    return group_id


def _compare(state, row, control):
    groups = []
    for repetition in (1, 2):
        existing = next((g for g in state["groups"].values() if g["kind"] == "compare"
                         and (g["control_sha"], g["candidate_sha"], g["repetition"])
                         == (control, row["candidate_sha"], repetition)), None)
        groups.append(existing["id"] if existing and not row["remeasure"]
                      else _new_group(state, control, row["candidate_sha"], repetition, row["id"]))
    row["comparisons"].append(dict(control_sha=control, group_ids=groups, status="pending"))
    _seat(state, row["agent"], "queued", "等待单卡队列完成两组交错复测")


def _finish_round(state, row, decision, detail):
    row.update(decision=decision, conclusion=detail, evaluated_at=_now())
    _seat(state, row["agent"], "reviewing", detail, report=row.get("report"))


def _advance(state):
    """在同一状态事务中结算已齐的组，并与刚更新的整题最佳衔接。"""
    if state["noise_band"] is None:
        calibration = [g for g in state["groups"].values() if g["kind"] == "calibrate"]
        if not all(g["status"] == "completed" for g in calibration):
            return
        state["noise_band"] = {key: max(abs(g["gains_us"][key]) for g in calibration)
                               for key in calibration[0]["gains_us"]}
    band = state["noise_band"]
    for row in state["rounds"].values():
        if row["decision"] is not None or not row["candidate_sha"]:
            continue
        if not row["comparisons"]:
            if row["candidate_sha"] == row["parent_sha"] and not row["remeasure"]:
                prior = next(r for r in reversed(state["reports"]) if r["source_sha256"] == row["candidate_sha"] and r["rankable"])
                row.update(report=prior["report"], report_ids=[prior["id"]])
                _finish_round(state, row, "duplicate", "源码与已通过全套的父版一致，共享既有报告")
                continue
            _compare(state, row, row["parent_sha"])
        while not row["decision"]:
            comparison = row["comparisons"][-1]
            groups = [state["groups"][gid] for gid in comparison["group_ids"]]
            failed = next((g for g in groups if g["status"] == "failed"), None)
            if failed:
                evidence = [r for r in state["reports"] if r["group_id"] in comparison["group_ids"]
                            and r["source_sha256"] == row["candidate_sha"] and not r["rankable"]]
                if evidence:
                    row["report"] = evidence[-1]["report"]
                    row["report_ids"] = list(dict.fromkeys([*row["report_ids"], *(r["id"] for r in evidence)]))
                comparison.update(status="failed", reason=failed["reason"])
                _finish_round(state, row, "invalid", failed["reason"])
                break
            if not all(g["status"] == "completed" for g in groups):
                break
            control = comparison["control_sha"]
            gains = {key: [g["gains_us"][key] for g in groups] for key in band}
            wins = [shape for shape in state["shape_ids"] if all(v > band[shape] for v in gains[shape])]
            confirmed = all(v > band["total_us"] for v in gains["total_us"])
            comparison.update(status="completed", gains_us=gains, shape_wins=wins, whole_win=confirmed)
            report_ids = [rid for g in groups for rid in g["report_ids"]]
            row["report_ids"] = list(dict.fromkeys([*row["report_ids"], *report_ids]))
            sample = next(r for r in reversed(state["reports"]) if r["id"] in report_ids
                          and r["source_sha256"] == row["candidate_sha"])
            row["report"] = sample["report"]
            for shape in wins:
                identity = (shape, row["candidate_sha"], control)
                if not any((r["shape_id"], r["source_sha256"], r["control_sha"]) == identity
                           for r in state["shape_routes"]):
                    state["shape_routes"].append(dict(shape_id=shape, source_sha256=row["candidate_sha"],
                        control_sha=control, gains_us=gains[shape], band_us=band[shape], round_id=row["id"],
                        round_agent=row["agent"], group_ids=comparison["group_ids"], report=sample["report"]))
            best = state["best"]["source_sha256"]
            if row["candidate_sha"] == best:
                _finish_round(state, row, "shared_best", "该完整源码已是当前整题最佳，共享配对证据")
            elif control != best and (confirmed or control == row["parent_sha"] and control != row.get("best_at_submit", control)):
                _compare(state, row, best)
            elif confirmed:
                state["best"] = dict(_result(sample), round_id=row["id"], agent=row["agent"],
                                     group_ids=comparison["group_ids"])
                _finish_round(state, row, "promoted", "相对当前整题最佳的两组总收益均超过自比波动带")
            elif any(c.get("shape_wins") for c in row["comparisons"]):
                _finish_round(state, row, "shape_only", "保留已复现的逐形状收益与对应对照源码")
            elif all(all(v < -band[s] for v in gains[s]) for s in row["card"]["target_cases"]):
                _finish_round(state, row, "regressed", "预期形状在两组测量中均显著变慢，其他形状也无已复现收益")
            else:
                _finish_round(state, row, "indistinguishable", "两组比较未能将收益与自比波动区分")


def _merge_experience(state, row, entry_id, experience):
    if not re.fullmatch(r"[\w.-]{1,96}", entry_id):
        raise ValueError("经验条目标识需要 1 至 96 个字母、数字、点、短横线或下划线")
    report = row["report"]
    if not report:
        raise ValueError("经验需要关联本轮源码和本地报告")
    path = Path(state["experience_path"])
    identity = _sha(f"{state['campaign_id']}/{row['id']}/{entry_id}".encode())
    marker = f"<!-- local-campaign {identity} "
    stamp = marker + _sha(experience.encode()) + " -->"
    with path.with_name("." + path.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = path.read_text(encoding="utf-8") if path.exists() else f"# {state['problem']} 共享经验\n"
        if marker in original:
            if stamp not in original:
                raise ValueError("同一经验条目标识的内容不同，请用新标识勘误")
            return
        link = os.path.relpath(row.get("round_report") or report, path.parent)
        block = (f"\n\n{stamp}\n### {row['agent']} 第 {row['round']} 轮 · {entry_id}\n\n{experience}\n\n"
                 f"候选 {row['candidate_sha']}，父版 {row['parent_sha']}。[轮次与逐腿证据](<{link}>)\n")
        _atomic(path, (original.rstrip() + block).encode())


class Campaign:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / "state.json"

    @contextmanager
    def _edit(self, *, create=False):
        if create:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (self.root / "state.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            before = None if create and not self.path.exists() else self.path.read_bytes()
            state = {} if before is None else json.loads(before)
            yield state
            for row in state["rounds"].values():
                if row["decision"] and (not row.get("round_report") or row.get("published_review") != row.get("reviewed_at")):
                    row["round_report"] = str(Path(row["workspace"]) / "round.json")
                    row["published_review"] = row.get("reviewed_at")
                    ids = set(row["report_ids"])
                    record = dict(row, problem=state["problem"], dataset_version=state["dataset_version"],
                                  protocol_version=state["protocol_version"], conditions=state["conditions"],
                                  noise_band=state["noise_band"], reports=[r for r in state["reports"] if r["id"] in ids])
                    _atomic(Path(row["round_report"]), _json_bytes(record))
            after = _json_bytes(state)
            if after != before:
                _atomic(self.path, after)

    def _source_path(self, sha):
        return self.root / "sources" / sha / "solution.cu"

    def _store_source(self, data):
        sha = _sha(data)
        path = self._source_path(sha)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError(f"已归档源码被修改：{sha}")
        else:
            _atomic(path, data)
        return sha, path

    def init(self, problem, baseline, *, experience=None, language="CUDA C++", contract=""):
        from .local_eval import PROTOCOL_VERSION, manifest
        try:
            frozen = manifest(problem)
        except KeyError as exc:
            raise ValueError(f"没有冻结的本地题目清单：{problem}") from exc
        data = _source(baseline)
        if language != "CUDA C++":
            raise ValueError("当前本地评测器接受完整的 CUDA C++ .cu 源文件")
        exp = Path(experience).resolve() if experience else self.root.parent / "EXPERIENCE.md"
        with self._edit(create=True) as state:
            config = dict(problem=frozen["problem"], baseline_sha=_sha(data), language=language, contract=contract,
                          dataset_version=frozen["dataset_version"], manifest_sha256=frozen["manifest_sha256"],
                          protocol_version=PROTOCOL_VERSION,
                          shape_ids=[s["shape_id"] for s in frozen["shapes"]], experience_path=str(exp))
            if state:
                if any(state.get(key) != value for key, value in config.items()):
                    raise ValueError("目录已绑定另一配置；新的基线或评测集请使用独立运行目录")
                self._store_source(data)
                return state["baseline_sha"]
            state.update(config, schema_version=1, campaign_id=str(uuid.uuid4()), root=str(self.root),
                         comparison_orders=["ABBA", "BAAB"],
                         rounds={}, agents={}, groups={}, reports=[], shape_routes=[], stalled=[], conditions=None,
                         best=None, noise_band=None, created_at=_now())
            sha, _ = self._store_source(data)
            for agent in AGENTS:
                state["agents"][agent] = dict(agent=agent, round=None, round_id=None, state="idle", parent_sha=sha,
                    candidate_sha=None, hypothesis=None, detail="等待选择父版与机制", last_activity=_now(),
                    report=None, pid=None, proc_start=None, source=None)
            for repeat in range(1, 4):
                _new_group(state, sha, sha, repeat)
            _device(state, "idle", "等候基线全套与三组自比")
            exp.parent.mkdir(parents=True, exist_ok=True)
            with exp.with_name("." + exp.name + ".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not exp.exists():
                    _atomic(exp, f"# {state['problem']} 共享经验\n".encode())
        return sha

    def set_round_target(self, rounds):
        """记录本次有限迭代的每席累计轮数，供看板判断整场是否结束。"""
        if type(rounds) is not int or rounds < 1:
            raise ValueError("目标轮次必须为正整数")
        with self._edit() as state:
            if rounds >= max(s["round"] or 0 for s in state["agents"].values()):
                state["round_target"] = rounds
            return state.get("round_target")

    def begin(self, agent, card, *, parent_sha=None, borrowed_from="", changed_premise="", remeasure=False, worker_id=None):
        with self._edit() as state:
            if agent not in AGENTS:
                raise ValueError("未知 Agent")
            if isinstance(card, dict) and isinstance(card.get("target_cases"), list):
                card = dict(card, target_cases=[f"n{x}" if type(x) is int and f"n{x}" in state["shape_ids"] else x
                                                 for x in card["target_cases"]])
            workers._check_card(card, set(state["shape_ids"]))
            span, param = card.get("parameter_range", []), card.get("parameter")
            if (not isinstance(span, list) or any(not isinstance(x, str) or not x.strip() for x in span)
                    or len(span) != len(set(span)) or (span and param not in span) or (not span and param is not None)):
                raise ValueError("parameter_range 需列出互异配置，parameter 指认当前完整源码的配置")
            slot = state["agents"][agent]
            parent = parent_sha or slot["parent_sha"]
            if slot["state"] in ("developing", "queued", "evaluating", "reviewing"):
                raise ValueError(f"{agent} 当前轮次仍在{slot['state']}，可从原轮次续做")
            if parent not in _eligible_parents(state):
                raise ValueError("父版需要同条件的本地全套通过报告")
            family = _text(card["family"], "family").casefold()
            stalled = [s for s in state["stalled"] if s["active"] and s["parent_sha"] == parent and s["family"] == family]
            if stalled and not changed_premise.strip():
                raise ValueError("这一父版与机制已停滞，重开需写明新证据或改变的前提")
            for item in stalled:
                item.update(active=False, reopened_with=changed_premise.strip())
            number = (slot["round"] or 0) + 1
            rid = f"{agent}-r{number:04d}"
            work = self.root / "agents" / agent / f"round-{number:04d}"
            parent_path = self._source_path(parent)
            code = _source(parent_path)
            if _sha(code) != parent:
                raise ValueError("父版归档源码指纹不符")
            _atomic(work / ("parent" + parent_path.suffix), code)
            row = dict(id=rid, round=number, agent=agent, parent_sha=parent, candidate_sha=None, source=None,
                       workspace=str(work), card=card, borrowed_from=borrowed_from, changed_premise=changed_premise,
                       remeasure=bool(remeasure), created_at=_now(), summary="", comparisons=[], decision=None,
                       conclusion="", report=None, report_ids=[], review=None,
                       worker=dict(backend="collaboration", model=MODEL, reasoning_effort="max",
                                   agent_id=_text(worker_id, "外部模型席位 ID")) if worker_id else None)
            state["rounds"][rid] = row
            _seat(state, agent, "developing", "按已公布的假设开发完整源码", round=number, round_id=rid,
                  parent_sha=parent, candidate_sha=None, hypothesis=card["change"], report=None, source=None,
                  pid=None, proc_start=None)
            return row

    def submit(self, agent, source, *, summary="", round_id=None):
        data = _source(source)
        with self._edit() as state:
            row = _active(state, agent, round_id)
            if row["candidate_sha"] == _sha(data):
                return row
            if state["agents"][agent]["state"] != "developing" or row["candidate_sha"]:
                raise ValueError("该轮已提交了另一份完整源码")
            sha, path = self._store_source(data)
            _atomic(Path(row["workspace"]) / path.name, data)
            row.update(candidate_sha=sha, source=str(path), summary=summary, submitted_at=_now(),
                       best_at_submit=state["best"]["source_sha256"])
            _seat(state, agent, "queued", "等待基线校准或单卡评测", candidate_sha=sha, source=str(path))
            _advance(state)
            return row

    def _claim(self):
        from .local_eval import PROTOCOL_VERSION, manifest
        with self._edit() as state:
            group = next((g for g in state["groups"].values() if g["status"] in ("queued", "running")), None)
            if group is None:
                return None
            if (state["protocol_version"] != PROTOCOL_VERSION
                    or manifest(state["problem"])["manifest_sha256"] != state["manifest_sha256"]):
                reason = "冻结数据或协议已升级；请使用新运行目录重新校准"
                _device(state, "blocked", reason)
                return dict(status="blocked", reason=reason, group_id=group["id"])
            group.update(status="running", attempt=group["attempt"] + 1, report_ids=[], reason="", started_at=_now())
            _device(state, "busy", f"正在测量 {group['id']}，第 {group['attempt']} 次尝试", running=True)
            rid = group["round_id"]
            agent = state["rounds"][rid]["agent"] if rid else None
            order = group.get("order", "ABBA")
            if agent:
                _seat(state, agent, "evaluating", f"{group['id']} · {order} 第 {group['repetition']} 组",
                      pid=os.getpid(), proc_start=_proc_start(os.getpid()))
            requests = []
            for index, side in enumerate(order):
                sha = group["control_sha"] if side == "A" else group["candidate_sha"]
                path = self._source_path(sha)
                report_name = f"{group['id']}-a{group['attempt']:03d}-l{index}.json"
                report_path = self.root / "reports" / report_name
                report_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                requests.append(dict(problem=state["problem"], language=state["language"], source_path=str(path),
                    source_sha256=sha, report_path=str(report_path), group_id=group["id"], attempt=group["attempt"],
                    leg_index=index, side=side, agent=agent, round_id=rid, kind=group["kind"],
                    dataset_version=state["dataset_version"], manifest_sha256=state["manifest_sha256"],
                    protocol_version=state["protocol_version"]))
            return requests

    def _record(self, request, raw):
        from .local_eval import recompute
        if isinstance(raw, (str, Path)):
            encoded = Path(raw).read_bytes()
            raw = json.loads(encoded)
        else:
            encoded = _json_bytes(raw)
        if not isinstance(raw, dict):
            raise DeviceUnavailable("远端响应缺少评测报告对象")
        path = Path(request["report_path"])
        _atomic(path, encoded)
        with self._edit() as state:
            group = state["groups"][request["group_id"]]
            key = raw.get("run_id")
            reason = ""
            try:
                if uuid.UUID(str(key)).version != 4 or str(uuid.UUID(str(key))) != key:
                    raise ValueError
            except ValueError:
                reason = "报告缺少独立测量的 UUIDv4"
            if key and any(r["run_id"] == key for r in state["reports"]):
                reason = "重复的测量 run_id 不能充当独立测量腿"
            if any(raw.get(k) != request[k] for k in
                   ("problem", "source_sha256", "dataset_version", "manifest_sha256", "protocol_version")):
                reason = "题目、源码、评测集或计时协议与本次请求不符"
            if raw.get("status") == "running":
                reason = "本腿中断，已有逐例数据只作诊断"
            if raw.get("failure_kind") == "infrastructure":
                reason = str(raw.get("error") or "远端基础设施报告故障")
            try:
                stats = recompute(raw)
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                stats = dict(rankable=False, total_us=None, shape_means_us={}, errors=[str(exc)])
            passed = raw.get("status") == "ok" and raw.get("rankable") is True and stats["rankable"]
            if not reason:
                env = raw.get("environment", {})
                if not isinstance(env, dict) or any(not env.get(k) for k in ENV_KEYS) or not raw.get("protocol"):
                    reason = "缺少 GPU、驱动、工具链、编译选项或计时条件"
                else:
                    signature = _conditions(raw)
                    if (state["conditions"] is not None and signature != state["conditions"]
                            and not _archived_conditions_match(state, signature)):
                        reason = "设备、工具链或协议发生变化；请在新运行目录重新测量基线与波动"
                    elif passed:
                        state["conditions"] = signature
            rankable = bool(passed and not reason)
            if not rankable:
                errors = stats.get("errors", []) or []
                detail = reason or raw.get("error") or ("、".join(map(str, errors)) if errors else "全套校验或计时不完整")
            else:
                detail = ""
            item = dict(id=path.stem, group_id=group["id"], attempt=group["attempt"], leg_index=request["leg_index"],
                        side=request["side"], agent=request["agent"], round_id=request["round_id"],
                        source_sha256=request["source_sha256"], source=request["source_path"], report=str(path),
                        report_sha256=_sha(encoded), run_id=key, recorded_at=_now(), status="ok" if rankable else "failed",
                        rankable=rankable, total_us=stats.get("total_us") if rankable else None,
                        shape_means_us=stats.get("shape_means_us", {}) if rankable else {}, error=str(detail))
            state["reports"].append(item)
            group["report_ids"].append(item["id"])
            state["device"]["last_activity"] = _now()
            if request["round_id"]:
                row = state["rounds"][request["round_id"]]
                row["report_ids"].append(item["id"])
                if request["side"] == "B":
                    row["report"] = item["report"]
                    state["agents"][row["agent"]].update(report=item["report"], last_activity=_now())
            elif rankable and state["best"] is None:
                state["best"] = dict(_result(item), round_id=None, agent=None, group_ids=[])
        if reason:
            raise DeviceUnavailable(reason)
        return item

    def _block(self, group_id, detail):
        with self._edit() as state:
            group = state["groups"][group_id]
            group.update(status="queued", reason=detail)
            _device(state, "blocked", detail)
            if group["round_id"]:
                agent = state["rounds"][group["round_id"]]["agent"]
                _seat(state, agent, "queued", detail)
        return dict(status="blocked", group_id=group_id, reason=detail)

    def run_next(self, evaluate=None, *, evaluate_group=None):
        """消费一组。组回调可返回四份有序报告或逐腿 yield；单条回调接收相同 request。"""
        if (evaluate is None) == (evaluate_group is None):
            raise ValueError("需要指定一种评测回调")
        with (self.root / "evaluation.lock").open("a") as owner:
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return dict(status="busy", reason="该题已有评测进程持锁")
            requests = self._claim()
            if not requests:
                return dict(status="idle")
            if isinstance(requests, dict):
                return requests
            gid = requests[0]["group_id"]
            rows = []
            try:
                for req in requests:
                    if _sha(_source(req["source_path"])) != req["source_sha256"]:
                        raise DeviceUnavailable("候选归档的源码指纹已变化")
                stream = evaluate_group(requests) if evaluate_group else (evaluate(req) for req in requests)
                for i, raw in enumerate(stream):
                    if i == len(requests):
                        raise DeviceUnavailable("远端返回多余的测量腿")
                    rows.append(self._record(requests[i], raw))
                bad_control = next((r for r in rows if r["side"] == "A" and not r["rankable"]), None)
                failed = next((r for r in rows if r["side"] == "B" and not r["rankable"]), None)
                if bad_control or requests[0]["kind"] == "calibrate" and failed:
                    raise DeviceUnavailable("对照未通过全套：" + (bad_control or failed)["error"])
                if len(rows) != len(requests) and not failed:
                    raise DeviceUnavailable("远端未完成四腿；恢复后重跑整组")
            except DeviceUnavailable as exc:
                return self._block(gid, str(exc))
            except BaseException as exc:
                self._block(gid, f"{type(exc).__name__} 中断本组，恢复后重新测量")
                raise
            with self._edit() as state:
                group = state["groups"][gid]
                if failed:
                    group.update(status="failed", reason=failed["error"])
                    for other in state["groups"].values():
                        if (other["status"] == "queued" and other["kind"] == "compare"
                                and other["candidate_sha"] == group["candidate_sha"]):
                            other.update(status="failed", reason=failed["error"])
                else:
                    control = [_metrics(r) for r in rows if r["side"] == "A"]
                    candidate = [_metrics(r) for r in rows if r["side"] == "B"]
                    gains = {k: fmean(r[k] for r in control) - fmean(r[k] for r in candidate)
                             for k in control[0]}
                    group.update(status="completed", gains_us=gains)
                group["finished_at"] = _now()
                rid = group["round_id"]
                if rid:
                    _seat(state, state["rounds"][rid]["agent"], "queued", "等待第二组或当前最佳的确认测量")
                _device(state, "idle", "")
                _advance(state)
                return dict(status="completed", group_id=gid, outcome=group["status"],
                            report_ids=group["report_ids"], gains_us=group["gains_us"])

    def review(self, agent, conclusion, *, next_step="", entry_id=None, experience=None, waiting_reason=None, round_id=None):
        conclusion = _text(conclusion, "复盘结论")
        if bool(entry_id) != bool(experience):
            raise ValueError("共享经验需要同时给出条目标识和正文")
        review = dict(conclusion=conclusion, next_step=next_step, entry_id=entry_id, experience=experience or "无新增经验",
                      waiting_reason=waiting_reason)
        with self._edit() as state:
            row = state["rounds"].get(round_id) if round_id else _active(state, agent)
            if not row or row["agent"] != agent:
                raise ValueError("复盘轮次或席位不符")
            if row["review"] == review:
                return row
            if row["review"] is not None or row["decision"] is None or state["agents"][agent]["round_id"] != row["id"]:
                raise ValueError("需要已结算且尚待复盘的当前轮次")
            if experience:
                _merge_experience(state, row, entry_id, _text(experience, "经验正文"))
            row.update(review=review, reviewed_at=_now())
            family = row["card"]["family"].strip().casefold()
            span = set(row["card"].get("parameter_range", []))
            related = [r for r in state["rounds"].values()
                       if (r["parent_sha"], r["card"]["family"].strip().casefold()) == (row["parent_sha"], family)]
            latest_reset = max((r["created_at"] for r in related if r["changed_premise"]), default="")
            tried = {r["card"].get("parameter") for r in related if r["created_at"] >= latest_reset
                     and r["decision"] in ("indistinguishable", "regressed")}
            found_gain = any(r["created_at"] >= latest_reset and r["card"].get("parameter") in span
                             and r["decision"] in ("promoted", "shape_only", "shared_best") for r in related)
            reason = waiting_reason
            if not reason and row["decision"] == "regressed":
                reason = row["conclusion"]
            if not reason and span and span <= tried and not found_gain and row["decision"] in ("indistinguishable", "regressed"):
                reason = "预定参数范围已测完，未发现可复现收益：" + "、".join(sorted(span))
            if reason:
                state["stalled"].append(dict(parent_sha=row["parent_sha"], family=family, agent=agent,
                    round_id=row["id"], reason=_text(reason, "等待原因"), tried=sorted(tried - {None}),
                    report=row["report"], active=True))
            _seat(state, agent, "waiting_evidence" if reason else "idle", reason or conclusion)
            return row

    def attach_profile(self, source_sha, shape_id, report):
        """将剖析摘要绑定到本题已有源码与形状，仅供后续实验诊断。"""
        if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
            raise ValueError("剖析源码需要完整的 SHA-256")
        with Path(report).open("rb") as stream:
            data = stream.read(65537)
        if len(data) > 65536 or b"\0" in data or not data.decode("utf-8").strip():
            raise ValueError("请提供 64 KiB 以内的 UTF-8 文本剖析摘要")
        digest = _sha(data)
        entry = dict(source_sha256=source_sha, shape_id=shape_id, report_sha256=digest)
        with self._edit() as state:
            if shape_id not in state["shape_ids"]:
                raise ValueError("剖析形状不属于本题清单")
            if source_sha != state["baseline_sha"] and not any(r["candidate_sha"] == source_sha for r in state["rounds"].values()):
                raise ValueError("剖析源码不属于本题已归档的候选")
            path = self.root / "profile" / digest / "REPORT.md"
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError("已归档剖析报告与指纹不符")
            else:
                _atomic(path, data)
            profiles = state.setdefault("profiles", [])
            if entry not in profiles:
                profiles.append(entry)
        return entry

    def context(self, agent, *, compact=False):
        """沙箱 worker 收到同伴源码、最新全套原始数据、历史比较和共享经验正文。"""
        from .local_eval import manifest
        state = json.loads(self.path.read_bytes())
        if agent not in AGENTS:
            raise ValueError("未知 Agent")
        slot = state["agents"][agent]
        parent_sha = slot["parent_sha"]
        frozen = manifest(state["problem"])
        if frozen["manifest_sha256"] != state["manifest_sha256"]:
            raise ValueError("该目录的冻结清单已改变")
        peer_sources = {}
        reports = {}
        report_hashes = {r["report"]: r["report_sha256"] for r in state["reports"]}
        recent = {}
        measured = {}
        for trial in state["rounds"].values():
            if trial["candidate_sha"]:
                recent[trial["agent"]] = trial
                if trial["decision"] and trial.get("report"):
                    measured[trial["agent"]] = trial
        seats = [*recent.values(), *measured.values(), state["best"] or {}]
        if not any(s.get("report") and (s.get("candidate_sha") or s.get("source_sha256")) == parent_sha for s in seats):
            parent_report = next((r for r in reversed(state["reports"]) if r["source_sha256"] == parent_sha
                                  and r["rankable"]), None)
            if parent_report:
                seats.append(parent_report)
        for seat in seats:
            sha = seat.get("candidate_sha") or seat.get("source_sha256")
            if sha and sha != parent_sha and sha not in peer_sources:
                code = _source(self._source_path(sha))
                if _sha(code) != sha:
                    raise ValueError("同伴源码与已归档指纹不符")
                peer_sources[sha] = code.decode()
            if seat.get("report") and seat["report"] not in reports:
                raw = Path(seat["report"]).read_bytes()
                if _sha(raw) != report_hashes[seat["report"]]:
                    raise ValueError("共享给模型的原始报告与归档指纹不符")
                reports[seat["report"]] = json.loads(raw)
        parent = _source(self._source_path(parent_sha))
        if _sha(parent) != parent_sha:
            raise ValueError("父版源码与归档指纹不符")
        profiles = state.get("profiles", [])
        latest = {p["shape_id"]: p for p in profiles}
        parent_profiles = {p["shape_id"]: p for p in profiles if p["source_sha256"] == parent_sha}
        profile_evidence = []
        for entry in profiles:
            item = dict(entry)
            if not compact or entry in (latest[entry["shape_id"]], parent_profiles.get(entry["shape_id"])):
                raw = (self.root / "profile" / entry["report_sha256"] / "REPORT.md").read_bytes()
                if _sha(raw) != entry["report_sha256"]:
                    raise ValueError("共享给模型的剖析报告与归档指纹不符")
                item["markdown"] = raw.decode("utf-8")
            profile_evidence.append(item)
        rounds = list(state["rounds"].values())
        context = dict(problem=state["problem"], language=state["language"], contract=state["contract"], cases=frozen["shapes"],
                    reference=frozen.get("reference"), abi=frozen.get("abi"), agent=agent, current_round=slot["round_id"],
                    eligible_parents=_eligible_parents(state), input_cases=frozen["cases"], parent_sha=parent_sha, parent_source=parent.decode(),
                    related_sources=peer_sources, recent_raw_reports=reports, noise_band=state["noise_band"],
                    shape_routes=state["shape_routes"], profiles=profile_evidence,
                    comparison_orders=state.get("comparison_orders", ["ABBA"]),
                    experience=Path(state["experience_path"]).read_text(encoding="utf-8"), best=state["best"],
                    stalled=state["stalled"])
        if compact:
            fields = ("id", "round", "agent", "parent_sha", "candidate_sha", "card", "borrowed_from", "changed_premise",
                      "summary", "comparisons", "decision", "conclusion", "review")
            context["rounds"] = [{k: row.get(k) for k in fields} for row in rounds]
            context["measurements_by_source"] = _source_measurements(state["reports"])
        else:
            context.update(rounds=rounds, all_reports=state["reports"])
        return context

    def develop(self, agent, *, timeout=900):
        state = json.loads(self.path.read_bytes())
        row = _active(state, agent)
        rid = row["id"]
        workspace = Path(row["workspace"])
        with (workspace / "model.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("该席位已有模型调用") from exc
            with self._edit() as state:
                row = _active(state, agent, rid)
                if state["agents"][agent]["state"] != "developing":
                    raise ValueError("该轮当前不在开发状态")
                row["worker"] = dict(backend="codex exec", model=MODEL, reasoning_effort="max", started_at=_now())
                _seat(state, agent, "developing", "Codex 正在开发整题源码", pid=os.getpid(),
                      proc_start=_proc_start(os.getpid()))
            try:
                receipt = workspace / "worker-output.json"
                if receipt.is_file():
                    cached = json.loads(receipt.read_bytes())
                    result = cached["result"]
                    if ((cached["parent_sha"], cached["card"], cached["round_id"]) != (row["parent_sha"], row["card"], rid)
                            or (cached.get("model"), cached.get("reasoning_effort")) != (MODEL, "max")
                            or cached.get("source_sha256") != _sha(result["source"].encode())):
                        raise ValueError("已保存模型输出的固定档位、源码或轮次谱系不符")
                else:
                    result = workers.develop(self.context(agent, compact=True), row["card"], timeout=timeout, max_effort=True)
                    _atomic(receipt, _json_bytes(dict(round_id=rid, parent_sha=row["parent_sha"], card=row["card"],
                        model=MODEL, reasoning_effort="max", source_sha256=_sha(result["source"].encode()), result=result)))
                return self.submit(agent, result["source"].encode(), summary=result["summary"], round_id=rid)
            except BaseException as exc:
                with self._edit() as state:
                    if state["agents"][agent]["round_id"] == rid and state["agents"][agent]["state"] == "developing":
                        _seat(state, agent, "developing", f"模型调用未完成：{exc}", pid=None, proc_start=None)
                raise

    def snapshot(self):
        return dashboard_state(self.root)


def dashboard_state(workspace):
    path = Path(workspace) / "state.json"
    if not path.is_file():
        return None
    state = json.loads(path.read_bytes())
    for slot in [*state["agents"].values(), state["device"]]:
        pid = slot.get("pid")
        slot["alive"] = bool(slot.get("proc_start") and _proc_start(pid) == slot["proc_start"]) if pid else None
        if slot.get("model_phase") == "plan":
            slot["state"] = "developing"
        if slot["alive"] is False:
            cause = "进程已离开，保留最后活动时刻与在途任务"
            if "agent" in slot:
                slot["detail"] = cause
            else:
                slot.update(state="blocked", reason=cause)
    state["agents"] = list(state["agents"].values())
    state["rounds"] = list(state["rounds"].values())
    state["groups"] = list(state["groups"].values())
    state["queue"] = [g for g in state["groups"] if g["status"] in ("queued", "running")]
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="一道题的本地 campaign 目录")
    subs = parser.add_subparsers(dest="action", required=True)
    init = subs.add_parser("init", help="归档基线、冻结清单、排入三组自比")
    init.add_argument("problem")
    init.add_argument("baseline", type=Path)
    init.add_argument("--experience", type=Path)
    init.add_argument("--language", default="CUDA C++", choices=["CUDA C++"])
    init.add_argument("--contract", default="")
    subs.add_parser("status", help="看三席状态、队列、各轮报告和波动带")
    target = subs.add_parser("set-round-target", help="为协作席位登记本次结束的累计轮次")
    target.set_defaults(action="set_round_target")
    target.add_argument("rounds", type=int)
    profile = subs.add_parser("attach-profile", help="归档与候选 SHA、形状绑定的性能剖析摘要")
    profile.set_defaults(action="attach_profile")
    profile.add_argument("source_sha")
    profile.add_argument("shape_id")
    profile.add_argument("report", type=Path)
    for name, help_text in (("begin", "公布父版、机制和目标形状"), ("submit", "将完整源码送入本地队列"),
                            ("review", "结算复盘并可选择增补共享经验"), ("context", "读取源码、报告、共享经验"),
                            ("develop", "用固定的 gpt-6.2-astra max 开发当前轮完整源码")):
        sub = subs.add_parser(name, help=help_text)
        sub.add_argument("agent", choices=AGENTS)
        if name == "begin":
            sub.add_argument("card", type=Path, help="含 family/change/falsifier/target_cases 的 JSON 优化卡")
            sub.add_argument("--parent-sha")
            sub.add_argument("--borrowed-from", default="")
            sub.add_argument("--changed-premise", default="")
            sub.add_argument("--remeasure", action="store_true", help="同源也新建两个独立的对照组")
            sub.add_argument("--worker-id", help="记录外部已按 gpt-6.2-astra max 启动的协作 Agent 身份")
        elif name == "submit":
            sub.add_argument("source", type=Path)
            sub.add_argument("--summary", default="")
            sub.add_argument("--round-id")
        elif name == "review":
            sub.add_argument("conclusion")
            sub.add_argument("--next-step", default="")
            sub.add_argument("--entry-id")
            sub.add_argument("--experience", help="本轮增补或勘误的经验正文")
            sub.add_argument("--waiting-reason", help="需新证据才能重试本机制的实测依据")
            sub.add_argument("--round-id")
        elif name == "develop":
            sub.add_argument("--timeout", type=float, default=900)
        elif name == "context":
            sub.add_argument("--compact", action="store_true", help="显示模型实际使用的交接上下文")
    args = vars(parser.parse_args(argv))
    campaign = Campaign(args.pop("root"))
    action = args.pop("action")
    try:
        if action == "begin":
            args["card"] = json.loads(args["card"].read_bytes())
        result = campaign.snapshot() if action == "status" else getattr(campaign, action)(**args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, workers.WorkerError) as exc:
        parser.exit(2, f"local_campaign: {exc}\n")


if __name__ == "__main__":
    main()
