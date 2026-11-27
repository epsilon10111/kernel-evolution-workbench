"""按已配置的 campaign 读取本地席位及其不可变报告。"""

import json
from pathlib import Path
from statistics import median


AGENTS = ("a", "b", "c")


def _candidate_legs(row, reports):
    if not row.get("evaluated_at") or row.get("decision") in (None, "invalid", "duplicate"):
        return []
    return [reports[i] for i in row["report_ids"] if i in reports and reports[i].get("side") == "B"
            and reports[i]["source_sha256"] == row["candidate_sha"]]


def latency_trend(state):
    reports = {r["id"]: r for r in state["reports"] if r.get("rankable")}
    points = []

    def append(name, agent, number, decision, measured_at, samples):
        if samples:
            totals = [r["total_us"] for r in samples]
            total = median(totals)
            shapes = {s: median(r["shape_means_us"][s] for r in samples) for s in state["shape_ids"]}
            representative = min(samples, key=lambda r: abs(r["total_us"] - total))
            points.append(dict(id=name, agent=agent, round=number, decision=decision, measured_at=measured_at,
                               total_us=total, total_range_us=[min(totals), max(totals)], shape_means_us=shapes,
                               samples=len(samples), report_id=representative["id"]))

    baseline = [reports[rid] for g in state["groups"] if g.get("kind") == "calibrate" and g["status"] == "completed"
                for rid in g["report_ids"] if rid in reports]
    if baseline:
        append("baseline", None, 0, "baseline", baseline[-1]["recorded_at"], baseline)
    for row in sorted(state["rounds"], key=lambda r: r.get("evaluated_at") or ""):
        samples = _candidate_legs(row, reports)
        if samples:
            append(row["id"], row["agent"], row["round"], row["decision"], row["evaluated_at"], samples)
    return points


def top_sources(state):
    target = state.get("round_target")
    if (not target or state["queue"] or len(state["rounds"]) != target * len(AGENTS)
            or any(s["round"] != target or s["state"] not in ("idle", "waiting_evidence") for s in state["agents"])
            or not all(r.get("review") and r.get("decision") for r in state["rounds"])):
        return None
    reports = {r["id"]: r for r in state["reports"] if r.get("rankable")}
    sources = {}
    for row in sorted(state["rounds"], key=lambda r: (r["round"], r["agent"])):
        legs = _candidate_legs(row, reports)
        if legs and row.get("source"):
            sha = row["candidate_sha"]
            item = sources.setdefault(sha, dict(source_sha256=sha, agent=row["agent"], round=row["round"], totals={}))
            item["totals"].update((r["id"], r["total_us"]) for r in legs)
    for item in sources.values():
        totals = item.pop("totals").values()
        item.update(total_us=median(totals), samples=len(totals))
    return sorted(sources.values(), key=lambda i: (i["total_us"], i["round"], i["agent"], i["source_sha256"]))[:3]


def campaign_view(root):
    state = None
    if (root / "state.json").is_file():
        from .local_campaign import dashboard_state
        state = dashboard_state(root)
    preparation = None
    path = root / "preparation.json"
    if path.is_file():
        try:
            with path.open("rb") as stream:
                data = json.loads(stream.read(32769))
        except (OSError, ValueError):
            data = {"stage": "架构进度文件暂时无法读取"}
        if isinstance(data, dict):
            preparation = {k: data[k][:500] for k in ("stage", "detail", "updated_at")
                           if isinstance(data.get(k), str)}
    if state is None:
        return dict(problem=root.parent.name, initialized=False, preparation=preparation,
                    agents=[dict(agent=a, state="idle") for a in AGENTS],
                    best=None, reports=[], rounds=[], shape_routes=[], queue=[], device={}, shape_ids=[], trend=[], top3=None)
    return {**state, "initialized": True, "preparation": preparation, "trend": latency_trend(state),
            "top3": top_sources(state)}


def file_for(root, kind, key):
    """只接受当前 campaign 索引中登记的文件，即使 URL 被修改也不暴露其他目录。"""
    state = campaign_view(root)
    reports = state["reports"]
    if kind == "report":
        path = next((r.get("report") for r in reports if str(r["id"]) == key), None)
    elif kind == "source":
        entries = [state.get("best") or {}, *state["rounds"], *reports, *state["agents"]]
        path = next((r["source"] for r in entries if r.get("source") and
                     key in (r.get("source_sha256"), r.get("candidate_sha"))), None)
    else:
        return None
    if not path:
        return None
    target = Path(path).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        return None
    return target
