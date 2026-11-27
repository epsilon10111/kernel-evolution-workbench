"""并行运行 a、b、c 三席的有限轮本地开发。另开进程运行 python -m harness.local_runner drain --root DIR --watch。"""

import argparse
import fcntl
import hashlib
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from . import workers
from .local_campaign import AGENTS, MODEL, Campaign, _atomic, _json_bytes, _now, _proc_start, _seat


def _evidence(state):
    rounds = [(r["id"], r["candidate_sha"], r["decision"], r.get("reviewed_at")) for r in state["rounds"]]
    experience = Path(state["experience_path"]).read_bytes()
    evidence = [len(state["reports"]), rounds]
    if state.get("profiles"):
        evidence.append(state["profiles"])
    return hashlib.sha256(_json_bytes(evidence) + experience).hexdigest()


def _next_action(slot, state, rounds, evidence):
    number = slot["round"] or 0
    if number > rounds:
        return "done"
    if slot.get("model_phase") == "plan":
        return "done" if number == rounds else "plan"
    if slot["state"] in ("queued", "evaluating"):
        return "gpu"
    if slot["state"] == "reviewing":
        return "review"
    if slot["state"] == "developing":
        return "develop"
    if number == rounds:
        return "done"
    if state["best"] is None:
        return "gpu"
    if slot["state"] == "waiting_evidence" and slot.get("wait_evidence_token") == evidence:
        return "evidence"
    return "plan"


def _decision(campaign, slot, phase, evidence, timeout):
    """保存模型的结构化决策，掉线后可把同一份结果接回 begin/review。"""
    agent = slot["agent"]
    number = (slot["round"] or 0) + (phase == "plan")
    path = campaign.root / "agents" / agent / f"round-{number:04d}" / (phase + ".json")
    if path.is_file():
        saved = json.loads(path.read_bytes())
        if ((saved.get("model"), saved.get("reasoning_effort"), saved.get("phase"), saved.get("agent"), saved.get("round"))
                != (MODEL, "max", phase, agent, number)):
            raise ValueError("已存模型决策的档位或轮次不符")
        if not saved.get("rejected") and (saved["result"].get("action") != "wait" or saved["evidence_token"] == evidence):
            return saved["result"]
    with campaign._edit() as state:
        current = state["agents"][agent]
        if current["round_id"] != slot["round_id"]:
            raise ValueError("席位轮次在模型调用前已变更")
        current.update(model_phase=phase, pid=os.getpid(), proc_start=_proc_start(os.getpid()), last_activity=_now(),
                       detail="固定模型正在根据同伴报告选择父版与机制" if phase == "plan" else "固定模型正在复盘实测结果")
    try:
        result = workers.campaign_decision(campaign.context(agent, compact=True), phase, timeout=timeout)
        _atomic(path, _json_bytes(dict(model=MODEL, reasoning_effort="max", phase=phase, agent=agent,
                                      round=number, evidence_token=evidence, result=result, completed_at=_now())))
        return result
    except BaseException as exc:
        with campaign._edit() as state:
            current = state["agents"][agent]
            if current["round_id"] == slot["round_id"] and current.get("model_phase") == phase:
                current.update(model_phase=None, pid=None, proc_start=None, last_activity=_now(), detail=f"{phase} 未完成：{exc}")
        raise


def _step(campaign, slot, action, evidence, timeout):
    agent = slot["agent"]
    if action == "develop":
        row = campaign.develop(agent, timeout=timeout)
        return dict(agent=agent, action="submitted", round_id=row["id"], source_sha256=row["candidate_sha"])
    result = _decision(campaign, slot, action, evidence, timeout)
    if action == "review":
        row = campaign.review(agent, result["conclusion"], next_step=result["next_step"], entry_id=result["entry_id"] or None,
                              experience=result["experience"] or None, waiting_reason=result["waiting_reason"] or None,
                              round_id=slot["round_id"])
        return dict(agent=agent, action="reviewed", round_id=row["id"], decision=row["decision"])
    reason = result["waiting_reason"]
    if result["action"] != "wait":
        card = {k: result[k] for k in ("family", "change", "falsifier", "target_cases")}
        if result["parameter_range"]:
            card.update(parameter_range=result["parameter_range"], parameter=result["parameter"])
        try:
            row = campaign.begin(agent, card, parent_sha=result["parent_sha"], borrowed_from=result["borrowed_from"],
                                 changed_premise=result["changed_premise"])
        except ValueError as exc:
            path = campaign.root / "agents" / agent / f"round-{(slot['round'] or 0) + 1:04d}" / "plan.json"
            saved = json.loads(path.read_bytes())
            reason = f"计划未开工：{exc}"
            _atomic(path, _json_bytes(dict(saved, rejected=reason)))
        else:
            return dict(agent=agent, action="planned", round_id=row["id"], parent_sha=row["parent_sha"], family=card["family"])
    with campaign._edit() as state:
        if state["agents"][agent]["round_id"] != slot["round_id"]:
            return dict(agent=agent, action="superseded", reason="席位已收到新的开发轮次")
        _seat(state, agent, "waiting_evidence", reason, wait_evidence_token=evidence)
    return dict(agent=agent, action="waiting_evidence", reason=reason)


def run(campaign, rounds, *, poll=4.0, timeout=900, on_event=None):
    """rounds 是每席累计轮次上限，使用相同参数重启不会再多派 N 轮。"""
    if type(rounds) is not int or rounds < 1 or not 0 < poll <= 60 or not 0 < timeout < float("inf"):
        raise ValueError("轮数需为正整数，轮询间隔为 (0,60] 秒，模型超时需为正数")
    if campaign.snapshot() is None:
        raise ValueError("请先用 local_campaign init 建立本题的基线队列")
    with (campaign.root / "producer.lock").open("a") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("该题已有一个自动开发进程") from exc
        campaign.set_round_target(rounds)
        futures, failures = {}, {}
        with ThreadPoolExecutor(max_workers=len(AGENTS)) as pool:
            while True:
                # 先处理已完成 Future，再读取状态，避免把过期快照送给已推进的席位。
                for future in tuple(futures):
                    if future.done():
                        agent = futures.pop(future)
                        try:
                            result = future.result()
                        except (workers.WorkerError, ValueError, OSError) as exc:
                            failures[agent] = str(exc)
                            result = dict(agent=agent, action="worker_failed", reason=str(exc))
                            with campaign._edit() as state:
                                state["agents"][agent].update(model_error=str(exc), detail=f"模型工位需恢复：{exc}",
                                                            pid=None, proc_start=None, model_phase=None, last_activity=_now())
                        if on_event:
                            on_event(result)
                state = campaign.snapshot()
                token = _evidence(state)
                actions = {}
                for slot in state["agents"]:
                    agent = slot["agent"]
                    if agent in futures.values():
                        continue
                    if agent in failures:
                        actions[agent] = "failed"
                        continue
                    action = _next_action(slot, state, rounds, token)
                    actions[agent] = action
                    if action in ("plan", "develop", "review"):
                        future = pool.submit(_step, campaign, slot, action, token, timeout)
                        futures[future] = agent
                if futures:
                    wait(futures, timeout=poll, return_when=FIRST_COMPLETED)
                elif "gpu" in actions.values() or "evidence" in actions.values() and state["queue"]:
                    time.sleep(poll)
                else:
                    if failures:
                        raise RuntimeError("；".join(f"席位 {a}：{message}" for a, message in failures.items()))
                    completed = all(a == "done" for a in actions.values())
                    return dict(status="completed" if completed else "waiting_evidence", round_limit=rounds,
                                rounds={s["agent"]: s["round"] or 0 for s in state["agents"]},
                                waiting={s["agent"]: s["detail"] for s in state["agents"] if actions[s["agent"]] == "evidence"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, required=True, help="每席累计轮次上限。提高此值可从已有轮次继续")
    parser.add_argument("--poll", type=float, default=4.0)
    parser.add_argument("--timeout", type=float, default=900, help="每次固定模型调用的秒数")
    args = parser.parse_args(argv)
    def emit(value):
        print(json.dumps(value, ensure_ascii=False), flush=True)
    try:
        emit(dict(model=MODEL, reasoning_effort="max", agents=list(AGENTS), round_limit=args.rounds,
                  gpu_consumer=["python", "-m", "harness.local_runner", "drain", "--root", str(args.root), "--watch"]))
        emit(run(Campaign(args.root), args.rounds, poll=args.poll, timeout=args.timeout, on_event=emit))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"local_loop: {exc}\n")


if __name__ == "__main__":
    main()
