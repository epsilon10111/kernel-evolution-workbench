"""通过非交互 CLI 生成优化卡和单文件候选。"""

import json
import logging
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID


class WorkerError(RuntimeError):
    """可交给调用者记录的错误，不包含提示词、凭据或 CLI 的输出。"""


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


_TEXT = {"type": "string"}
_FABLE = "claude-fable-5-2"
_FABLE_BUDGET_USD = 1
_ASTRA = "gpt-6.2-astra"
_PLAN_TASK = (
    "只提出两张优化卡，必须采用不同的实际机制。每张优先只改变一个可归因的因素。"
    "给出机制 family、具体改动 change、能证伪该假设的观察 falsifier 和非空数组 target_cases。"
    "target_cases 只能从 context.cases 中选择原样的 shape_id；旧清单选择实际 N 数值，禁止使用测例序号或档位。"
    "区分实测结果和待验证的推测，禁止捏造证据。返回含 cards 的 JSON 对象。"
)
_DEVELOP_TASK = (
    "按所选 card 修改 context.parent_source，遵守 context.contract、语言及原有入口。"
    "只使用本次输入，直接在 source 字段返回一份可保存的完整源码，包含所有未改动的必要部分。"
    "禁止补丁、Markdown 围栏、文件路径代替源码或省略实现。language 原样使用 context.language，summary 简述改动。"
)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _check_card(card, sizes):
    if not isinstance(card, dict) or not all(_text(card.get(key)) for key in ("family", "change", "falsifier")):
        raise WorkerError("优化卡缺少机制、改动或可证伪的观察")
    targets = card.get("target_cases")
    if not isinstance(targets, list) or not targets or any(type(n) not in (str, int) or n not in sizes for n in targets):
        raise WorkerError("target_cases 必须来自 context.cases 的 shape_id 或实际 N")


def _case_sizes(context):
    cases = context.get("cases")
    if not isinstance(cases, (list, dict)) or not cases:
        raise WorkerError("context.cases 缺少实际 N")
    sizes = set()
    entries = cases.items() if isinstance(cases, dict) else ((None, item) for item in cases)
    for key, item in entries:
        if isinstance(item, dict):
            if "shape_id" in item:
                if not _text(item["shape_id"]):
                    raise WorkerError("context.cases 含有空的 shape_id")
                sizes.add(item["shape_id"])
                continue
            size = item.get("N", item.get("n", key))
        else:
            size = item
        if isinstance(size, str) and size.isascii() and size.isdecimal():
            size = int(size)
        if type(size) is not int:
            raise WorkerError("context.cases 含有无法识别的 N")
        sizes.add(size)
    return sizes


def _plan_schema(sizes):
    card = _object({
        "family": _TEXT, "change": _TEXT, "falsifier": _TEXT,
        "target_cases": {"type": "array", "minItems": 1, "description": "context.cases 中列出的形状 ID 或实际 N",
                         "items": {"enum": sorted(sizes, key=str)}},
    })
    return _object({"cards": {"type": "array", "minItems": 2, "maxItems": 2, "items": card}})


def _fable_usage(result, max_budget_usd=_FABLE_BUDGET_USD):
    usage = result.get("modelUsage")
    if not isinstance(usage, dict) or set(usage) != {_FABLE} or not isinstance(usage[_FABLE], dict):
        observed = []
        if isinstance(usage, dict):
            for model, item in list(usage.items())[:6]:
                label = model if re.fullmatch(r"claude-(?:fable|opus|sonnet|haiku)-[0-9.\-]+(?:\[1m\])?", str(model)) else "其他模型"
                generated = item.get("outputTokens") if isinstance(item, dict) else None
                observed.append(f"{label}/{generated if type(generated) is int else '未知'}")
        detail = "，".join(observed) or "无模型数据"
        raise WorkerError(f"Claude modelUsage 缺失或与固定 Fable 模型不符（{detail}）")
    counts = usage[_FABLE]
    if type(counts.get("outputTokens")) is not int or counts["outputTokens"] <= 0:
        raise WorkerError("Claude modelUsage 缺少可核实的 Fable 输出用量")
    verified = {"outputTokens": counts["outputTokens"]}
    for key in ("inputTokens", "cacheReadInputTokens", "cacheCreationInputTokens"):
        if key in counts:
            value = counts[key]
            if type(value) is not int or value < 0:
                raise WorkerError("Claude modelUsage 用量无效")
            verified[key] = value
    for record, key in ((counts, "costUSD"), (result, "total_cost_usd")):
        if key in record:
            cost = record[key]
            if (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
                    or (max_budget_usd is not None and cost > max_budget_usd)):
                raise WorkerError("Claude 回执费用无效或超出本轮预算")
    if "costUSD" in counts:
        verified["costUSD"] = counts["costUSD"]
    return {_FABLE: verified}


def _prompt(task, context, timeout, **fields):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise WorkerError("timeout 必须是正的有限秒数")
    if not isinstance(context, dict) or not _text(context.get("language")):
        raise WorkerError("context 缺少有效的 language")
    try:
        return json.dumps({"task": task, "context": context, **fields},
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError):
        raise WorkerError("worker 输入无法编码为 JSON") from None


@contextmanager
def _scratch(base_env):
    with tempfile.TemporaryDirectory(prefix="cli-worker-", dir="/tmp") as tmp:
        root = Path(tmp)
        home, work = root / "home", root / "work"
        home.mkdir()
        work.mkdir()
        env = dict(base_env, HOME=str(home), PWD=str(work), TMPDIR=tmp,
                   XDG_CONFIG_HOME=str(home / ".config"), XDG_CACHE_HOME=str(home / ".cache"),
                   XDG_DATA_HOME=str(home / ".local/share"), XDG_STATE_HOME=str(home / ".local/state"),
                   PATH=os.defpath, TERM="dumb")
        yield root, work, env


def _run(name, argv, prompt, work, env, timeout):
    try:
        with subprocess.Popen(
            argv, cwd=work, env=env, start_new_session=True, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if name in ("Claude", "Codex Fast", "Codex max") else subprocess.DEVNULL,
        ) as proc:
            try:
                stdout, stderr = proc.communicate(prompt, timeout=timeout)
            except BaseException:
                # 父进程可能已退出，而同组子进程仍占着输出管道。
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                raise
            _record_cli_usage(name, stdout)
            if name == "Codex Fast" and any(message in (stderr or b"").lower() for message in
                                            (b"will be omitted from requests", b"defaulting to fallback metadata")):
                raise WorkerError("Codex CLI 未保留 Fast 请求档位")
            if name == "Codex max" and any(message in (stderr or b"").lower() for message in
                                           (b"will be omitted from requests", b"defaulting to fallback metadata",
                                            b"falling back", b"unsupported reasoning")):
                raise WorkerError("Codex CLI 无法保持固定的 Astra max 请求")
            if proc.returncode:
                reason = ""
                if name == "Claude":
                    diagnostic = ((stdout or b"") + (stderr or b"")).lower()
                    for pattern, label in (
                        (rb"error_max_budget_usd|exceeded.*budget", "达到单次模型预算"),
                        (rb"\b(401|403)\b|authentication_error|invalid.*api.key", "网关认证失败"),
                        (rb"\b429\b|rate.limit", "网关限流"),
                        (rb"\b(500|502|503|504|529)\b|overloaded", "网关暂时不可用"),
                        (rb"timed?.?out|econnreset|enotfound|unable to connect|connection error", "网关连接失败"),
                        (rb"context.length|prompt is too long", "上下文长度超限"),
                    ):
                        if re.search(pattern, diagnostic):
                            reason = "，" + label
                            break
                raise WorkerError(f"{name} 执行失败（退出码 {proc.returncode}{reason}）")
            return stdout
    except subprocess.TimeoutExpired as exc:
        _record_cli_usage(name, exc.output)
        raise WorkerError(f"{name} 执行超时") from None
    except (OSError, ValueError):
        raise WorkerError(f"{name} 无法启动或完成通信") from None


def _json(raw, name):
    def unique_fields(pairs):
        fields = dict(pairs)
        if len(fields) != len(pairs):
            raise ValueError("duplicate key")
        return fields

    try:
        return json.loads(raw, object_pairs_hook=unique_fields)
    except (ValueError, TypeError, RecursionError):
        raise WorkerError(f"{name} 输出不是有效的单个 JSON 值") from None


def plan(context: dict, *, timeout=300, max_budget_usd=_FABLE_BUDGET_USD) -> list[dict]:
    if max_budget_usd is not None and (type(max_budget_usd) not in (int, float)
                                       or not math.isfinite(max_budget_usd) or max_budget_usd <= 0):
        raise WorkerError("单次模型预算需要为正数，或显式使用无金额上限模式")
    prompt = _prompt(_PLAN_TASK, context, timeout)
    sizes = _case_sizes(context)
    try:
        from .gateway import claude_env, resolve_claude
        binary, base_env = resolve_claude(), claude_env()
    except Exception:
        raise WorkerError("Claude 受控网关不可用") from None
    if not isinstance(base_env, dict) or not _text(base_env.get("ANTHROPIC_BASE_URL")):
        raise WorkerError("Claude 缺少显式网关地址")
    with _scratch(base_env) as (_, work, env):
        env.update(CLAUDE_CONFIG_DIR=str(Path(env["HOME"]) / ".claude"),
                   CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_AUTOUPDATER="1")
        argv = [binary, "-p", "--bare", "--safe-mode", "--restricted", "--tools", "",
                "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "",
                "--disable-slash-commands", "--permission-mode", "dontAsk", "--permission-prompts", "none",
                "--model", _FABLE, *(["--max-budget-usd", str(max_budget_usd)] if max_budget_usd is not None else []),
                "--output-format", "json", "--json-schema",
                json.dumps(_plan_schema(sizes)), "--system-prompt", "只使用标准输入中的上下文，按指定结构回答。"]
        result = _json(_run("Claude", argv, prompt, work, env, timeout), "Claude")
    if (not isinstance(result, dict) or result.get("type") != "result"
            or result.get("subtype") != "success" or result.get("is_error") is not False):
        raise WorkerError("Claude 未返回成功的最终结果")
    usage = _fable_usage(result, max_budget_usd)
    data = (result["structured_output"] if "structured_output" in result
            else _json(result.get("result"), "Claude"))
    cards = data.get("cards") if isinstance(data, dict) else None
    if not isinstance(cards, list) or len(cards) != 2:
        raise WorkerError("Claude 应返回恰好两张优化卡")
    for card in cards:
        _check_card(card, sizes)
    for field in ("family", "change"):
        if len({" ".join(card[field].casefold().split()) for card in cards}) != 2:
            raise WorkerError("Claude 的两张卡必须提出不同机制和改动")
    return [dict(card, modelUsage=usage) for card in cards]


# 该版 exec 仍注册非交互下不可用的 request_user_input。
_DISABLED = (
    "shell_tool", "unified_exec", "shell_snapshot", "view_image", "js_repl", "code_mode",
    "multi_agent", "multi_agent_v2", "apps", "plugins", "hooks", "browser_use", "computer_use",
    "workspace_dependencies", "skill_search", "skill_mcp_dependency_install",
    "plugin_discovery_prefetch", "daemon_auto_start", "goals", "default_mode_request_user_input",
)
_PROXY_ENV = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")


def _codex_sandbox(binary, root, env):
    # read-only 仍允许全盘读取。只挂载运行时、临时目录和 Codex 自己的只读登录文件。
    bwrap = shutil.which("bwrap", path=os.defpath)
    if not bwrap:
        raise WorkerError("Codex 缺少文件系统隔离程序 bwrap")
    try:
        exe = Path(binary).resolve(strict=True)
    except (OSError, RuntimeError):
        raise WorkerError("Codex 可执行文件不可用") from None
    cmd = [bwrap, "--die-with-parent", "--unshare-all", "--share-net", "--proc", "/proc",
           "--dev", "/dev", "--ro-bind", "/usr", "/usr"]
    for path in ("/bin", "/lib", "/lib64", "/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf",
                 "/etc/ssl/certs/ca-certificates.crt", "/etc/ssl/cert.pem", "/etc/pki/tls/certs/ca-bundle.crt"):
        cmd.extend(["--ro-bind-try", path, path])
    if not exe.is_relative_to("/usr"):
        # npm 的 JS 启动器通过模块解析启动同一 scope 中的原生二进制。
        install = exe.parent.parent.parent if exe.parts[-4:] == ("@openai", "codex", "bin", "codex.js") else exe
        cmd.extend(["--ro-bind", str(install), str(install)])
    cmd.extend(["--bind", str(root), str(root)])
    home = Path(env["HOME"]) / ".codex"
    home.mkdir()
    env["CODEX_HOME"] = str(home)
    if not env.get("CODEX_API_KEY"):
        login = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "auth.json"
        if not login.is_absolute() or not login.is_file():
            raise WorkerError("Codex 缺少自身登录或 CODEX_API_KEY")
        cmd.extend(["--ro-bind", str(login), str(home / "auth.json")])
    return [*cmd, "--", str(exe)]


def _model_catalog(root, *, fast=False, max_effort=False):
    # 独立 CLI 的空 HOME 没有桌面模型目录，CLI 此时会静默忽略 priority。
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    try:
        catalog = json.loads((home / "models_cache.json").read_bytes())
        models = [m for m in catalog["models"] if m["slug"] == _ASTRA]
        if len(models) != 1:
            raise ValueError
        if fast and not any(t["id"] == "priority" and t["name"] == "Fast" for t in models[0].get("service_tiers", [])):
            raise ValueError
        if max_effort and not any(level["effort"] == "max" for level in models[0].get("supported_reasoning_levels", [])):
            raise ValueError
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise WorkerError(f"当前 Codex 模型目录无法验证 Astra 的 {'max' if max_effort else 'Fast'} 档位") from exc
    target = root / "model-catalog.json"
    target.write_text(json.dumps({"models": models}))
    return target


def _write_usage(app, filename, records):
    home = Path.home()
    try:
        settings = json.loads((home / ".cc-switch" / "settings.json").read_bytes())
    except FileNotFoundError:
        settings = {}
    except (OSError, ValueError, RecursionError):
        logging.getLogger(__name__).warning("无法读取 CC Switch 的用量采集目录")
        return
    override = settings.get(f"{app}ConfigDir") if isinstance(settings, dict) else None
    override = override.strip() if isinstance(override, str) else ""
    directory = (home / override[2:] if override.startswith("~/") else
                 home if override == "~" else Path(override) if override else home / f".{app}")
    if not directory.is_absolute():
        logging.getLogger(__name__).warning("CC Switch 用量采集目录不是绝对路径")
        return
    archive = directory / ("archived_sessions" if app == "codex" else "projects/cc-switch-worker")
    try:
        archive.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=archive) as temp:
            for record in records:
                temp.write(json.dumps(record) + "\n")
            temp.flush()
            with suppress(FileExistsError):
                os.link(temp.name, archive / filename)
    except (OSError, ValueError):
        logging.getLogger(__name__).warning("无法保存 CC Switch 用量回执；不影响 CLI 输出")


def _usage_id(value):
    if isinstance(value, str):
        with suppress(ValueError):
            return str(UUID(value))
    return None


def _valid_token_counts(usage):
    return all(type(value) is int and 0 <= value < 2**32 for value in usage.values())


def _record_claude_usage(stdout):
    try:
        result = _json(stdout, "Claude")
    except WorkerError:
        return
    if not isinstance(result, dict) or result.get("type") != "result":
        return
    session = _usage_id(result.get("session_id"))
    model_usage = result.get("modelUsage")
    if not session or not isinstance(model_usage, dict):
        logging.getLogger(__name__).warning("Claude 未返回可导入 CC Switch 的用量回执")
        return
    keys = {"input_tokens": "inputTokens", "output_tokens": "outputTokens",
            "cache_read_input_tokens": "cacheReadInputTokens", "cache_creation_input_tokens": "cacheCreationInputTokens"}
    timestamp = datetime.now(timezone.utc).isoformat()
    records = []
    for model, usage in model_usage.items():
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_./:\[\]+-]{0,127}", model) or not isinstance(usage, dict):
            continue
        counts = {dest: usage[src] for dest, src in keys.items() if src in usage}
        if _valid_token_counts(counts) and any(counts.values()):
            records.append({
                "type": "assistant", "sessionId": session, "timestamp": timestamp,
                "message": {"id": f"cc-switch-worker-{session}-{model}", "model": model, "usage": counts},
            })
    if records:
        _write_usage("claude", f"{session}.jsonl", records)


def _record_codex_usage(stdout):
    thread_id = None
    usage = None
    for line in (stdout or b"").splitlines():
        if b'"thread.started"' not in line and b'"turn.completed"' not in line:
            continue
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "thread.started":
            thread_id = _usage_id(event.get("thread_id"))
        elif event.get("type") == "turn.completed":
            usage = event.get("usage")

    keys = ("input_tokens", "cached_input_tokens", "output_tokens")
    if not thread_id:
        return
    counts = {key: usage.get(key) for key in keys} if isinstance(usage, dict) else {}
    if not counts or not _valid_token_counts(counts) or counts["cached_input_tokens"] > counts["input_tokens"]:
        logging.getLogger(__name__).warning("Codex 未返回可导入 CC Switch 的用量回执")
        return
    if not (counts["input_tokens"] or counts["output_tokens"]):
        return

    timestamp = datetime.now(timezone.utc).isoformat()
    for_import = (
        ("session_meta", {"id": thread_id}),
        ("turn_context", {"model": _ASTRA}),
        ("event_msg", {"type": "token_count", "info": {"total_token_usage": counts}}),
    )
    _write_usage("codex", f"rollout-cc-switch-worker-{thread_id}.jsonl",
                 ({"timestamp": timestamp, "type": event_type, "payload": payload}
                  for event_type, payload in for_import))


def _record_cli_usage(name, stdout):
    if stdout:
        (_record_claude_usage if name == "Claude" else _record_codex_usage)(stdout)


def _codex_json(schema_value, prompt, timeout, codex_bin=None, *, fast=False, max_effort=False):
    binary = codex_bin or shutil.which("codex")
    if not binary:
        raise WorkerError("找不到 Codex CLI")
    names = ("CODEX_API_KEY", *_PROXY_ENV, *(name.upper() for name in _PROXY_ENV))
    with _scratch({name: os.environ[name] for name in names if name in os.environ}) as (root, work, env):
        argv = _codex_sandbox(binary, root, env)
        schema, final = root / "schema.json", root / "final.json"
        schema.write_text(json.dumps(schema_value), encoding="utf-8")
        argv.extend(["--no-daemon", "--ask-for-approval", "never", "exec", "--model", _ASTRA, "--json",
                     "--sandbox", "read-only", "--ephemeral", "--skip-git-repo-check",
                     "--ignore-user-config", "--ignore-rules", "--strict-config", "--enable", "skip_host_skill_discovery"])
        for flag in _DISABLED:
            argv.extend(["--disable", flag])
        for setting in ('web_search="disabled"', "mcp_servers={}", "project_doc_max_bytes=0",
                        'shell_environment_policy.inherit="none"'):
            argv.extend(["-c", setting])
        if fast or max_effort:
            catalog = _model_catalog(root, fast=fast, max_effort=max_effort)
            argv.extend(["-c", f"model_catalog_json={json.dumps(str(catalog))}"])
        if fast:
            argv.extend(["--enable", "fast_mode", "-c", 'service_tier="priority"'])
        if max_effort:
            argv.extend(["-c", 'model_reasoning_effort="max"'])
        argv.extend(["--output-schema", str(schema), "--output-last-message", str(final), "-"])
        _run("Codex max" if max_effort else "Codex Fast" if fast else "Codex", argv, prompt, work, env, timeout)
        try:
            # 最终文件跨越挂载命名空间，宿主不能跟随子进程留下的链接或阻塞在 FIFO 上。
            fd = os.open(final, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as output:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise WorkerError("Codex 最终输出不是普通文件")
                data = _json(output.read(), "Codex")
        except OSError:
            raise WorkerError("Codex 缺少最终输出文件") from None
    return data


def propose_hinge(context: dict, *, timeout=240) -> dict:
    options = list(context["options"])
    prompt = _prompt(
        "为 A 题设计一次可证伪的结构实验。选择 options 中一个与父版不同的机制，说明要改变哪个瓶颈、"
        "预计受益的原因和什么观测会推翻假设。若这些机制均无法解释瓶颈，另提一个具体的新机制，"
        "写入 next_mechanism 供下一轮实现评估；无额外提案时用空串。用中文，不推测尚未测量的成绩。", context, timeout)
    fields = dict(version={"type": "string", "enum": options}, hypothesis=_TEXT, falsifier=_TEXT, next_mechanism=_TEXT)
    data = _codex_json(_object(fields), prompt, timeout, fast=True)
    if (not isinstance(data, dict) or set(data) != set(fields) or data["version"] not in options
            or not all(_text(data[k]) and len(data[k]) <= 1024 for k in ("hypothesis", "falsifier"))
            or not isinstance(data["next_mechanism"], str) or len(data["next_mechanism"]) > 1024):
        raise WorkerError("Codex 返回的结构假设或反证条件无效")
    return data


def develop(context: dict, card: dict, *, timeout=600, codex_bin=None, max_effort=False) -> dict:
    prompt = _prompt(_DEVELOP_TASK, context, timeout, card=card)
    _check_card(card, _case_sizes(context))
    data = _codex_json(_object({
        "source": _TEXT, "language": {"type": "string", "enum": [context["language"]]}, "summary": _TEXT,
    }), prompt, timeout, codex_bin, max_effort=max_effort)
    if (not isinstance(data, dict) or set(data) != {"language", "source", "summary"} or data["language"] != context["language"]
            or not _text(data.get("source")) or not _text(data.get("summary"))):
        raise WorkerError("Codex 返回的源码、语言或摘要格式错误")
    source = data["source"].strip()
    if ("\x00" in source or source.startswith(("```", "~~~", "diff --git ", "*** Begin Patch", "@@ "))
            or source.endswith(("```", "~~~")) or (source.startswith("--- ") and "\n+++ " in source)):
        raise WorkerError("Codex 返回了无效源码、补丁或 Markdown 围栏")
    return data


def campaign_decision(context, phase, *, timeout=900):
    """三席自主选题和复盘共用显式的 Astra max 模型请求。"""
    if phase == "plan":
        shape_ids = [s["shape_id"] for s in context["cases"]]
        fields = dict(action={"type": "string", "enum": ["develop", "wait"]},
                      parent_sha={"type": "string", "enum": context["eligible_parents"]},
                      family=_TEXT, change=_TEXT, falsifier=_TEXT,
                      target_cases={"type": "array", "items": {"type": "string", "enum": shape_ids}},
                      borrowed_from=_TEXT, changed_premise=_TEXT,
                      parameter_range={"type": "array", "items": _TEXT}, parameter=_TEXT, waiting_reason=_TEXT)
        task = ("你是 context.agent 席。本轮独立负责完整算子。先读每位同伴最近的源码和评测、全部历史比较、"
                "逐形状及逐例数据、反例和共享经验。根据实际证据从 eligible_parents 选择完整正确的父版。"
                "measurements_by_source 汇总各源码的独立有效腿；晋升依据在 rounds.comparisons。"
                "若有 profiles，结合其源码与形状解释瓶颈，profiler 时长只用于诊断。"
                "提出一个具体的新机制、可推翻它的观测和预期形状。说明所借用的版本或想法。"
                "若重开已经停滞的父版与机制，在 changed_premise 写明改变的前提。"
                "可以自行开新路线。若要做可归因的参数扫描，parameter_range 列出预定配置，"
                "parameter 指本轮将固化进整份源码的一个配置。没有扫描时数组为空、parameter 为空串。"
                "若已有证据无法支持新实验，选择 wait 并具体说明 waiting_reason。"
                "为 develop 时 target_cases 使用清单原样的 shape_id。绝不猜测成绩或使用跟用例身份绑定的答案。")
    elif phase == "review":
        fields = dict(conclusion=_TEXT, next_step=_TEXT, entry_id=_TEXT, experience=_TEXT, waiting_reason=_TEXT)
        task = ("以 context.agent 的身份复盘 current_round。按已测的配对收益、自比波动、实测对照的 SHA、"
                "正确性和原始报告总结本轮。给出下一步可判定的实验。仅在有可复用新发现时写经验，注明形状、"
                "收益与反例或重试条件，entry_id 使用 1 到 96 个字母、数字、下划线、点或短横线。"
                "没有新增时将 entry_id 与 experience 同时留空。已证伪且本路线需要新证据时说明 waiting_reason，"
                "否则留空。复盘结论须与本地配对证据一致。")
    else:
        raise WorkerError("未知本地模型阶段")
    result = _codex_json(_object(fields), _prompt(task, context, timeout), timeout, max_effort=True)
    if not isinstance(result, dict) or result.keys() != fields.keys():
        raise WorkerError("固定模型返回的决策字段不完整")
    for key, spec in fields.items():
        if spec["type"] == "string" and not isinstance(result[key], str):
            raise WorkerError(f"固定模型的 {key} 字段类型错误")
    if phase == "plan":
        if result["action"] not in ("develop", "wait") or result["parent_sha"] not in context["eligible_parents"]:
            raise WorkerError("模型计划的动作或父版不符")
        if result["action"] == "develop":
            _check_card(result, set(shape_ids))
            span, value = result["parameter_range"], result["parameter"]
            if (not isinstance(span, list) or any(not _text(x) for x in span) or len(set(span)) != len(span)
                    or (span and value not in span) or (not span and value)):
                raise WorkerError("参数扫描的范围与当前选项不符")
        elif not _text(result["waiting_reason"]):
            raise WorkerError("等待新证据需要具体原因")
    elif (not _text(result["conclusion"]) or bool(result["entry_id"]) != bool(result["experience"].strip())
          or (result["entry_id"] and not re.fullmatch(r"[\w.-]{1,96}", result["entry_id"]))):
        raise WorkerError("模型复盘的结论或经验条目标识无效")
    return result
