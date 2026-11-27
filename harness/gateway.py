"""在 Claude CLI 子进程内复用桌面端已应用的静态网关，不迁移密钥文件。"""

import json
import os
import re
import shutil
import stat
from pathlib import Path
from urllib.parse import urlsplit


def desktop_dir():
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "Claude-3p"


def resolve_claude() -> str:
    binaries = [path for path in (desktop_dir() / "claude-code").glob("*/*/claude")
                if re.fullmatch(r"\d+\.\d+\.\d+", path.parents[1].name)
                and path.is_file() and os.access(path, os.X_OK)]
    if binaries:
        return str(max(binaries, key=lambda path: tuple(map(int, path.parents[1].name.split(".")))))
    binary = shutil.which("claude")
    if not binary:
        raise ValueError("未找到 Claude Code CLI（桌面附带版本或 PATH）")
    return binary


def private_json(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or info.st_size > 1024 * 1024):
            raise ValueError("网关配置必须是本人私有的小型普通文件")
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError("无法识别桌面网关配置格式")
    return result


def claude_env() -> dict[str, str]:
    library = desktop_dir() / "configLibrary"
    meta = private_json(library / "_meta.json")
    active = meta.get("appliedId")
    if (not isinstance(active, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", active)
            or not any(isinstance(item, dict) and item.get("id") == active for item in meta.get("entries", []))):
        raise ValueError("未找到已应用的桌面网关配置")
    profile = private_json(library / f"{active}.json")
    if profile.get("inferenceProvider") != "gateway" or profile.get("inferenceCredentialKind") != "static":
        raise ValueError("仅支持当前已应用的静态 gateway，不回退其他提供商")
    endpoint = profile.get("inferenceGatewayBaseUrl")
    raw_key = profile.get("inferenceGatewayApiKey")
    key = raw_key.strip(" \t\r\n") if isinstance(raw_key, str) else None
    if not isinstance(endpoint, str) or not isinstance(key, str) or not key or not key.isascii() or re.search(r"[\s\x00-\x1f\x7f]", key):
        raise ValueError("桌面网关地址或凭据无效")
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment or re.search(r"[\s\\]", endpoint)):
        raise ValueError("仅支持不内嵌凭据的 HTTPS 网关地址")
    header = {"bearer": "ANTHROPIC_AUTH_TOKEN", "x-api-key": "ANTHROPIC_API_KEY"}.get(
        profile.get("inferenceGatewayAuthScheme", "bearer"))
    if not header:
        raise ValueError("不支持该网关鉴权方式")
    allowed = ("HOME", "PATH", "LANG", "LC_ALL", "TMPDIR", "HTTPS_PROXY", "https_proxy",
               "NO_PROXY", "no_proxy", "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS")
    return {**{name: os.environ[name] for name in allowed if name in os.environ},
            "ANTHROPIC_BASE_URL": endpoint.rstrip("/"), header: key,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1"}
