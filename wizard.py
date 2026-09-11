#!/home/mohamed/.config/litellm/venv/bin/python
"""Single unified LiteLLM config wizard: keys -> validated -> models -> loop -> proxy test.

v2: quota-aware, health-aware deployment manager. Compiles provider
credentials + model capabilities + quota domains + routing policy into
LiteLLM YAML. LiteLLM remains the runtime router; this wizard is the
control plane / configuration compiler.
"""
__version__ = "3.0.0"
SCHEMA_VERSION = 2
import datetime
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    import yaml
except ImportError:
    # Re-exec with the bundled venv python (sibling ./venv) when launched
    # via system python (e.g. `./wizard.py` with /usr/bin/env python3).
    _venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
    if os.path.exists(_venv_py) and os.path.abspath(sys.executable) != os.path.abspath(_venv_py):
        os.execv(_venv_py, [_venv_py, os.path.abspath(__file__)] + sys.argv[1:])
    print("[!] PyYAML missing. Run with: ~/.config/litellm/venv/bin/python ~/.config/litellm/wizard.py")
    sys.exit(1)

CONFIG_DIR = os.path.expanduser("~/.config/litellm")
# Env overrides for safe testing (e.g. LITELLM_DB_FILE=/tmp/test_db.json)
DB_FILE = os.environ.get("LITELLM_DB_FILE", os.path.join(CONFIG_DIR, "providers_db.json"))
YAML_FILE = os.environ.get("LITELLM_YAML_FILE", os.path.join(CONFIG_DIR, "config.yaml"))
SECRET_FILE = os.environ.get("LITELLM_SECRET_FILE", os.path.join(CONFIG_DIR, ".master_key"))
# MASTER_KEY is resolved at runtime via get_master_key() (env -> secret file
# -> generated). No hard-coded credential. Kept as a deprecated fallback name
# so older imports do not crash; never used as an actual secret.
MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "")
PROXY_URL = "http://localhost:4000"
TIMEOUT = 15
UA = {"User-Agent": "litellm-wizard/2.0", "Accept": "application/json"}

# Proxy abstraction: pluggable backends, vault → proxy compile
PROXY_TYPES = ("litellm", "biofrost")
DEFAULT_PROXY_TYPE = "litellm"
PROXY_KEY = "_proxy"
HARNESS_KEY = "_harnesses"

# Verified against installed LiteLLM (1.100.0, litellm.types.router):
# - routing_strategy "usage-based-routing-v2" is valid.
# - enable_pre_call_checks (bool) is valid.
# - RetryPolicy supports ONLY: BadRequest, Authentication, Timeout,
#   RateLimit, ContentPolicyViolation, InternalServer retries.
#   (No ServiceUnavailable/Default keys — older configs emitted them.)
ROUTING_STRATEGY = "usage-based-routing-v2"
RETRY_POLICY = {
    "AuthenticationErrorRetries": 0,
    "BadRequestErrorRetries": 0,
    "ContentPolicyViolationErrorRetries": 0,
    "RateLimitErrorRetries": 0,
    "TimeoutErrorRetries": 1,
    "InternalServerErrorRetries": 1,
}
ROUTER_NUM_RETRIES = 1
ROUTER_COOLDOWN_TIME = 60
ROUTER_ALLOWED_FAILS = 1
# M2 routing/cooldowns: a 429 cools the deployment immediately (the
# next project's deployment in the same pool takes over); transient
# timeouts/5xx get ONE failure before cooling down (shorter penalty).
ALLOWED_FAILS_POLICY = {
    "RateLimitErrorAllowedFails": 0,
    "TimeoutErrorAllowedFails": 1,
    "InternalServerErrorAllowedFails": 1,
    "ServiceUnavailableErrorAllowedFails": 1,
    "BadGatewayErrorAllowedFails": 1,
}
COOLDOWN_RATE_LIMIT_S = 60.0   # 429: quota bucket needs a real pause
COOLDOWN_TRANSIENT_S = 30.0    # timeouts/5xx: likely temporary
PROVIDER_COOLDOWN_S = {"gemini": COOLDOWN_RATE_LIMIT_S}

PROVIDERS = {
    "1": {"id": "gemini", "name": "Google Gemini (AI Studio)", "prefix": "gemini/", "type": "api"},
    "2": {"id": "openrouter", "name": "OpenRouter", "prefix": "openrouter/", "type": "api"},
    "3": {"id": "anthropic", "name": "Anthropic (Claude)", "prefix": "anthropic/", "type": "api"},
    "4": {"id": "openai", "name": "OpenAI (GPT/o3)", "prefix": "openai/", "type": "api"},
    "5": {"id": "opencode_zen", "name": "OpenCode Zen", "prefix": "openai/", "base_url": "https://opencode.ai/zen/v1", "type": "custom_api"},
    "6": {"id": "tokenrouter", "name": "TokenRouter", "prefix": "openai/", "base_url": "https://api.tokenrouter.com/v1", "type": "custom_api"},
    "7": {"id": "zai", "name": "Z.AI (GLM Models)", "prefix": "openai/", "base_url": "https://api.z.ai/api/paas/v4", "type": "custom_api"},
    "8": {"id": "ollama_cloud", "name": "Ollama Cloud / Hosted Remote", "prefix": "ollama/", "type": "remote_ollama"},
    "9": {"id": "ollama_local", "name": "Local Ollama Engine", "prefix": "ollama/", "type": "local_ollama"},
    "10": {"id": "custom", "name": "Custom OpenAI-compatible endpoint", "prefix": "openai/", "type": "custom_api"},
}

# Provider behavior metadata for routing/display decisions. Only verified
# facts are encoded; everything else stays "unknown". Quota semantics are
# deliberately conservative — community numbers are NOT treated as facts.
# default_rpm/tpm carry (value, source) where source documents provenance.
PROVIDER_META = {
    "gemini": {"protocol": "gemini", "auth": "query-key", "quota_scope": "project",
               "trust_tier": "official", "free_tier": "free-tier",
               "default_rpm": (10, "conservative preset (free-tier safety; per Google project, not per key)"),
               "default_tpm": (None, "unknown — varies by model/project"),
               "supports_tools": "unknown", "supports_streaming": True,
               "supports_vision": "unknown", "supports_reasoning": "unknown"},
    "openrouter": {"protocol": "openai", "auth": "bearer", "quota_scope": "account",
                   "trust_tier": "gateway", "free_tier": "free+paid",
                   "default_rpm": (None, "unknown — account/key dependent"),
                   "default_tpm": (None, "unknown"), "supports_tools": "unknown",
                   "supports_streaming": True, "supports_vision": "unknown",
                   "supports_reasoning": "unknown"},
    "anthropic": {"protocol": "anthropic", "auth": "x-api-key", "quota_scope": "account",
                  "trust_tier": "official", "free_tier": "paid",
                  "default_rpm": (None, "unknown"), "default_tpm": (None, "unknown"),
                  "supports_tools": True, "supports_streaming": True,
                  "supports_vision": "unknown", "supports_reasoning": "unknown"},
    "openai": {"protocol": "openai", "auth": "bearer", "quota_scope": "account",
               "trust_tier": "official", "free_tier": "paid",
               "default_rpm": (None, "unknown"), "default_tpm": (None, "unknown"),
               "supports_tools": True, "supports_streaming": True,
               "supports_vision": "unknown", "supports_reasoning": "unknown"},
    "opencode_zen": {"protocol": "openai", "auth": "bearer", "quota_scope": "account",
                     "trust_tier": "established", "free_tier": "free-tier",
                     "default_rpm": (None, "unknown"), "default_tpm": (None, "unknown"),
                     "supports_tools": "unknown", "supports_streaming": "unknown",
                     "supports_vision": "unknown", "supports_reasoning": "unknown"},
    "tokenrouter": {"protocol": "openai", "auth": "bearer", "quota_scope": "account",
                    "trust_tier": "gateway", "free_tier": "free-tier",
                    "default_rpm": (None, "unknown"), "default_tpm": (None, "unknown"),
                    "supports_tools": "unknown", "supports_streaming": "unknown",
                    "supports_vision": "unknown", "supports_reasoning": "unknown"},
    "zai": {"protocol": "openai", "auth": "bearer", "quota_scope": "account",
            "trust_tier": "established", "free_tier": "free+paid",
            "default_rpm": (None, "unknown"), "default_tpm": (None, "unknown"),
            "supports_tools": "unknown", "supports_streaming": "unknown",
            "supports_vision": "unknown", "supports_reasoning": "unknown"},
    "ollama_cloud": {"protocol": "ollama", "auth": "bearer", "quota_scope": "account",
                     "trust_tier": "official", "free_tier": "free+paid",
                     "default_rpm": (None, "unknown"), "default_tpm": (None, "unknown"),
                     "supports_tools": "unknown", "supports_streaming": True,
                     "supports_vision": "unknown", "supports_reasoning": "unknown"},
    "ollama_local": {"protocol": "ollama", "auth": "none", "quota_scope": "endpoint",
                     "trust_tier": "official", "free_tier": "local",
                     "default_rpm": (None, "no upstream quota — local engine"),
                     "default_tpm": (None, "no upstream quota — local engine"),
                     "supports_tools": "unknown", "supports_streaming": True,
                     "supports_vision": "unknown", "supports_reasoning": "unknown"},
    "custom": {"protocol": "openai", "auth": "bearer", "quota_scope": "unknown",
               "trust_tier": "custom", "free_tier": "unknown",
               "default_rpm": (None, "unknown"), "default_tpm": (None, "unknown"),
               "supports_tools": "unknown", "supports_streaming": "unknown",
               "supports_vision": "unknown", "supports_reasoning": "unknown"},
}

TRUST_ORDER = ["official", "established", "gateway", "custom", "unknown"]

QUOTA_SPLIT_MODES = ("shared-estimate", "per-deployment-split", "manual")
VALIDATION_MODES = ("FAST", "STRICT", "SAMPLE")

MODEL_HINTS = {
    "opencode_zen": "Examples: big-pickle, mimo-v2.5-free, nemotron-3-ultra-free (bare IDs, no opencode/ prefix)",
    "tokenrouter": "Console: tokenrouter.com/console/token, base https://api.tokenrouter.com/v1 (sk-... keys). api.tokenrouter.io (tr_... keys) also accepted — wizard auto-detects.",
    "zai": "Examples: glm-4.7-flash, glm-5.3 (check /models for current IDs)",
    "ollama_cloud": "Examples: gemma4:31b, nemotron-3-ultra, gpt-oss:120b",
    "ollama_local": "Examples: qwen2.5-coder:7b, llama3.3:70b",
    "gemini": "Examples: gemini-3.8-flash, gemini-3.5-flash, gemini-3.1-pro-preview",
    "openrouter": "Examples: minimax/minimax-m3:free, nvidia/nemotron-3.5-lightning:free",
    "custom": "Pick from the live catalog fetched from YOUR base URL — no guessing needed",
}


def _atomic_write_json(path, data):
    """Atomic JSON write (temp file + fsync + rename). Never half-writes."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_text(path, text, mode=0o600):
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            raw = json.load(f)
        return migrate_db(raw)
    return migrate_db({})


def save_db(data):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    data = migrate_db(data)  # normalize before persisting
    _atomic_write_json(DB_FILE, data)


def snippet(k):
    k = str(k)
    return f"...{k[-6:]}" if len(k) > 6 else k


ALIAS_KEY = "_aliases"


def _needs_drop_params(pid, m):
    """Per-model drop_params: only for models known to reject tool_choice.

    Global drop_params broke gemma4:31b agent tool-calling (raw JSON output),
    so keep surgical. Currently only free/gpt-5.6-luna needs it.
    """
    bare = (m.split("/")[-1] if "/" in m else m).lower()
    # strip free markers for comparison
    bare = bare.removesuffix(":free")
    bare = bare.removesuffix("-free")
    return bare == "gpt-5.6-luna"


def _get_aliases(db_data):
    """Return unified alias map {canonical: [{provider, model},...]} or {}."""
    a = db_data.get(ALIAS_KEY)
    if isinstance(a, dict):
        return a
    # legacy key
    b = db_data.get("_unified")
    if isinstance(b, dict):
        return b
    return {}


# ================= v2 architecture: credentials / quota / compiler ==========

ROLES_KEY = "_roles"
QUOTA_KEY = "_quota_domains"
SETTINGS_KEY = "_settings"
HEALTH_KEY = "_health"
PERF_KEY = "_performance"  # probe latency memory: "pid:model" -> {samples, ...}

DEFAULT_SETTINGS = {
    "routing_preference": "balanced",  # balanced | capacity-first | reliability-first
    "quota_split_mode": "per-deployment-split",  # shared-estimate | per-deployment-split | manual
    "validation_mode": "FAST",  # FAST | STRICT | SAMPLE
    "sample_size": 2,
}


# ---------- master key (no hard-coded credential) ----------

def get_master_key():
    """Resolve the gateway master key: env -> secret file -> generate.

    Never falls back to a hard-coded value. Generated keys are persisted
    with mode 0600 so restarts reuse the same credential.
    """
    env = os.environ.get("LITELLM_MASTER_KEY", "").strip()
    if env:
        return env
    try:
        if os.path.exists(SECRET_FILE):
            with open(SECRET_FILE, "r") as f:
                saved = f.read().strip().splitlines()
                if saved and saved[0].strip():
                    return saved[0].strip()
    except OSError:
        pass
    new_key = "sk-litellm-" + secrets.token_urlsafe(32)
    try:
        _atomic_write_text(SECRET_FILE, new_key + "\n", mode=0o600)
    except OSError:
        pass
    return new_key


def _mask_secret(s):
    s = str(s or "")
    if len(s) <= 6:
        return "***"
    return f"...{s[-4:]}"


# ---------- credentials ----------

def credential_id(secret):
    """Stable, secret-free credential identity (never the raw key)."""
    h = hashlib.sha256(str(secret).encode()).hexdigest()[:12]
    return f"cred-{h}"


def _provider_entry(db, pid):
    e = db.get(pid)
    if not isinstance(e, dict):
        e = {}
        db[pid] = e
    return e


def normalize_credentials(pdata):
    """Ensure pdata has credential records mirroring legacy keys[].

    credentials[] is the source of truth; keys[] is kept in sync for
    backward compatibility. Stable IDs survive reordering.
    """
    keys = list(pdata.get("keys") or [])
    creds = pdata.get("credentials")
    if not isinstance(creds, list):
        creds = []
    by_secret = {}
    for c in creds:
        if isinstance(c, dict) and c.get("secret"):
            by_secret[str(c["secret"])] = c
    merged = []
    seen_ids = set()
    for k in keys:
        k = str(k)
        c = by_secret.get(k)
        if c is None:
            c = {"id": credential_id(k), "secret": k, "label": "",
                 "quota_domain": "", "enabled": True, "project_id": "",
                 "validation": {"status": "unknown", "checked_at": "",
                                "message": ""}}
        else:
            c = dict(c)
            if not c.get("id"):
                c["id"] = credential_id(k)
            c.setdefault("enabled", True)
            c.setdefault("quota_domain", "")
            c.setdefault("label", "")
            c.setdefault("project_id", "")
            c.setdefault("validation", {"status": "unknown", "checked_at": "",
                                        "message": ""})
        # dedup by secret; keep first
        if c["id"] in seen_ids and any(x.get("secret") == k for x in merged):
            continue
        seen_ids.add(c["id"])
        merged.append(c)
    # drop credential records whose secret vanished from keys[]
    keyset = {str(k) for k in keys}
    merged = [c for c in merged if str(c.get("secret")) in keyset]
    pdata["credentials"] = merged
    pdata["keys"] = [str(c["secret"]) for c in merged if c.get("enabled") is not False or True]
    # keys[] keeps ALL secrets (enabled or quarantined); enabled flag lives on cred
    return merged


def iter_credentials(pdata, enabled_only=False):
    creds = pdata.get("credentials")
    if not isinstance(creds, list) or not creds:
        normalize_credentials(pdata)
        creds = pdata.get("credentials", [])
    for c in creds:
        if not isinstance(c, dict):
            continue
        if enabled_only and (c.get("enabled") is False or c.get("quarantined")):
            continue
        if c.get("secret"):
            yield c


def default_quota_domain_id(pid, cred):
    """Automatic default: one credential = one quota domain (secret-free).

    If the credential has a project_id (e.g., Google project), group by
    project instead — keys from one project share one speed limit.
    """
    proj = cred.get("project_id") or ""
    if proj:
        return f"project:{pid}:{proj}"
    return f"credential:{cred.get('id', 'unknown')}"


def effective_quota_domain(pid, cred):
    """The domain a credential REALLY belongs to right now.

    An explicit named domain wins — unless it is an automatic
    per-credential default that went stale after a project_id was set
    (the project grouping is more truthful than the old default).
    """
    qd = (cred.get("quota_domain") or "") if isinstance(cred, dict) else ""
    if qd and not qd.startswith("credential:"):
        return qd
    proj = (cred.get("project_id") or "") if isinstance(cred, dict) else ""
    if proj:
        return f"project:{pid}:{proj}"
    return qd or default_quota_domain_id(pid, cred)


# ---------- database migration ----------

def migrate_db(raw):
    """Migrate any legacy DB shape to the current schema. Idempotent.

    Preserves: provider keys/models/endpoints/labels/base_url, aliases
    (both _aliases and legacy _unified), custom endpoints. Never deletes
    valid user configuration; stale alias members are retained here and
    cleaned by normalize_aliases() with user visibility.
    """
    if not isinstance(raw, dict):
        raw = {}
    db = raw
    # legacy alias key -> canonical
    if ALIAS_KEY not in db and isinstance(db.get("_unified"), dict):
        db[ALIAS_KEY] = db["_unified"]
    if ALIAS_KEY in db and not isinstance(db[ALIAS_KEY], dict):
        db[ALIAS_KEY] = {}
    db.setdefault(ALIAS_KEY, {})
    # dedup alias members, drop malformed entries
    for canon in list(db[ALIAS_KEY].keys()):
        members = db[ALIAS_KEY].get(canon)
        if not isinstance(members, list):
            db[ALIAS_KEY][canon] = []
            continue
        seen, clean = set(), []
        for m in members:
            if not isinstance(m, dict):
                continue
            pid, mod = m.get("provider"), m.get("model")
            if not pid or not mod or (pid, mod) in seen:
                continue
            seen.add((pid, mod))
            clean.append({"provider": pid, "model": mod})
        db[ALIAS_KEY][canon] = clean
    # remove legacy key only after successful copy
    if "_unified" in db and ALIAS_KEY in db:
        try:
            del db["_unified"]
        except KeyError:
            pass
    # settings
    if not isinstance(db.get(SETTINGS_KEY), dict):
        db[SETTINGS_KEY] = {}
    for k, v in DEFAULT_SETTINGS.items():
        db[SETTINGS_KEY].setdefault(k, v)
    # roles
    if not isinstance(db.get(ROLES_KEY), dict):
        db[ROLES_KEY] = {}
    # quota domains
    if not isinstance(db.get(QUOTA_KEY), dict):
        db[QUOTA_KEY] = {}
    # health
    if not isinstance(db.get(HEALTH_KEY), dict):
        db[HEALTH_KEY] = {}
    # proxy (pluggable, vault is source)
    if not isinstance(db.get(PROXY_KEY), dict):
        db[PROXY_KEY] = {}
    db[PROXY_KEY].setdefault("type", DEFAULT_PROXY_TYPE)
    if db[PROXY_KEY].get("type") not in PROXY_TYPES:
        db[PROXY_KEY]["type"] = DEFAULT_PROXY_TYPE
    db[PROXY_KEY].setdefault("routing", ROUTING_STRATEGY)
    # harnesses (detected, not stored)
    # providers: normalize credentials + defaults
    for pid, pdata in list(db.items()):
        if pid.startswith("_"):
            continue
        if not isinstance(pdata, dict):
            db[pid] = {"keys": [], "models": [], "endpoints": []}
            pdata = db[pid]
        pdata.setdefault("keys", [])
        pdata.setdefault("models", [])
        pdata.setdefault("endpoints", [])
        if not isinstance(pdata["keys"], list):
            pdata["keys"] = []
        if not isinstance(pdata["models"], list):
            pdata["models"] = []
        if not isinstance(pdata["endpoints"], list):
            pdata["endpoints"] = []
        # dedup models preserving order
        seen_m, clean_m = set(), []
        for m in pdata["models"]:
            if m not in seen_m:
                seen_m.add(m)
                clean_m.append(m)
        pdata["models"] = clean_m
        creds = normalize_credentials(pdata)
        # default quota domains: one per credential (secret-free IDs)
        for c in creds:
            if not c.get("quota_domain"):
                c["quota_domain"] = default_quota_domain_id(pid, c)
            qd = c["quota_domain"]
            if qd.startswith("credential:") and qd not in db[QUOTA_KEY]:
                db[QUOTA_KEY][qd] = {"rpm": None, "tpm": None, "rpd": None,
                                     "confidence": "unknown", "provider": pid,
                                     "updated_at": "", "source": "automatic default"}
        # legacy per-provider quota block migration
        legacy_q = pdata.pop("quota", None)
        if isinstance(legacy_q, dict):
            for c in creds:
                qd = c.get("quota_domain")
                if qd and qd not in db[QUOTA_KEY]:
                    db[QUOTA_KEY][qd] = {"rpm": legacy_q.get("rpm"),
                                         "tpm": legacy_q.get("tpm"),
                                         "rpd": legacy_q.get("rpd"),
                                         "confidence": legacy_q.get("confidence", "manual"),
                                         "provider": pid, "updated_at": "",
                                         "source": "migrated provider quota"}
        pdata.setdefault("disabled", False)
    db["_schema_version"] = SCHEMA_VERSION
    return db


def validate_db(db):
    """Return list of diagnostic strings; empty means structurally sound."""
    issues = []
    for pid, pdata in db.items():
        if pid.startswith("_"):
            continue
        if not isinstance(pdata, dict):
            issues.append(f"provider '{pid}' is not an object")
            continue
        for c in pdata.get("credentials", []):
            if not isinstance(c, dict) or not c.get("id") or not c.get("secret"):
                issues.append(f"provider '{pid}' has malformed credential record")
            elif not c.get("quota_domain"):
                issues.append(f"provider '{pid}' credential {c.get('id')} lacks quota domain")
    for qd, q in (db.get(QUOTA_KEY) or {}).items():
        if not isinstance(q, dict):
            issues.append(f"quota domain '{qd}' is not an object")
    for canon, members in _get_aliases(db).items():
        if not members:
            issues.append(f"alias '{canon}' has zero members")
        for m in members or []:
            if not isinstance(m, dict) or not m.get("provider") or not m.get("model"):
                issues.append(f"alias '{canon}' has malformed member")
    for role, spec in (db.get(ROLES_KEY) or {}).items():
        if not isinstance(spec, dict) or not spec.get("pools"):
            issues.append(f"role '{role}' has no pools")
    return issues


def normalize_aliases(db):
    """Remove stale members (model no longer configured) from active routes.

    Returns (cleaned_count, emptied_aliases). Stale members are dropped from
    the alias map but reported; empty aliases are removed. Idempotent.
    """
    aliases = _get_aliases(db)
    if ALIAS_KEY not in db or not isinstance(db[ALIAS_KEY], dict):
        db[ALIAS_KEY] = dict(aliases)
    cleaned, emptied = 0, []
    for canon in list(db[ALIAS_KEY].keys()):
        members = db[ALIAS_KEY].get(canon) or []
        kept = []
        for m in members:
            if not isinstance(m, dict):
                cleaned += 1
                continue
            pid, mod = m.get("provider"), m.get("model")
            pdata = db.get(pid) if pid else None
            if pid and mod and isinstance(pdata, dict) and mod in (pdata.get("models") or []):
                kept.append(m)
            else:
                cleaned += 1
        if not kept:
            del db[ALIAS_KEY][canon]
            emptied.append(canon)
        else:
            db[ALIAS_KEY][canon] = kept
    return cleaned, emptied


# ---------- quota domains ----------

def ensure_quota_domain(db, qd_id, provider="", rpm=None, tpm=None,
                        rpd=None, confidence="manual", source="user"):
    db.setdefault(QUOTA_KEY, {})
    q = db[QUOTA_KEY].get(qd_id)
    if not isinstance(q, dict):
        q = {}
        db[QUOTA_KEY][qd_id] = q
    if rpm is not None:
        q["rpm"] = rpm
    if tpm is not None:
        q["tpm"] = tpm
    if rpd is not None:
        q["rpd"] = rpd
    q.setdefault("confidence", confidence)
    if provider:
        q.setdefault("provider", provider)
    q["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    q["source"] = source
    for k in ("rpm", "tpm", "rpd", "confidence", "provider", "updated_at", "source"):
        q.setdefault(k, None if k in ("rpm", "tpm", "rpd") else "")
    return q


def quota_members(db, qd_id):
    """All (pid, credential) pairs assigned to a quota domain."""
    out = []
    for pid, pdata in db.items():
        if pid.startswith("_") or not isinstance(pdata, dict):
            continue
        for c in iter_credentials(pdata):
            if c.get("quota_domain") == qd_id:
                out.append((pid, c))
    return out


def estimate_capacity(db, members):
    """Effective capacity = sum over UNIQUE quota domains (never keys x RPM).

    members: iterable of (pid, model) or deployment dicts. Returns dict with
    known/unknown breakdown, labeled as estimates.
    """
    domains = set()
    for m in members:
        if isinstance(m, dict):
            qd = m.get("quota_domain")
            if qd:
                domains.add(qd)
        else:
            pid, mod = m
            pdata = db.get(pid) if isinstance(db.get(pid), dict) else None
            if pdata:
                # union ALL credential domains backing this pool: capacity is
                # per-domain, so every backing domain counts (never keys x RPM,
                # never just the first credential's domain).
                for c in iter_credentials(pdata):
                    qd = effective_quota_domain(pid, c)
                    if qd:
                        domains.add(qd)
    rpm_known, tpm_known, unknown = 0, 0, 0
    for qd in domains:
        q = (db.get(QUOTA_KEY) or {}).get(qd) or {}
        r, t = q.get("rpm"), q.get("tpm")
        if isinstance(r, (int, float)) and r > 0:
            rpm_known += r
        else:
            unknown += 1
        if isinstance(t, (int, float)) and t > 0:
            tpm_known += t
    return {"rpm_known": rpm_known, "tpm_known": tpm_known,
            "domains": len(domains), "unknown_domains": unknown}


def resolve_deployment_limits(db, pid, cred, model):
    """Limit precedence: deployment override > quota-domain per-model >
    quota-domain > provider known default > conservative default > unknown
    (None). Google free tiers are per project+model, so a per-model
    override on the domain beats the domain-wide number."""
    dep_over = (cred.get("limits") or {}) if isinstance(cred, dict) else {}
    qd = effective_quota_domain(pid, cred)
    q = (db.get(QUOTA_KEY) or {}).get(qd) or {}
    per_model = q.get("per_model") if isinstance(q, dict) else None
    pm = {}
    if isinstance(per_model, dict):
        cand = per_model.get(model)
        if isinstance(cand, dict):
            pm = cand
    meta = PROVIDER_META.get(pid) or PROVIDER_META.get("custom")
    if pid.startswith("custom_"):
        meta = PROVIDER_META.get("custom")
    rpm = dep_over.get("rpm", None)
    tpm = dep_over.get("tpm", None)
    if rpm is None:
        rpm = pm.get("rpm")
    if tpm is None:
        tpm = pm.get("tpm")
    if rpm is None:
        rpm = q.get("rpm")
    if tpm is None:
        tpm = q.get("tpm")
    if rpm is None and meta:
        rpm = (meta.get("default_rpm") or (None,))[0]
    if tpm is None and meta:
        tpm = (meta.get("default_tpm") or (None,))[0]
    # local ollama has no upstream quota — leave None (unlimited)
    return rpm, tpm


def split_shared_quota(domain_rpm, n_deployments, mode=None):
    """Conservative per-deployment allocation for a shared quota domain.

    Never emits n x domain quota. per-deployment-split divides evenly
    (floor, min 1); shared-estimate keeps domain value on first deployment
    only... in practice we always divide — overstatement is worse than
    understatement for free tiers.
    """
    if not isinstance(domain_rpm, (int, float)) or domain_rpm <= 0:
        return None
    if not n_deployments or n_deployments <= 1:
        return int(domain_rpm)
    mode = mode or "per-deployment-split"
    if mode == "manual":
        return int(domain_rpm)
    return max(1, int(domain_rpm // n_deployments))


# ---------- capability model ----------

def classify_model_tier(model_id):
    """Capability tier from model name. Conservative: unknown on doubt."""
    s = (model_id or "").lower()
    if re.search(r"reason|thinking|qwq|(?<![a-z0-9])r1(?![a-z0-9])", s):
        return "reasoning"
    if (re.search(r"(?<![a-z0-9])(pro|ultra|max)(?![a-z0-9])", s)
            or re.search(r"70b|120b|550b|32b", s)):
        return "pro"
    if (re.search(r"(?<![a-z0-9])(lite|mini)(?![a-z0-9])", s)
            or "lightning" in s or "flash-lite" in s):
        return "lite"
    if "flash" in s:
        return "flash"
    if "luna" in s or "pickle" in s:
        return "unknown"
    return "unknown"


# Provider families for context-window lookup (installed litellm map keys
# are family-prefixed: "gemini/...", "openai/...", ...).
_CTX_FAMILIES = {
    "gemini": ("gemini/",),
    "openrouter": ("openrouter/",),
    "anthropic": ("anthropic/",),
    "openai": ("openai/",),
    "zai": ("z-ai/",),
    "ollama_cloud": ("ollama/",),
    "ollama_local": ("ollama/",),
    "tokenrouter": ("openai/", "gemini/", "anthropic/"),
    "opencode_zen": ("openai/", "gemini/", "anthropic/"),
    "custom": ("openai/", "gemini/", "anthropic/"),
}


def lookup_context_window(pid, model_id):
    """Context window (input tokens) from the installed litellm model map.

    Exact id-string matches only — same model id, optionally with the
    provider family prefix (``gemini/flash`` for ``flash``). Never fuzzy,
    never guessed: returns an int or None. litellm stays a lazy optional
    import so wizard.py remains stdlib + PyYAML.
    """
    up = str(model_id or "").strip()
    if not up:
        return None
    try:
        from litellm import model_cost as _mc
    except Exception:  # noqa: BLE001 -- no litellm installed means unknown ctx
        return None
    core = up.split("/")[-1]
    stems = []
    for stem in (core, core.removesuffix(":free"), up, up.removesuffix(":free")):
        if stem and stem not in stems:
            stems.append(stem)
    cands: list[str] = []
    for stem in stems:
        cands.append(stem)
        for fam in _CTX_FAMILIES.get(pid, ("openai/", "gemini/", "anthropic/")):
            cands.append(fam + stem)
    if pid == "openrouter" and "/" in up:  # openrouter/<provider>/<model> keys
        cands.append("openrouter/" + up)
        cands.append("openrouter/" + up.removesuffix(":free"))
    for key in cands:
        entry = _mc.get(key)
        if isinstance(entry, dict):
            n = entry.get("max_input_tokens")
            if isinstance(n, bool):
                continue
            if isinstance(n, (int, float)) and n > 0:
                return int(n)
    return None


def infer_capabilities(pid, model_id):
    """Capability metadata. Unknown unless verified — never fabricate."""
    s = (model_id or "").lower()
    tier = classify_model_tier(model_id)
    caps: dict = {"tier": tier, "tools": "unknown", "streaming": "unknown",
                  "vision": "unknown", "reasoning": "unknown",
                  "context_window": lookup_context_window(pid, model_id) or "unknown",
                  "structured_output": "unknown"}
    if tier == "reasoning":
        caps["reasoning"] = True
    if pid in ("anthropic", "openai"):
        caps["tools"] = True
        caps["streaming"] = True
    if pid == "gemini":
        caps["streaming"] = True
    if "vision" in s or "image" in s:
        caps["vision"] = True
    return caps


def capability_compatible(a, b):
    """Two capability dicts are auto-mergeable only when tiers match (or one
    is unknown-but-plausible) and no explicit incompatibility exists."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    ta, tb = a.get("tier"), b.get("tier")
    if ta != tb:
        # unknown tier never auto-merges with a known tier
        return False
    if ta in ("pro", "reasoning") and tb in ("pro", "reasoning") and ta != tb:
        return False
    for k in ("tools", "vision", "reasoning"):
        va, vb = a.get(k), b.get(k)
        if va is True and vb is False:
            return False
        if va is False and vb is True:
            return False
    return True


def _group_safety(pid_a, model_a, pid_b, model_b):
    """Return AUTO / SUGGESTED / MANUAL for a pair of model references."""
    if _alias_stem(model_a) != _alias_stem(model_b):
        return "MANUAL"
    ca, cb = infer_capabilities(pid_a, model_a), infer_capabilities(pid_b, model_b)
    if capability_compatible(ca, cb) and ca.get("tier") != "unknown":
        return "AUTO"
    if capability_compatible(ca, cb):
        return "SUGGESTED"
    # same stem but incompatible/unknown tier -> suggest, never silent
    return "SUGGESTED"


# ---------- health / error classification ----------

def classify_http_error(status, raw=""):
    """Map HTTP outcome to health class. 429 != auth error; 401/403 !=
    server failure; timeout != permanent failure."""
    raw = str(raw or "")
    if status == 200:
        return "OK"
    if status == 429 or raw.startswith("RATELIMIT"):
        return "RATE_LIMITED"
    if status in (401, 403):
        return "AUTH_ERROR"
    if status == 404:
        return "BAD_REQUEST"
    if status == 400:
        return "BAD_REQUEST"
    if status in (408, 504) or "timeout" in raw.lower() or "timed out" in raw.lower():
        return "TIMEOUT"
    if status in (500, 502, 503):
        return "SERVER_ERROR"
    if status is None:
        if "timeout" in raw.lower():
            return "TIMEOUT"
        return "UNAVAILABLE"
    return "UNKNOWN"


def record_validation(db, pid, cred_id, status, message=""):
    """Persist credential validation state (valid vs health vs quota)."""
    pdata = db.get(pid)
    if not isinstance(pdata, dict):
        return
    for c in pdata.get("credentials", []) or []:
        if isinstance(c, dict) and c.get("id") == cred_id:
            c["validation"] = {"status": status,
                               "checked_at": datetime.datetime.now().isoformat(timespec="seconds"),
                               "message": str(message)[:200]}
            break
    db.setdefault(HEALTH_KEY, {})[f"{pid}:{cred_id}"] = {
        "status": status, "message": str(message)[:200],
        "at": datetime.datetime.now().isoformat(timespec="seconds")}


def deployment_health(db, pid, model):
    """Aggregate health label for a (provider, model) deployment pool."""
    h = db.get(HEALTH_KEY) or {}
    states = [v.get("status") for k, v in h.items()
              if k.startswith(f"{pid}:") and isinstance(v, dict)]
    if not states:
        return "unknown"
    if any(s == "ok" for s in states):
        return "healthy" if "throttled" not in states else "partially-throttled"
    if any(s == "throttled" for s in states):
        return "throttled"
    if any(s == "invalid" for s in states):
        return "invalid"
    return "unknown"


# ---------- performance memory (probe latency, rolling window) ----------

PERF_WINDOW = 5  # remembered probe latencies per (provider, model)


def record_probe_latency(db, pid, model, seconds):
    """Remember one probe latency (seconds) for a (provider, model).

    Rolling window of the last PERF_WINDOW samples; failures and
    non-numeric values are ignored. Never secrets — ids only.
    """
    if not model or not isinstance(seconds, (int, float)) or seconds < 0:
        return
    perf = db.setdefault(PERF_KEY, {})
    rec = perf.get(f"{pid}:{model}")
    if not isinstance(rec, dict):
        rec = {"samples": []}
    samples = [s for s in (rec.get("samples") or [])
               if isinstance(s, (int, float)) and s > 0]
    samples.append(round(float(seconds), 3))
    rec["samples"] = samples[-PERF_WINDOW:]
    rec["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    perf[f"{pid}:{model}"] = rec


def probe_latency(db, pid, model):
    """Mean latency (seconds) from the recent probe window, or None."""
    rec = (db.get(PERF_KEY) or {}).get(f"{pid}:{model}")
    if not isinstance(rec, dict):
        return None
    samples = [s for s in (rec.get("samples") or [])
               if isinstance(s, (int, float)) and s > 0]
    if not samples:
        return None
    return round(sum(samples) / len(samples), 3)


# ---------- deployment compiler ----------

def _provider_info(db, pid):
    p = next((p for p in PROVIDERS.values() if p["id"] == pid), None)
    if p is not None:
        return p
    if pid == "custom" or pid.startswith("custom_"):
        pdata = db.get(pid, {})
        if pdata.get("base_url"):
            return {"id": pid, "name": pdata.get("label", pid),
                    "prefix": "openai/", "type": "custom_api"}
    return None


# Builtin OpenAI-compatible bases (first-run defaults, derived from
# PROVIDERS so they cannot drift; a stored override always wins via
# effective_base_url).
_BUILTIN_BASES = {p["id"]: p["base_url"] for p in PROVIDERS.values()
                  if p.get("base_url")}


def effective_base_url(db, pid, endpoint=None):
    """Single source of truth for custom_api base URLs.

    Precedence: explicit ``endpoint`` arg > stored ``endpoints[0]`` >
    stored ``base_url`` > builtin default. Returns None when nothing is
    known (true-custom providers before their URL is entered). ``db`` may
    be None (stateless callers fall back to the builtin).
    """
    if endpoint:
        return str(endpoint).rstrip("/")
    pdata = {}
    if isinstance(db, dict):
        entry = db.get(pid)
        if isinstance(entry, dict):
            pdata = entry
    stored = (pdata.get("endpoints") or [None])[0] or pdata.get("base_url")
    if stored:
        return str(stored).rstrip("/")
    if pid in _BUILTIN_BASES:
        return _BUILTIN_BASES[pid]
    info = None
    try:
        info = _provider_info(db if isinstance(db, dict) else {}, pid)
    except Exception:  # noqa: BLE001 -- provider lookup must never fail resolution
        info = None
    if isinstance(info, dict) and info.get("base_url"):
        return str(info["base_url"]).rstrip("/")
    return None


def _litellm_model_for_provider(pid, model_id, ptype):
    if ptype in ("local_ollama", "remote_ollama"):
        return f"ollama/{model_id}"
    if ptype == "custom_api":
        if pid in ("tokenrouter", "custom") or pid.startswith("custom_"):
            return f"openai/{model_id}"
        bare = model_id.split("/")[-1] if "/" in model_id else model_id
        return f"openai/{bare}"
    prefix = next((p.get("prefix", "") for p in PROVIDERS.values() if p["id"] == pid), "")
    if pid.startswith("custom_"):
        prefix = "openai/"
    if prefix and not model_id.startswith(prefix):
        return f"{prefix}{model_id}"
    return model_id


def _gateway_alias_for_model(pid, model_id, ptype):
    if ptype == "api":
        return model_id.split("/")[-1] if "/" in model_id else model_id
    if ptype == "custom_api":
        if pid in ("tokenrouter", "custom") or pid.startswith("custom_"):
            return model_id.split("/")[-1] if "/" in model_id else model_id
        return model_id.split("/")[-1] if "/" in model_id else model_id
    return model_id


def build_deployments(db):
    """Compile DB -> deployment list. One deployment = one provider +
    credential + endpoint + upstream model + logical pool + quota domain +
    limits + capabilities + health + enabled. Deterministic ordering:
    pool -> provider -> quota-domain -> credential."""
    normalize_aliases(db)
    aliases = _get_aliases(db)
    member_pool = {}
    for canon, members in aliases.items():
        for m in members or []:
            if isinstance(m, dict) and m.get("provider") and m.get("model"):
                member_pool[(m["provider"], m["model"])] = canon
    seen = set()
    split_mode = (db.get(SETTINGS_KEY) or {}).get("quota_split_mode", "per-deployment-split")
    # count deployments per (quota domain, upstream model) for shared-quota
    # splitting. Upstream RPM limits are conventionally per model within a
    # domain (e.g. Gemini free tier is per model per project), so deployments
    # of the SAME model in one domain split that model's quota; different
    # models do not dilute each other. Documented as an estimate.
    domain_counts = {}
    pending = []
    for pid in sorted([k for k in db if not k.startswith("_")]):
        pdata = db.get(pid)
        if not isinstance(pdata, dict) or pdata.get("disabled"):
            continue
        p_info = _provider_info(db, pid)
        if not p_info:
            continue
        ptype = p_info.get("type")
        endpoints = pdata.get("endpoints", []) or []
        base_url = pdata.get("base_url") or p_info.get("base_url")
        for model_id in pdata.get("models", []) or []:
            if not model_id or not isinstance(model_id, str):
                continue
            pool = member_pool.get((pid, model_id)) or _gateway_alias_for_model(pid, model_id, ptype)
            caps = infer_capabilities(pid, model_id)
            if ptype == "local_ollama":
                eps = endpoints or ["http://localhost:11434"]
                for ep in sorted(eps):
                    key = (pool, pid, "", ep, model_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    pending.append({"logical_model": pool, "provider": pid,
                                    "credential_id": "", "quota_domain": f"endpoint:{ep}",
                                    "endpoint": ep, "upstream_model": model_id,
                                    "rpm": None, "tpm": None, "capabilities": caps,
                                    "health": deployment_health(db, pid, model_id),
                                    "enabled": True})
                    domain_counts[(f"endpoint:{ep}", model_id)] = \
                        domain_counts.get((f"endpoint:{ep}", model_id), 0) + 1
                continue
            if ptype == "custom_api" and not (base_url or endpoints) and not effective_base_url(db, pid):
                continue  # invalid empty endpoint -> excluded (invariant 6)
            if ptype == "remote_ollama" and not endpoints:
                continue  # remote ollama needs an endpoint
            for cred in iter_credentials(pdata):
                if cred.get("enabled") is False or cred.get("quarantined"):
                    continue
                secret = cred.get("secret")
                if not secret and ptype not in ("local_ollama",):
                    continue  # missing credential where required -> excluded
                qd = effective_quota_domain(pid, cred)
                eps = endpoints or ([base_url] if base_url else [None])
                for ep in eps:
                    key = (pool, pid, cred.get("id"), str(ep), model_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    rpm, tpm = resolve_deployment_limits(db, pid, cred, model_id)
                    pending.append({"logical_model": pool, "provider": pid,
                                    "credential_id": cred.get("id"), "secret": secret,
                                    "quota_domain": qd, "endpoint": ep,
                                    "upstream_model": model_id, "rpm": rpm, "tpm": tpm,
                                    "capabilities": caps,
                                    "health": deployment_health(db, pid, model_id),
                                    "enabled": True})
                    domain_counts[(qd, model_id)] = domain_counts.get((qd, model_id), 0) + 1
    # apply shared-quota split so N deployments on one domain never claim N x quota
    trust_rank = {t: i for i, t in enumerate(TRUST_ORDER)}
    def _trust_of(d):
        meta = PROVIDER_META.get(d["provider"]) or PROVIDER_META.get("custom")
        return trust_rank.get((meta or {}).get("trust_tier", "unknown"), 99)
    for d in pending:
        n = domain_counts.get((d["quota_domain"], d["upstream_model"]), 1)
        if n > 1 and isinstance(d.get("rpm"), (int, float)):
            d["rpm"] = split_shared_quota(d["rpm"], n, split_mode)
    pending.sort(key=lambda d: (d["logical_model"], _trust_of(d), d["provider"],
                                d["quota_domain"], d["credential_id"] or ""))
    return list(pending)


def build_model_pools(deployments):
    pools = {}
    for d in deployments:
        pools.setdefault(d["logical_model"], []).append(d)
    return pools


def build_roles(db, pools):
    """Compile role aliases -> {role: {pools, fallbacks, model}}. Roles point
    at logical pools; strict capability roles exclude unknown deps."""
    roles = {}
    for role, spec in (db.get(ROLES_KEY) or {}).items():
        if not isinstance(spec, dict):
            continue
        wanted = list(spec.get("pools") or [])
        valid = [p for p in wanted if p in pools]
        if not valid:
            continue  # no role pointing to missing pools (invariant 4)
        req = spec.get("requires") or {}
        if any(req.get(k) for k in ("tools", "vision", "reasoning")):
            kept = []
            for p in valid:
                deps = pools[p]
                ok = True
                for dep in deps:
                    caps = dep.get("capabilities") or {}
                    for k in ("tools", "vision", "reasoning"):
                        if req.get(k) and caps.get(k) is not True:
                            ok = False
                            break
                if ok:
                    kept.append(p)
            valid = kept
            if not valid:
                continue
        fallbacks = [p for p in (spec.get("fallback") or []) if p in pools and p not in valid]
        roles[role] = {"pools": valid, "fallback": fallbacks}
    return roles


def validate_compiled_config(deployments, pools, roles):
    errors = []
    seen = set()
    for d in deployments:
        key = (d["logical_model"], d["provider"], d.get("credential_id"),
               str(d.get("endpoint")), d["upstream_model"])
        if key in seen:
            errors.append(f"duplicate deployment: {key}")
        seen.add(key)
        if not d.get("logical_model") or not d.get("upstream_model"):
            errors.append(f"deployment with empty model name: {key}")
        if d.get("endpoint") == "":
            errors.append(f"deployment with empty endpoint: {key}")
    for pool, deps in pools.items():
        if not deps:
            errors.append(f"alias '{pool}' has zero valid deployments")
    for role, spec in roles.items():
        for p in spec.get("pools", []):
            if p not in pools:
                errors.append(f"role '{role}' points to missing pool '{p}'")
    return errors


def compile_config(db):
    """Full pipeline: DB (vault + proxy) -> deployments -> pools -> roles -> proxy dict."""
    db = migrate_db(db)
    deployments = build_deployments(db)
    pools = build_model_pools(deployments)
    roles = build_roles(db, pools)
    errors = validate_compiled_config(deployments, pools, roles)
    return deployments, pools, roles, errors

def get_proxy_type(db: dict[str, Any]) -> str:
    """Proxy type from vault DB, not hardcoded. tui must use this."""
    p = db.get(PROXY_KEY) if isinstance(db.get(PROXY_KEY), dict) else {}
    t = p.get("type") if isinstance(p, dict) else None
    return t if t in PROXY_TYPES else DEFAULT_PROXY_TYPE

def set_proxy_type(db: dict[str, Any], proxy_type: str) -> None:
    if proxy_type not in PROXY_TYPES:
        raise ValueError(f"unknown proxy: {proxy_type}")
    if not isinstance(db.get(PROXY_KEY), dict):
        db[PROXY_KEY] = {}
    db[PROXY_KEY]["type"] = proxy_type


def deployment_diff(old_deps, new_deps):
    """Secret-free diff of deployment graphs."""
    def key(d):
        return (d["logical_model"], d["provider"], d.get("credential_id"),
                str(d.get("endpoint")), d["upstream_model"])
    old, new = {key(d) for d in old_deps}, {key(d) for d in new_deps}
    old_pools = {d["logical_model"] for d in old_deps}
    new_pools = {d["logical_model"] for d in new_deps}
    return {"added": len(new - old), "removed": len(old - new),
            "pools_added": sorted(new_pools - old_pools),
            "pools_removed": sorted(old_pools - new_pools)}


def generate_yaml(db_data, _prev_deployments=None):
    """Compile DB -> LiteLLM YAML. Stages:

    DB -> migrate/normalize -> credentials -> quota domains -> model
    metadata -> deployments -> logical pools -> role aliases ->
    routing/fallback policy -> LiteLLM YAML.

    Guarantees: no duplicate deployments, no stale members, no empty
    aliases, no roles pointing at missing pools, no missing credentials,
    no empty endpoints, no alias collisions, only router options verified
    against the installed LiteLLM, secrets never logged.
    Raises ValueError (leaving config.yaml untouched) on violations.
    Returns the number of emitted routes.
    """
    deployments, pools, roles, errors = compile_config(db_data)
    if errors:
        raise ValueError("refusing to write broken config: " + "; ".join(errors[:8]))
    if _prev_deployments is not None:
        diff = deployment_diff(_prev_deployments, deployments)
        parts = [f"+{diff['added']} deployments", f"-{diff['removed']} deployments"]
        if diff["pools_added"]:
            parts.append("+" + str(len(diff["pools_added"])) + " pools: " + ", ".join(diff["pools_added"][:5]))
        if diff["pools_removed"]:
            parts.append("-" + str(len(diff["pools_removed"])) + " pools: " + ", ".join(diff["pools_removed"][:5]))
        print("  [plan] " + " | ".join(parts))
    model_list = []
    for d in deployments:
        pid = d["provider"]
        p_info = _provider_info(db_data, pid)
        if not p_info:
            continue
        ptype = p_info.get("type")
        pool = d["logical_model"]
        upstream = d["upstream_model"]
        litellm_model = _litellm_model_for_provider(pid, upstream, ptype)
        params = {"model": litellm_model}
        if ptype == "local_ollama":
            params["api_base"] = d.get("endpoint")
        elif ptype == "remote_ollama":
            params["api_base"] = d.get("endpoint")
            if d.get("secret"):
                params["api_key"] = d["secret"]
        elif ptype == "custom_api":
            base_url = effective_base_url(db_data, pid)
            if not base_url:
                continue
            params["api_base"] = base_url
            if d.get("secret"):
                params["api_key"] = d["secret"]
        elif ptype == "api":
            if d.get("secret"):
                params["api_key"] = d["secret"]
            else:
                continue
        else:
            continue
        if isinstance(d.get("rpm"), (int, float)) and d["rpm"] > 0:
            params["rpm"] = int(d["rpm"])
        if isinstance(d.get("tpm"), (int, float)) and d["tpm"] > 0:
            params["tpm"] = int(d["tpm"])
        # per-deployment cooldown (verified: LiteLLM reads cooldown_time
        # from litellm_params/model_info; 429-prone providers pause longer)
        params["cooldown_time"] = PROVIDER_COOLDOWN_S.get(pid, COOLDOWN_TRANSIENT_S)
        if _needs_drop_params(pid, upstream):
            params["drop_params"] = True
        model_list.append({"model_name": pool, "litellm_params": params})
    # Role aliases: model_group_alias (role -> primary pool) + ordered
    # fallbacks (role -> remaining pools + explicit fallbacks). Verified
    # against installed LiteLLM Router (model_group_alias + fallbacks).
    model_group_alias = {}
    fallbacks = []
    for role in sorted(roles):
        spec = roles[role]
        primaries = list(spec.get("pools") or [])
        extra = list(spec.get("fallback") or [])
        if not primaries:
            continue
        if role in pools:
            print(f"  [!] role '{role}' collides with model pool name — role skipped (rename one).")
            continue
        model_group_alias[role] = primaries[0]
        rest = primaries[1:] + [p for p in extra if p not in primaries]
        if rest:
            fallbacks.append({role: rest})
    config = {
        "general_settings": {
            # env-backed: LiteLLM resolves os.environ/* at startup, so the
            # secret lives in the environment/0600 secret file, not in YAML.
            "master_key": "os.environ/LITELLM_MASTER_KEY",
        },
        "model_list": model_list,
        "router_settings": {
            "routing_strategy": ROUTING_STRATEGY,
            "num_retries": ROUTER_NUM_RETRIES,
            "cooldown_time": ROUTER_COOLDOWN_TIME,
            "allowed_fails": ROUTER_ALLOWED_FAILS,
            "allowed_fails_policy": dict(ALLOWED_FAILS_POLICY),
            "enable_pre_call_checks": True,
            "retry_policy": dict(RETRY_POLICY),
        },
    }
    if model_group_alias:
        config["model_group_alias"] = model_group_alias
    if fallbacks:
        config["fallbacks"] = fallbacks
    text = yaml.dump(config, default_flow_style=False, sort_keys=False)
    # validate before replacing: must parse and carry no raw secrets in keys
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("model_list"), list):
        raise ValueError("compiled YAML failed self-validation")
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(YAML_FILE)) or ".",
                               prefix=".config-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.chmod(tmp, 0o600)
        os.replace(tmp, YAML_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return len(model_list)


def safe_generate_yaml(db):
    """generate_yaml() that reports instead of raising in interactive flows."""
    try:
        return generate_yaml(db)
    except ValueError as e:
        print(f"  [!] Config not written: {e}")
        return None


# ---------- direct provider validation (no proxy needed) ----------

def _valid_url(url):
    """Coerce a user-supplied base URL into a usable form, or return None.

    Auto-prefix https:// when a scheme is missing and never let invalid URLs
    escape into urllib (which raises ValueError("unknown url type") instead
    of returning a clean error).
    """
    url = (url or "").strip()
    if not url:
        return None
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        host = url.split("/")[0]
        if host.startswith("localhost") or host.startswith("127.") or host == "[::1]":
            url = "http://" + url
        else:
            url = "https://" + url
    try:
        urllib.parse.urlparse(url)
    except Exception:
        return None
    return url


def _get(url, headers=None, timeout=TIMEOUT):
    """GET JSON. Returns (status:int|None, data:dict|None, raw:str)."""
    url = _valid_url(url)
    if not url:
        return None, None, "invalid URL"
    h = dict(UA)
    h.update(headers or {})
    try:
        req = urllib.request.Request(url, headers=h, method="GET")
    except Exception as e:
        return None, None, str(e)[:300]
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode(errors="replace")
            try:
                return r.status, json.loads(raw), raw[:500]
            except Exception:
                return r.status, None, raw[:500]
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode(errors="replace")
        except Exception:
            raw = str(e)
        try:
            return e.code, json.loads(raw), raw[:500]
        except Exception:
            return e.code, None, raw[:500]
    except Exception as e:
        return None, None, str(e)[:300]


def _post(url, payload, headers=None, timeout=30):
    """POST JSON. Returns (status:int|None, data:dict|None, raw:str)."""
    url = _valid_url(url)
    if not url:
        return None, None, "invalid URL"
    h = dict(UA)
    h.update(headers or {})
    h["Content-Type"] = "application/json"
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers=h, method="POST")
    except Exception as e:
        return None, None, str(e)[:300]
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode(errors="replace")
            try:
                return r.status, json.loads(raw), raw[:800]
            except Exception:
                return r.status, None, raw[:800]
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode(errors="replace")
        except Exception:
            raw = str(e)
        low = raw.lower()
        tag = "RATELIMIT" if (e.code == 429 or '"code":429' in raw or '"code": 429' in low) else ""
        try:
            return e.code, json.loads(raw), (tag + " " + raw[:600]).strip()
        except Exception:
            return e.code, None, (tag + " " + raw[:600]).strip()
    except Exception as e:
        return None, None, str(e)[:300]


def _gemini_models(key):
    s, d, raw = _get(f"https://generativelanguage.googleapis.com/v1beta/models?key={urllib.parse.quote(key)}")
    if s == 200 and d and "models" in d:
        ids = [m.get("name", "").replace("models/", "") for m in d["models"]]
        return True, f"OK ({len(ids)} models visible)", ids
    if s == 400 and "API key not valid" in raw:
        return False, "invalid key (400 API key not valid)", None
    if s in (401, 403):
        return False, f"rejected (HTTP {s}): {raw[:120]}", None
    return False, f"HTTP {s}: {raw[:150]}", None


def _openrouter_key(key):
    s, d, raw = _get("https://openrouter.ai/api/v1/auth/key",
                     {"Authorization": f"Bearer {key}"})
    if s == 200:
        return True, "OK (key valid)", None
    return False, f"HTTP {s}: {raw[:150]}", None


def _openrouter_models(key):
    s, d, raw = _get("https://openrouter.ai/api/v1/models",
                     {"Authorization": f"Bearer {key}"})
    if s == 200 and d and "data" in d:
        return [m.get("id") for m in d["data"] if m.get("id")]
    return None


def _anthropic_key(key):
    s, d, raw = _get("https://api.anthropic.com/v1/models",
                     {"x-api-key": key, "anthropic-version": "2023-06-01"})
    if s == 200 and d and "data" in d:
        return True, f"OK ({len(d['data'])} models)", [m.get("id") for m in d["data"]]
    if s in (401, 403):
        return False, f"invalid key (HTTP {s})", None
    return False, f"HTTP {s}: {raw[:150]}", None


def _openai_key(key):
    s, d, raw = _get("https://api.openai.com/v1/models",
                     {"Authorization": f"Bearer {key}"})
    if s == 200 and d and "data" in d:
        return True, f"OK ({len(d['data'])} models)", [m.get("id") for m in d["data"]]
    if s == 401:
        return False, "invalid key (401 incorrect API key)", None
    return False, f"HTTP {s}: {raw[:150]}", None


def _zen_key(key):
    s, d, raw = _get("https://opencode.ai/zen/v1/models",
                     {"Authorization": f"Bearer {key}"})
    if s == 200 and d and "data" in d:
        return True, f"OK ({len(d['data'])} models)", [m.get("id") for m in d["data"]]
    if s in (401, 403):
        return False, f"rejected (HTTP {s}): {raw[:150]}", None
    return False, f"HTTP {s}: {raw[:150]}", None


TOKENROUTER_BASES = ["https://api.tokenrouter.com/v1", "https://api.tokenrouter.io/v1"]


def _tokenrouter_key(key):
    # Two live endpoints: tokenrouter.com (sk-... keys, official console) and
    # tokenrouter.io v2 (tr_... keys). Try .com first, fall back to .io.
    # Returns (ok, msg, ids|None, base|None).
    last = None
    for base in TOKENROUTER_BASES:
        s, d, raw = _get(f"{base}/models", {"Authorization": f"Bearer {key}"})
        if s == 200 and d:
            ids = [m.get("id") for m in d.get("data", [])] if isinstance(d.get("data"), list) else None
            return True, f"OK @ {base} ({len(ids) if ids else '?'} models)", ids, base
        last = (s, raw, base)
    s, raw, base = last
    if s == 401:
        return False, "invalid key (401 @ both .com and .io) — check key at tokenrouter.com/console/token", None, None
    return False, f"HTTP {s}: {raw[:150]}", None, None


def _zai_key(key, base=None):
    base = (base or _BUILTIN_BASES["zai"]).rstrip("/")
    s, d, raw = _get(f"{base}/models",
                     {"Authorization": f"Bearer {key}"})
    if s == 200 and d and "data" in d:
        return True, f"OK ({len(d['data'])} models)", [m.get("id") for m in d["data"]]
    if s in (401, 403):
        return False, f"invalid key (HTTP {s})", None
    return False, f"HTTP {s}: {raw[:150]}", None


def _ollama_cloud_ep_key(endpoint, key):
    base = endpoint.rstrip("/")
    s, d, raw = _get(f"{base}/api/tags", {"Authorization": f"Bearer {key}"})
    if s == 200 and d and "models" in d:
        return True, f"OK ({len(d['models'])} models)", [m.get("name") for m in d["models"]]
    if s in (401, 403):
        return False, f"invalid key (HTTP {s})", None
    return False, f"HTTP {s}: {raw[:150]}", None


def _ollama_local_ep(endpoint):
    base = endpoint.rstrip("/")
    s, d, raw = _get(f"{base}/api/tags")
    if s == 200 and d and "models" in d:
        names = [m.get("name") for m in d["models"]]
        return True, f"OK reachable ({len(names)} local models)", names
    if s is None:
        return False, f"unreachable: {raw[:150]}", None
    return False, f"HTTP {s}: {raw[:150]}", None


def _oai_compat_models(base, key):
    """Generic OpenAI-compatible GET {base}/models. Returns (ok, msg, [(id,label)]|None)."""
    s, d, raw = _get(f"{base.rstrip('/')}/models", {"Authorization": f"Bearer {key}"})
    if s == 200 and d and "data" in d:
        items = sorted([(m["id"], m.get("name") or m["id"]) for m in d["data"] if m.get("id")])
        return True, f"OK @ {base.rstrip('/')} ({len(items)} models)", items
    if s in (401, 403):
        return False, f"invalid key (HTTP {s})", None
    return False, f"HTTP {s}: {raw[:150]}", None


def validate_keys(pid, keys, endpoints):
    """Test every key directly. Returns (results, available_models|None).

    results = [(key, ok, msg), ...]
    """
    results = []
    avail = None
    print(f"  [*] Testing {len(keys)} key(s) directly against provider...")
    if pid == "gemini":
        for k in keys:
            ok, msg, ids = _gemini_models(k)
            if ids and avail is None:
                avail = ids
            print(f"      [{'OK' if ok else 'FAIL'}] {snippet(k)} -> {msg}")
            results.append((k, ok, msg))
    elif pid == "openrouter":
        # one models fetch for hint, per-key auth check
        if keys:
            avail = _openrouter_models(keys[0])
        for k in keys:
            ok, msg, _ = _openrouter_key(k)
            print(f"      [{'OK' if ok else 'FAIL'}] {snippet(k)} -> {msg}")
            results.append((k, ok, msg))
    elif pid == "anthropic":
        for k in keys:
            ok, msg, ids = _anthropic_key(k)
            if ids and avail is None:
                avail = ids
            print(f"      [{'OK' if ok else 'FAIL'}] {snippet(k)} -> {msg}")
            results.append((k, ok, msg))
    elif pid == "openai":
        for k in keys:
            ok, msg, ids = _openai_key(k)
            if ids and avail is None:
                avail = ids
            print(f"      [{'OK' if ok else 'FAIL'}] {snippet(k)} -> {msg}")
            results.append((k, ok, msg))
    elif pid == "opencode_zen":
        for k in keys:
            ok, msg, ids = _zen_key(k)
            if ids and avail is None:
                avail = ids
            print(f"      [{'OK' if ok else 'FAIL'}] {snippet(k)} -> {msg}")
            results.append((k, ok, msg))
    elif pid == "tokenrouter":
        for k in keys:
            ok, msg, ids, base = _tokenrouter_key(k)
            if ids and avail is None:
                avail = ids
            print(f"      [{'OK' if ok else 'FAIL'}] {snippet(k)} -> {msg}")
            results.append((k, ok, msg))
    elif pid == "zai":
        base = effective_base_url(None, pid, (endpoints or [None])[0])
        for k in keys:
            ok, msg, ids = _zai_key(k, base)
            if ids and avail is None:
                avail = ids
            print(f"      [{'OK' if ok else 'FAIL'}] {snippet(k)} -> {msg}")
            results.append((k, ok, msg))
    elif pid == "ollama_cloud":
        ep = (endpoints or ["https://ollama.com"])[0]
        for k in keys:
            ok, msg, ids = _ollama_cloud_ep_key(ep, k)
            if ids and avail is None:
                avail = ids
            print(f"      [{'OK' if ok else 'FAIL'}] {snippet(k)} -> {msg}")
            results.append((k, ok, msg))
    elif pid == "ollama_local":
        pass  # endpoint validated separately
    elif pid == "custom" or pid.startswith("custom_"):
        base = (endpoints or [None])[0]
        if not base:
            print("      [FAIL] no base URL stored for custom provider")
            return [(k, False, "no base URL") for k in keys], None
        for k in keys:
            ok, msg, items = _oai_compat_models(base, k)
            if items and avail is None:
                avail = [mid for mid, _ in items]
            print(f"      [{'OK' if ok else 'FAIL'}] {snippet(k)} -> {msg}")
            results.append((k, ok, msg))
    return results, avail


# ---------- online model catalog (no more guessing IDs) ----------

def fetch_catalog(pid, key, endpoint=None):
    """Fetch live model list -> [(id, label), ...] or None on failure."""
    try:
        if pid == "gemini":
            s, d, _ = _get(f"https://generativelanguage.googleapis.com/v1beta/models?key={urllib.parse.quote(key)}")
            if s != 200 or not d or "models" not in d:
                return None
            out = []
            for m in d["models"]:
                if "generateContent" not in (m.get("supportedGenerationMethods") or []):
                    continue
                mid = m.get("name", "").replace("models/", "")
                if mid:
                    out.append((mid, m.get("displayName") or mid))
            return sorted(out)
        if pid == "openrouter":
            s, d, _ = _get("https://openrouter.ai/api/v1/models",
                           {"Authorization": f"Bearer {key}"})
            if s != 200 or not d or "data" not in d:
                # public fallback (no auth)
                s, d, _ = _get("https://openrouter.ai/api/v1/models")
                if s != 200 or not d or "data" not in d:
                    return None
            return sorted([(m["id"], m.get("name") or m["id"]) for m in d["data"] if m.get("id")])
        if pid == "anthropic":
            s, d, _ = _get("https://api.anthropic.com/v1/models",
                           {"x-api-key": key, "anthropic-version": "2023-06-01"})
            if s != 200 or not d or "data" not in d:
                return None
            return sorted([(m["id"], m.get("display_name") or m["id"]) for m in d["data"] if m.get("id")])
        if pid == "openai":
            s, d, _ = _get("https://api.openai.com/v1/models",
                           {"Authorization": f"Bearer {key}"})
            if s != 200 or not d or "data" not in d:
                return None
            return sorted([(m["id"], m["id"]) for m in d["data"] if m.get("id")])
        if pid == "opencode_zen":
            base = (endpoint or _BUILTIN_BASES["opencode_zen"]).rstrip("/")
            s, d, _ = _get(f"{base}/models",
                           {"Authorization": f"Bearer {key}"})
            if s != 200 or not d or "data" not in d:
                return None
            return sorted([(m["id"], m["id"]) for m in d["data"] if m.get("id")])
        if pid == "tokenrouter":
            bases = [endpoint] if endpoint else TOKENROUTER_BASES
            for base in bases:
                s, d, _ = _get(f"{base.rstrip('/')}/models",
                               {"Authorization": f"Bearer {key}"})
                if s == 200 and d and isinstance(d.get("data"), list):
                    return sorted([(m["id"], m.get("name") or m["id"]) for m in d["data"] if m.get("id")])
            return None
        if pid == "custom" or pid.startswith("custom_"):
            if not endpoint:
                return None
            ok, _, items = _oai_compat_models(endpoint, key)
            return items if ok else None
        if pid == "zai":
            base = (endpoint or _BUILTIN_BASES["zai"]).rstrip("/")
            s, d, _ = _get(f"{base}/models",
                           {"Authorization": f"Bearer {key}"})
            if s != 200 or not d or "data" not in d:
                return None
            return sorted([(m["id"], m["id"]) for m in d["data"] if m.get("id")])
        if pid == "ollama_cloud":
            base = (endpoint or "https://ollama.com").rstrip("/")
            s, d, _ = _get(f"{base}/api/tags", {"Authorization": f"Bearer {key}"})
            if s != 200 or not d or "models" not in d:
                return None
            return sorted([(m["name"], m["name"]) for m in d["models"] if m.get("name")])
        if pid == "ollama_local":
            base = (endpoint or "http://localhost:11434").rstrip("/")
            s, d, _ = _get(f"{base}/api/tags")
            if s != 200 or not d or "models" not in d:
                return None
            return sorted([(m["name"], m["name"]) for m in d["models"] if m.get("name")])
    except Exception:
        return None
    return None


def _norm(s):
    return s.strip().lower().replace("_", "-").replace(" ", "-").replace("--", "-")


def _is_free_model(mid, label=""):
    """Heuristic: free-tier models (OpenRouter :free, Zen free IDs, etc.)."""
    m = (mid or "").lower()
    return "free" in m or m == "big-pickle" or "free" in (label or "").lower()


def pick_models(pid, pname, existing, catalog):
    """Interactive picker over live catalog. Returns list of model IDs (bare)."""
    print(f"\nModels for {pname} — live catalog ({len(catalog)} available, no guessing needed).")
    if existing:
        print(f"Already saved: {' '.join(existing)}")
    print("Pick with commas or spaces: numbers (1,3), ranges (1-5 = 1 through 5),")
    print("  /filter text, exact IDs, display names — or ALL. Numbers always refer to")
    print("  the list currently shown. DONE finishes, CLEAR resets your picks.")
    # alias map: normalized id + normalized label -> id
    alias = {}
    for mid, label in catalog:
        alias[_norm(mid)] = mid
        alias[_norm(label)] = mid
        alias[_norm(mid.split("/")[-1])] = mid
    chosen = []
    free_only = [item for item in catalog if _is_free_model(item[0], item[1])]
    if free_only and len(free_only) < len(catalog):
        view = free_only
        expanded = False
        print(f"  [FREE] showing {len(free_only)} free models first — type MORE for all {len(catalog)}")
    else:
        view = list(catalog)
        expanded = True
    page, per = 0, 30
    while True:
        total_pages = max(1, (len(view) + per - 1) // per)
        page = max(0, min(page, total_pages - 1))
        chunk = view[page * per:(page + 1) * per]
        scope = "free" if not expanded else "all"
        print(f"\n--- catalog ({scope}) page {page + 1}/{total_pages} ({len(view)} shown) ---")
        for i, (mid, label) in enumerate(chunk, start=page * per + 1):
            mark = "*" if mid in chosen or mid in (existing or []) else " "
            extra = f"  [{label}]" if label != mid else ""
            print(f"  {mark}[{i:3d}] {mid}{extra[:80]}")
        if len(view) > per:
            print("  [N]ext [P]rev")
        if not expanded:
            print("  [MORE] show all models")
        try:
            raw = input(f"Select (chosen={len(chosen)}, DONE to finish): ").strip()
        except EOFError:
            break
        if not raw:
            continue
        low = raw.lower()
        if low in ("done", "d", "q", "finish"):
            break
        if low == "clear":
            chosen = []
            continue
        if low in ("more", "paid", "rest", "show all", "showall"):
            view = list(catalog)
            expanded = True
            page = 0
            continue
        if low in ("n", "next") and len(view) > per:
            page += 1
            continue
        if low in ("p", "prev") and len(view) > per:
            page -= 1
            continue
        if low == "all":
            for mid, _ in view:
                if mid not in chosen:
                    chosen.append(mid)
            print(f"  [+] selected all {len(view)} in view")
            continue
        if raw.startswith("/"):
            q = _norm(raw[1:])
            view = [(m, l) for m, l in catalog if q in _norm(m) or q in _norm(l)]
            page = 0
            if not view:
                print("  [!] filter matched nothing — showing full catalog")
                view = list(catalog)
            continue
        # token stream: numbers, ranges, IDs, or multi-word display names
        # ("1 Nemotron 3.5 Lightning Free" -> #1 + display-name match)
        ok_any = False
        words = raw.replace(",", " ").split()
        nums, rest = [], []
        for tok in words:
            if tok.isdigit() or ("-" in tok and tok.replace("-", "").isdigit()):
                nums.append(tok)
            else:
                rest.append(tok)
        for tok in nums:
            if "-" in tok:
                try:
                    a, b = tok.split("-", 1)
                    for n in range(int(a), int(b) + 1):
                        if 1 <= n <= len(view):
                            mid = view[n - 1][0]
                            if mid not in chosen:
                                chosen.append(mid)
                            ok_any = True
                        else:
                            print(f"  [!] #{n} out of range")
                except Exception:
                    pass
            else:
                n = int(tok)
                if 1 <= n <= len(view):
                    mid = view[n - 1][0]
                    if mid not in chosen:
                        chosen.append(mid)
                    ok_any = True
                else:
                    print(f"  [!] #{tok} out of range")
        # greedy longest-phrase match for display names ("Nemotron 3.5 Lightning Free")
        i = 0
        while i < len(rest):
            matched = None
            for j in range(min(len(rest), i + 6), i, -1):
                phrase = " ".join(rest[i:j])
                t = phrase.split("/")[-1] if phrase.startswith("opencode/") else phrase
                mid = alias.get(_norm(t))
                if mid:
                    matched = (mid, j)
                    break
            if matched:
                mid, j = matched
                if mid not in chosen:
                    chosen.append(mid)
                ok_any = True
                i = j
            else:
                # strip opencode/ prefix automatically
                t = rest[i].split("/")[-1] if rest[i].startswith("opencode/") else rest[i]
                mid = alias.get(_norm(t))
                if mid:
                    if mid not in chosen:
                        chosen.append(mid)
                    ok_any = True
                else:
                    # unknown manual ID — allow (new/preview), warn
                    if t not in chosen:
                        chosen.append(t)
                    print(f"  [?] '{rest[i]}' not in catalog — kept as manual ID (check spelling)")
                    ok_any = True
                i += 1
        if ok_any:
            print(f"  [+] chosen now ({len(chosen)}): {' '.join(chosen[-8:])}" + (" ..." if len(chosen) > 8 else ""))
    return chosen


# ---------- per-model direct test (not just keys) ----------

def test_single_model(pid, model, key, endpoint=None):
    """Direct minimal completion. Returns (status, msg): OK | RATELIMIT | FAIL."""
    try:
        if pid == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model, safe='')}:generateContent?key={urllib.parse.quote(key)}"
            s, d, raw = _post(url, {"contents": [{"parts": [{"text": "hi"}]}],
                                    "generationConfig": {"maxOutputTokens": 1}})
        elif pid == "openrouter":
            s, d, raw = _post("https://openrouter.ai/api/v1/chat/completions",
                              {"model": model, "messages": [{"role": "user", "content": "hi"}],
                               "max_tokens": 1},
                              {"Authorization": f"Bearer {key}",
                               "HTTP-Referer": "http://localhost", "X-Title": "litellm-wizard"})
        elif pid == "anthropic":
            s, d, raw = _post("https://api.anthropic.com/v1/messages",
                              {"model": model, "max_tokens": 1,
                               "messages": [{"role": "user", "content": "hi"}]},
                              {"x-api-key": key, "anthropic-version": "2023-06-01"})
        elif pid == "openai":
            s, d, raw = _post("https://api.openai.com/v1/chat/completions",
                              {"model": model, "messages": [{"role": "user", "content": "hi"}],
                               "max_tokens": 1},
                              {"Authorization": f"Bearer {key}"})
        elif pid in ("opencode_zen", "tokenrouter", "zai") or pid == "custom" or pid.startswith("custom_"):
            if pid == "tokenrouter":
                bases = [endpoint] if endpoint else TOKENROUTER_BASES
                mid = model  # .com needs full prefixed ID (z-ai/...)
                to = 90  # cold starts are slow
            elif pid == "custom" or pid.startswith("custom_"):
                if not endpoint:
                    return "FAIL", "no base URL stored for custom provider"
                bases = [endpoint]
                mid = model  # keep full ID from catalog
                to = 60
            else:
                bases = [endpoint or _BUILTIN_BASES[pid]]
                mid = model.split("/")[-1]  # bare ID
                to = 30
            bare = mid
            s, d, raw = None, None, ""
            for base in bases:
                s, d, raw = _post(f"{base.rstrip('/')}/chat/completions",
                                  {"model": mid, "messages": [{"role": "user", "content": "hi"}],
                                   "max_tokens": 1},
                                  {"Authorization": f"Bearer {key}"}, timeout=to)
                if s != 401:
                    break
        elif pid in ("ollama_cloud", "ollama_local"):
            base = (endpoint or ("https://ollama.com" if pid == "ollama_cloud" else "http://localhost:11434")).rstrip("/")
            hdr = {"Authorization": f"Bearer {key}"} if pid == "ollama_cloud" else {}
            s, d, raw = _post(f"{base}/api/chat",
                              {"model": model, "messages": [{"role": "user", "content": "hi"}],
                               "stream": False, "options": {"num_predict": 1}}, hdr, timeout=60)
        else:
            return "FAIL", "unknown provider"
    except Exception as e:
        return "FAIL", str(e)[:150]
    if s == 200:
        return "OK", "completion OK"
    if s == 429 or (isinstance(raw, str) and raw.startswith("RATELIMIT")):
        return "RATELIMIT", "valid but throttled (429) — kept"
    low = (raw or "")[:200].replace("\n", " ")
    hint = ""
    if "not supported" in low and pid == "opencode_zen":
        hint = " (Responses-only or OpenCode-client-only free tier)"
    if "MissingSessionID" in (raw or ""):
        hint = " (free tier works only inside OpenCode client)"
    if s in (401, 403):
        return "FAIL", f"HTTP {s}: {low[:140]}{hint}"
    if s == 404:
        return "FAIL", f"model not found (404): {low[:140]}"
    return "FAIL", f"HTTP {s}: {low[:150]}{hint}"


def probe_model_classified(pid, model, key, endpoint=None):
    """Probe one model+credential, returning structured (class, msg).

    Classes: OK | RATE_LIMITED | AUTH_ERROR | BAD_REQUEST | SERVER_ERROR |
    TIMEOUT | UNAVAILABLE | UNKNOWN. Distinguishes credential validity
    from model access from health from quota saturation.
    """
    st, msg = test_single_model(pid, model, key, endpoint)
    if st == "OK":
        return "OK", msg
    if st == "RATELIMIT":
        return "RATE_LIMITED", msg
    m = (msg or "")
    if "HTTP 401" in m or "HTTP 403" in m:
        return "AUTH_ERROR", msg
    if "HTTP 404" in m or "HTTP 400" in m:
        return "BAD_REQUEST", msg
    if "HTTP 500" in m or "HTTP 502" in m or "HTTP 503" in m:
        return "SERVER_ERROR", msg
    low = m.lower()
    if "timeout" in low or "timed out" in low:
        return "TIMEOUT", msg
    if "unreachable" in low or "connection" in low or "no base url" in low:
        return "UNAVAILABLE", msg
    return "UNKNOWN", msg


def plan_probe_matrix(pid, models, keys, endpoints, mode="FAST", sample_size=2, db=None):
    """Return list of (model, key, endpoint) probes for a validation mode.

    FAST: each model x one credential. STRICT: each model x every
    credential. SAMPLE: each model x N credentials (one per quota domain
    first). Never surprises the user — callers print len() first.
    """
    keys = list(keys or [])
    models = list(models or [])
    if not keys or pid == "ollama_local":
        ep = (endpoints or [None])[0] if endpoints else None
        return [(m, None, ep) for m in models]
    ep = (endpoints or [None])[0] if endpoints else None
    if mode == "STRICT":
        return [(m, k, ep) for m in models for k in keys]
    if mode == "SAMPLE":
        buckets, seen_qd, rest = [], set(), []
        qd_of = {}
        if db is not None:
            pdata = db.get(pid, {}) if isinstance(db.get(pid), dict) else {}
            for c in pdata.get("credentials", []) or []:
                if isinstance(c, dict) and c.get("secret"):
                    qd_of[str(c["secret"])] = c.get("quota_domain", "")
        for k in keys:
            qd = qd_of.get(str(k), "")
            if qd and qd not in seen_qd and len(buckets) < max(1, sample_size):
                seen_qd.add(qd)
                buckets.append(k)
            else:
                rest.append(k)
        chosen = buckets + [k for k in rest if k not in buckets][:max(0, sample_size - len(buckets))]
        if not chosen:
            chosen = keys[:1]
        return [(m, k, ep) for m in models for k in chosen]
    return [(m, keys[0], ep) for m in models]  # FAST


def test_models(pid, models, key, endpoint=None, keys=None, mode="FAST",
                sample_size=2, db=None, sleep_s=1.5):
    """Probe models with structured classification.

    Default (FAST) preserves the old behavior: one credential per model.
    Returns list of (model, class, msg).
    """
    matrix = [(m, key, endpoint) for m in models]
    if keys is not None and mode in ("STRICT", "SAMPLE") and len(keys or []) > 1:
        matrix = plan_probe_matrix(pid, models, keys, [endpoint] if endpoint else [],
                                   mode, sample_size, db)
    tag = {"FAST": "first key", "STRICT": "every key", "SAMPLE": f"sample({sample_size})"}.get(mode, "first key")
    first = snippet(matrix[0][1]) if matrix and matrix[0][1] else "no-key"
    print(f"  [*] Testing {len(models)} model(s) directly (minimal ping, {tag} {first}, {len(matrix)} probes)...")
    results = []
    for m, k, ep in matrix:
        t0 = time.monotonic()
        cls, msg = probe_model_classified(pid, m, k, ep)
        took = time.monotonic() - t0
        label = {"OK": "OK", "RATE_LIMITED": "WAIT"}.get(cls, "FAIL")
        extra = f" [{snippet(k)}]" if mode != "FAST" and k else ""
        lat = f" ({took:.1f}s)" if cls in ("OK", "RATE_LIMITED") else ""
        print(f"      [{label}] {m}{extra}{lat} -> {msg}")
        results.append((m, cls, msg))
        if cls in ("OK", "RATE_LIMITED") and db is not None:
            record_probe_latency(db, pid, m, took)
        if db is not None and k:
            pdata = db.get(pid, {}) if isinstance(db.get(pid), dict) else {}
            cid = next((c.get("id") for c in (pdata.get("credentials") or [])
                        if isinstance(c, dict) and str(c.get("secret")) == str(k)), None)
            if cid:
                vstat = {"OK": "ok", "RATE_LIMITED": "throttled",
                         "AUTH_ERROR": "invalid"}.get(cls, "unknown")
                record_validation(db, pid, cid, vstat, msg)
        time.sleep(sleep_s)
    # collapse per-model: OK wins, then RATE_LIMITED, else first hard failure
    if mode != "FAST":
        by_model = {}
        for m, cls, msg in results:
            by_model.setdefault(m, []).append((cls, msg))
        order = {"OK": 0, "RATE_LIMITED": 1, "TIMEOUT": 2, "SERVER_ERROR": 2,
                 "UNAVAILABLE": 2, "UNKNOWN": 3, "BAD_REQUEST": 4, "AUTH_ERROR": 5}
        collapsed = []
        for m in models:
            marble = sorted(by_model.get(m, [("UNKNOWN", "no probe")]),
                            key=lambda t: order.get(t[0], 9))[0]
            collapsed.append((m, marble[0], marble[1]))
        return collapsed
    return [(m, ("OK" if c == "OK" else ("RATE_LIMITED" if c == "RATE_LIMITED" else c)), msg)
            for m, c, msg in [(m, c, msg) for m, c, msg in results]]


# ---------- input helpers ----------

def input_keys(prompt_name, existing=None):
    """Collect keys. Returns (added, removed): new key strings + existing keys to drop."""
    existing = list(existing or [])
    print(f"\nEnter API keys for {prompt_name} (paste, space/comma/newline separated).")
    if existing:
        print(f"  Current keys ({len(existing)}): " +
              ", ".join(f"[{i}] {snippet(k)}" for i, k in enumerate(existing, 1)))
        print("  Type REMOVE to delete some first.")
    print("Type DONE (or Enter on an empty line) when finished.")
    print("Finishing with nothing typed = keep the existing keys.")
    lines, removed = [], []
    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if line.upper() == "REMOVE" and existing:
            try:
                nums = input("  Numbers to delete, comma/space separated, no ranges (e.g. 1,3 — empty cancels): ").strip()
            except EOFError:
                continue
            if not nums:
                continue
            for tok in nums.replace(",", " ").split():
                if tok.isdigit() and 1 <= int(tok) <= len(existing):
                    k = existing[int(tok) - 1]
                    if k not in removed:
                        removed.append(k)
                        print(f"  [-] will delete {snippet(k)}")
                else:
                    print(f"  [!] #{tok} out of range")
            remaining = [k for k in existing if k not in removed]
            print(f"  Remaining: {', '.join(snippet(k) for k in remaining) or '(none)'}")
            continue
        if line.upper() == "DONE" or (not line and lines):
            break
        if not line and not lines:
            return [], removed  # keep existing (minus removals)
        if line:
            lines.append(line)
    added = [k.strip() for k in " ".join(lines).replace(",", " ").split() if k.strip()]
    return added, removed


def input_models_manual(pid, pname, existing):
    print(f"\nModels for {pname} (manual entry — catalog unreachable).")
    if pid in MODEL_HINTS:
        print(MODEL_HINTS[pid])
    elif pid == "custom" or pid.startswith("custom_"):
        print(MODEL_HINTS["custom"])
    if existing:
        print(f"Current: {' '.join(existing)}")
    print("Type model IDs separated by commas or spaces.")
    print("Empty = keep current. 'CLEAR' = replace all.")
    try:
        raw = input("Models: ").strip()
    except EOFError:
        return existing
    if not raw:
        return existing
    if raw.upper() == "CLEAR":
        return []
    return [t.split("/")[-1] if t.startswith("opencode/") else t
            for t in raw.replace(",", " ").split() if t.strip()]


def check_models_against_available(models, available):
    if not available or not models:
        return
    avail_set = set(available)
    # normalize gemini models/ prefix
    norm_avail = {a.replace("models/", "") for a in avail_set}
    unknown = [m for m in models if m not in avail_set and m.split("/")[-1] not in norm_avail and m not in norm_avail]
    if unknown:
        print(f"  [!] These models were NOT in the provider list: {' '.join(unknown)}")
        print("      (may be new/preview IDs — allowed, but double-check spelling)")


# ---------- per-provider flow with validation gate ----------

def _custom_id(name):
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "endpoint"
    return f"custom_{slug}"


def create_custom_provider(db):
    """Prompt for name + base URL of an OpenAI-compatible endpoint.

    Returns a dynamic provider dict (or None on abort). The entry is created
    in db immediately so keys/models/steps below have somewhere to live.
    """
    print("\nCustom OpenAI-compatible endpoint (DeepSeek direct, Groq, Mistral, xAI, Together, ...)")
    print("Works with any API shaped like OpenAI: GET {base}/models + POST {base}/chat/completions.")
    try:
        name = input("Short name (e.g. DeepSeek direct): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not name:
        print("  [!] Name required.")
        return None
    pid = _custom_id(name)
    cur_base = db.get(pid, {}).get("base_url", "")
    if cur_base:
        print(f"  Current base URL: {cur_base}")
    try:
        base = input("Base URL including /v1 if the docs show one (e.g. https://api.deepseek.com/v1).\nEmpty keeps the shown URL, or aborts if this is a new endpoint: ").strip().rstrip("/")
    except (EOFError, KeyboardInterrupt):
        return None
    if not base:
        base = cur_base
    if not base:
        print("  [!] Base URL required.")
        return None
    base = _valid_url(base)
    if not base:
        print("  [!] Invalid Base URL (expected e.g. https://api.deepseek.com/v1).")
        return None
    if pid not in db:
        db[pid] = {"keys": [], "models": [], "endpoints": []}
    db[pid]["base_url"] = base
    db[pid]["label"] = name
    save_db(db)
    print(f"  [+] Endpoint: {base}")
    return {"id": pid, "name": f"{name} (custom)", "prefix": "openai/",
            "base_url": base, "type": "custom_api"}


def _custom_entries(db):
    """(pid, label) for user-created custom endpoints, sorted."""
    out = []
    for pid, entry in db.items():
        if pid == "custom" or pid.startswith("custom_"):
            out.append((pid, entry.get("label") or pid))
    return sorted(out)


def _named_custom_provider(db, name):
    """create_custom_provider for `add <name>`: skips the name prompt, only asks for the base URL.

    Reuses an existing custom slot when the name matches (exact or substring),
    so adding keys never forks a duplicate provider.
    """
    pid = _custom_id(name)
    if pid not in db:
        for epid, label in _custom_entries(db):
            q, l = _norm_name(name), _norm_name(label)
            if q == l or (len(q) >= 3 and len(l) >= 3 and (q in l or l in q)):
                pid = epid
                break
    base = db.get(pid, {}).get("base_url", "")
    if not base:
        try:
            base = input(f"Base URL for {name} (e.g. https://api.deepseek.com/v1, empty aborts): ").strip().rstrip("/")
        except (EOFError, KeyboardInterrupt):
            return None
    if not base:
        return None
    base = _valid_url(base)
    if not base:
        print(f"  [!] Invalid Base URL for {name} (expected e.g. https://api.deepseek.com/v1).")
        return None
    db.setdefault(pid, {"keys": [], "models": [], "endpoints": []})
    db[pid]["base_url"] = base
    db[pid]["label"] = name
    save_db(db)
    print(f"  [+] Endpoint: {base}")
    return {"id": pid, "name": f"{name} (custom)", "prefix": "openai/",
            "base_url": base, "type": "custom_api"}


def configure_provider(db, provider):
    pid = provider["id"]
    pname = provider["name"]
    ptype = provider["type"]
    if pid not in db:
        db[pid] = {"keys": [], "models": [], "endpoints": []}
    entry = db[pid]
    saved_snapshot = json.loads(json.dumps(entry))  # for abort

    # --- step 1: endpoints ---
    if ptype == "local_ollama":
        cur = entry.get("endpoints", [])
        if cur:
            print(f"  Current endpoint(s): {' '.join(cur)}")
        try:
            ep = input("Local Ollama URL (default http://localhost:11434 — empty keeps saved, or uses default): ").strip()
        except EOFError:
            return False
        endpoints = cur if not ep else [ep]
        if not endpoints:
            endpoints = ["http://localhost:11434"]
            entry["endpoints"] = endpoints
        if endpoints != cur:
            entry["endpoints"] = endpoints
        # GATE: endpoint must be reachable before models step
        while True:
            ok, msg, avail = _ollama_local_ep(endpoints[0])
            print(f"  [{'OK' if ok else 'FAIL'}] {endpoints[0]} -> {msg}")
            if ok:
                catalog = [(n, n) for n in (avail or [])]
                return _models_step(db, provider, avail, catalog, [], endpoints)
            nxt = input("  Endpoint unreachable. [R]etry / [C]hange / [A]bort? ").strip().lower()
            if nxt == "c":
                try:
                    ep = input("Local Ollama URL (empty keeps the current one): ").strip()
                except EOFError:
                    db[pid] = saved_snapshot
                    return False
                if ep:
                    endpoints = [ep]
                    entry["endpoints"] = endpoints
            elif nxt == "a":
                db[pid] = saved_snapshot
                print("  [-] Aborted, nothing saved.")
                return False
            # r = retest loop

    if ptype == "remote_ollama":
        cur = entry.get("endpoints", [])
        if cur:
            print(f"  Current endpoint(s): {' '.join(cur)}")
        try:
            ep = input("Remote Ollama base URL (empty keeps saved — required the first time): ").strip()
        except EOFError:
            return False
        if ep and ep not in entry["endpoints"]:
            entry["endpoints"] = [ep]
        if not entry["endpoints"]:
            print("  [!] No endpoint set, aborting.")
            db[pid] = saved_snapshot
            return False

    # --- step 2: keys + GATE (all must pass before models step) ---
    if ptype not in ("local_ollama",):
        pending_new, pending_rm = input_keys(pname, entry.get("keys", []))
        # candidate set = existing minus removals + new (dedup, preserve order)
        candidate = [k for k in entry.get("keys", []) if k not in pending_rm]
        if pending_rm:
            print(f"  [-] Dropped {len(pending_rm)} key(s).")
        for k in pending_new:
            if k not in candidate:
                candidate.append(k)
        if not candidate:
            print("  [!] No keys (existing or new). Aborting provider.")
            db[pid] = saved_snapshot
            return False
        if pid == "custom" or pid.startswith("custom_"):
            if not entry.get("base_url") and not provider.get("base_url"):
                try:
                    b = input("Base URL including /v1 if the docs show one (empty aborts): ").strip().rstrip("/")
                except (EOFError, KeyboardInterrupt):
                    db[pid] = saved_snapshot
                    return False
                if not b:
                    print("  [!] Base URL required for custom providers.")
                    db[pid] = saved_snapshot
                    return False
                b = _valid_url(b)
                if not b:
                    print("  [!] Invalid Base URL (expected e.g. https://api.deepseek.com/v1).")
                    db[pid] = saved_snapshot
                    return False
                entry["base_url"] = b
            elif provider.get("base_url"):
                entry["base_url"] = provider["base_url"]
            endpoints = [entry["base_url"]]
        else:
            endpoints = entry.get("endpoints", [])
        while True:
            results, avail = validate_keys(pid, candidate, endpoints)
            bad = [r for r in results if not r[1]]
            if not bad:
                entry["keys"] = candidate
                normalize_credentials(entry)
                for k, ok, msg in results:
                    cid = next((c.get("id") for c in entry.get("credentials", [])
                                if isinstance(c, dict) and str(c.get("secret")) == str(k)), None)
                    if cid:
                        record_validation(db, pid, cid, "ok", msg)
                try:
                    quota_grouping_prompt(db, pid, candidate)
                except Exception as e:
                    print(f"  [!] Quota grouping skipped: {e}")
                print("  [+] All keys valid — proceeding to models.")
                break
            print(f"  [!] {len(bad)}/{len(results)} key(s) FAILED. Next step blocked until all OK.")
            print("  [R]e-enter keys / [K]eep only valid ones / [S]ave anyway (expect errors later) / [A]bort")
            try:
                ch = input("  Choice [R/K/S/A]: ").strip().lower()
            except EOFError:
                db[pid] = saved_snapshot
                return False
            if ch == "k":
                candidate = [k for (k, ok, _) in results if ok]
                if not candidate:
                    print("  [!] No valid keys left, aborting.")
                    db[pid] = saved_snapshot
                    return False
                entry["keys"] = candidate
                normalize_credentials(entry)
                try:
                    quota_grouping_prompt(db, pid, candidate)
                except Exception as e:
                    print(f"  [!] Quota grouping skipped: {e}")
                print("  [+] Kept valid keys only — proceeding to models.")
                break
            elif ch == "a":
                db[pid] = saved_snapshot
                print("  [-] Aborted, nothing saved.")
                return False
            elif ch == "s":
                entry["keys"] = candidate
                normalize_credentials(entry)
                for k, ok, msg in results:
                    cid = next((c.get("id") for c in entry.get("credentials", [])
                                if isinstance(c, dict) and str(c.get("secret")) == str(k)), None)
                    if cid:
                        record_validation(db, pid, cid, "ok" if ok else "invalid", msg)
                print("  [!] Saved anyway with failing keys — expect proxy 401s for this provider.")
                break
            else:  # re-enter
                pending_new, pending_rm = input_keys(pname + " (retry)", entry.get("keys", []))
                candidate = [k for k in entry.get("keys", []) if k not in pending_rm]
                for k in pending_new:
                    if k not in candidate:
                        candidate.append(k)
                if not candidate:
                    candidate = [k for (k, ok, _) in results if ok]
                if not candidate:
                    print("  [!] Still no keys, aborting.")
                    db[pid] = saved_snapshot
                    return False
        # live catalog for picker (rich labels); fallback to key-validation ids
        ep0 = (entry.get("endpoints", []) or [None])[0]
        if pid == "tokenrouter" and entry.get("keys"):
            _, _, _, tr_base = _tokenrouter_key(entry["keys"][0])
            if tr_base:
                entry["base_url"] = tr_base
                print(f"  [+] TokenRouter endpoint: {tr_base}")
                ep0 = tr_base
        if (pid == "custom" or pid.startswith("custom_")) and entry.get("base_url"):
            ep0 = entry["base_url"]
        catalog = fetch_catalog(pid, entry["keys"][0], ep0) if entry.get("keys") else None
        if not catalog and avail:
            catalog = [(m, m) for m in avail]
        if (pid == "custom" or pid.startswith("custom_")) and entry.get("base_url"):
            step_ep = [entry["base_url"]]
        else:
            step_ep = [entry["base_url"]] if pid == "tokenrouter" and entry.get("base_url") else entry.get("endpoints", [])
        return _models_step(db, provider, avail, catalog, entry.get("keys", []),
                            step_ep)
    return False


def _models_step(db, provider, avail, catalog, keys, endpoints):
    pid = provider["id"]
    entry = db[pid]
    if catalog:
        entry["catalog_checked_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        entry["catalog_source"] = "live provider catalog"
        # retirement check: previously configured but no longer advertised
        cat_ids = {mid for mid, _ in catalog}
        retired = [m for m in (entry.get("models") or []) if m not in cat_ids]
        if retired:
            print(f"  [!] Previously configured model(s) no longer advertised: {' '.join(retired)}")
            try:
                keep = input("  Keep anyway (marked stale, excluded from healthy claims)? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                keep = "n"
            if keep not in ("y", "yes"):
                entry["models"] = [m for m in entry.get("models", []) if m in cat_ids]
                print(f"  [-] Dropped retired model(s): {' '.join(retired)}")
            else:
                entry.setdefault("stale_models", [])
                for m in retired:
                    if m not in entry["stale_models"]:
                        entry["stale_models"].append(m)
                print("  [i] Kept as stale (deliberate, documented).")
        picked = pick_models(pid, provider["name"], entry.get("models", []), catalog)
        if not picked and entry.get("models"):
            picked = list(entry["models"])  # DONE with nothing new = keep
        candidate = []
        for m in list(entry.get("models", [])) + picked:
            if m not in candidate:
                candidate.append(m)
    else:
        merged = input_models_manual(pid, provider["name"], entry.get("models", []))
        if merged == []:
            entry["models"] = []
            save_db(db)
            print(f"\n[+] Cleared models for {provider['name']}.")
            return True
        candidate = []
        for m in list(entry.get("models", [])) + merged:
            if m not in candidate:
                candidate.append(m)
    if not candidate:
        print("  [!] No models selected.")
        return False
    # GATE: per-model direct test before saving. Default FAST (one key per
    # model); STRICT/SAMPLE selectable for per-credential differences.
    test_key = (keys or [None])[0]
    test_ep = (endpoints or [None])[0] if endpoints else None
    settings = db.get(SETTINGS_KEY) or {}
    vmode = settings.get("validation_mode", "FAST")
    vsize = settings.get("sample_size", 2)
    if test_key or pid == "ollama_local":
        while True:
            if vmode in ("STRICT", "SAMPLE") and keys and len(keys) > 1 and pid != "ollama_local":
                n_probes = len(candidate) * (len(keys) if vmode == "STRICT" else min(vsize, len(keys)))
                print(f"  [*] Validation mode {vmode}: ~{n_probes} upstream probes.")
                try:
                    ok = input("  Proceed? [Y/n]: ").strip().lower()
                except EOFError:
                    return False
                if ok not in ("", "y", "yes"):
                    vmode = "FAST"
            results = test_models(pid, candidate, test_key, test_ep, keys=keys,
                                  mode=vmode, sample_size=vsize, db=db)
            fails = [r for r in results if r[1] not in ("OK", "RATE_LIMITED")]
            waits = [r for r in results if r[1] == "RATE_LIMITED"]
            if waits:
                print(f"  [~] {len(waits)} throttled (429) but valid — kept: {' '.join(m for m, _, _ in waits)}")
            if not fails:
                print("  [+] All models passed (or throttled-but-valid) — saving.")
                break
            by_cls = {}
            for m, cls, msg in fails:
                by_cls.setdefault(cls, []).append(m)
            print(f"  [!] {len(fails)}/{len(results)} model(s) FAILED. Save blocked. " +
                  ", ".join(f"{len(v)}x {k}" for k, v in sorted(by_cls.items())))
            print("  [K]eep passing ones only / [R]e-pick models / [A]bort without saving")
            try:
                ch = input("  Choice [K/R/A]: ").strip().lower()
            except EOFError:
                return False
            if ch == "k":
                candidate = [m for (m, st, _) in results if st in ("OK", "RATE_LIMITED")]
                if not candidate:
                    print("  [!] Nothing passing left.")
                    return False
                break
            elif ch == "a":
                print("  [-] Aborted, models unchanged.")
                return False
            else:
                if catalog:
                    candidate = pick_models(pid, provider["name"], [], catalog)
                    if not candidate:
                        return False
                else:
                    candidate = input_models_manual(pid, provider["name"], [])
                    if not candidate:
                        return False
    else:
        check_models_against_available(candidate, avail)
    # --- auto-unify every stem appearing >=2 times across providers (silent) ---
    # respects version and tier (flash/pro/lite never merged) via canonical_stem
    try:
        _auto_unify_stems(db, pid, candidate)
    except Exception as e:
        print(f"  [!] Auto-unify skipped: {e}")
    entry["models"] = candidate
    save_db(db)
    try:
        n = generate_yaml(db)
    except ValueError as e:
        print(f"  [!] Config not written: {e}")
        return False
    print(f"\n[+] Saved {provider['name']}: {len(entry.get('keys', []))} key(s), "
          f"{len(entry.get('models', []))} model(s). Total routes: {n}")
    return True


# ---------- proxy tests: gateway smoke (T) / pool test (P) / full sweep (F) ----------

def _gateway_request(alias, master_key, timeout=25):
    """One minimal chat completion via the gateway. Returns (class, msg)."""
    payload = json.dumps({"model": alias, "messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 1}).encode()
    req = urllib.request.Request(
        f"{PROXY_URL}/v1/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {master_key}", "Content-Type": "application/json",
                 "User-Agent": "litellm-wizard/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode(errors="replace"))
            if r.status == 200 and "choices" in body:
                return "OK", "choices OK"
            return "UNKNOWN", "unexpected body"
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode(errors="replace")
            try:
                msg = json.loads(raw).get("error", {}).get("message", raw)[:150].replace("\n", " ")
            except Exception:
                msg = raw[:150].replace("\n", " ")
        except Exception:
            msg = str(e)[:150]
        return classify_http_error(e.code, msg), f"HTTP {e.code}: {msg}"
    except Exception as e:
        return classify_http_error(None, str(e)), str(e)[:150]


def proxy_smoke_test():
    """Gateway smoke test: does each logical alias work through LiteLLM?
    One request per alias. Structured classification (429 != auth error)."""
    if not os.path.exists(YAML_FILE):
        print("[!] No config.yaml, skipping proxy test.")
        return {}
    with open(YAML_FILE) as f:
        cfg = yaml.safe_load(f) or {}
    model_list = cfg.get("model_list", [])
    seen, uniq = set(), []
    for item in model_list:
        a = item.get("model_name")
        if a not in seen:
            seen.add(a)
            uniq.append(item)
    print(f"\n[*] Gateway smoke test: {len(uniq)} unique aliases ({len(model_list)} routes) via {PROXY_URL}")
    counts = {}
    master_key = get_master_key()
    for i, item in enumerate(uniq, 1):
        alias = item.get("model_name")
        cls, msg = _gateway_request(alias, master_key)
        counts[cls] = counts.get(cls, 0) + 1
        print(f"  [{i:02d}/{len(uniq):02d}] [{cls}] {alias}" + ("" if cls == "OK" else f" -> {msg[:120]}"))
        time.sleep(2.0)
    ok = counts.get("OK", 0)
    print(f"\n  === Gateway: {ok} OK | {counts.get('RATE_LIMITED', 0)} throttled | "
          f"{counts.get('AUTH_ERROR', 0)} auth errors | {counts.get('BAD_REQUEST', 0)} bad requests | "
          f"{counts.get('SERVER_ERROR', 0) + counts.get('TIMEOUT', 0) + counts.get('UNAVAILABLE', 0)} server/transient | "
          f"{counts.get('UNKNOWN', 0)} unknown | {len(uniq)} aliases ===")
    return counts


def pool_health_test(db, sleep_s=1.5):
    """Pool/deployment test: does each underlying deployment work directly?

    Tests every deployment independently via provider adapters (no gateway).
    Shows estimated probe count and asks confirmation when large. 429 =
    throttled (kept), never a permanent failure; one transient 500 never
    removes a deployment.
    """
    deployments, pools, _, errors = compile_config(db)
    if errors:
        print(f"  [!] Cannot test: {'; '.join(errors[:5])}")
        return {}
    n = len(deployments)
    print(f"\n[*] Pool test: {n} deployments across {len(pools)} pools (direct provider calls).")
    if n > 40:
        try:
            ans = input(f"  {n} upstream probes — proceed? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  [-] Pool test cancelled.")
            return {}
        if ans not in ("", "y", "yes"):
            print("  [-] Pool test cancelled.")
            return {}
    counts = {}
    for i, d in enumerate(deployments, 1):
        pid, upstream = d["provider"], d["upstream_model"]
        secret = d.get("secret") if d.get("credential_id") else None
        ep = d.get("endpoint")
        if pid == "ollama_local":
            cls, msg = probe_model_classified(pid, upstream, None, ep)
        elif not secret:
            cls, msg = "UNAVAILABLE", "no credential stored"
        else:
            cls, msg = probe_model_classified(pid, upstream, secret, ep)
        counts[cls] = counts.get(cls, 0) + 1
        qd = (d.get("quota_domain") or "")[:28]
        print(f"  [{i:02d}/{n:02d}] [{cls}] {d['logical_model']} via {pid} ({qd})"
              + ("" if cls == "OK" else f" -> {msg[:110]}"))
        time.sleep(sleep_s)
    ok = counts.get("OK", 0)
    print(f"\n  === Pool: {ok} OK | {counts.get('RATE_LIMITED', 0)} throttled | "
          f"{counts.get('AUTH_ERROR', 0)} auth errors | {len(deployments)} deployments ===")
    return counts


def full_sweep(db):
    """Full sweep: credentials + models + deployments + aliases + gateway
    consistency. Slower; explicitly requested via F."""
    print("\n=== Full sweep: DB consistency + deployments + gateway ===")
    issues = validate_db(db)
    cleaned, emptied = normalize_aliases(db)
    if cleaned:
        print(f"  [i] normalize_aliases: dropped {cleaned} stale member(s)" +
              (f"; emptied: {', '.join(emptied)}" if emptied else ""))
        save_db(db)
    for line in issues:
        print(f"  [!] {line}")
    if not issues and not cleaned:
        print("  [OK] DB structure sound.")
    deployments, pools, roles, errors = compile_config(db)
    if errors:
        for e in errors[:8]:
            print(f"  [!] {e}")
        return
    print(f"  [i] {len(deployments)} deployments / {len(pools)} pools / {len(roles)} roles compile clean.")
    try:
        with open(YAML_FILE) as f:
            cfg = yaml.safe_load(f) or {}
        print(f"  [OK] config.yaml parses ({len(cfg.get('model_list', []))} routes).")
    except Exception as e:
        print(f"  [!] config.yaml unreadable: {e}")
    pool_health_test(db)
    proxy_smoke_test()
    print("  === Sweep complete ===")


def wait_for_proxy(timeout_s=60):
    """Bounded readiness poll (replaces blind fixed sleep)."""
    deadline = time.time() + timeout_s
    url = f"{PROXY_URL}/health"
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers=dict(UA), method="GET")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        # fallback: /v1/models with master key proves readiness too
        try:
            req = urllib.request.Request(
                f"{PROXY_URL}/v1/models",
                headers={"Authorization": f"Bearer {get_master_key()}"}, method="GET")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def restart_proxy():
    import shutil as _shutil
    backup = None
    try:
        if os.path.exists(YAML_FILE):
            backup = YAML_FILE + f".prev-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            _shutil.copy2(YAML_FILE, backup)
        subprocess.run(["systemctl", "--user", "restart", "litellm"], check=True)
        print("[*] Restart requested — waiting for readiness (bounded 60s)...")
        if wait_for_proxy(60):
            print("[+] LiteLLM ready.")
            return True
        print("[!] Service did not become ready within 60s.")
        if backup:
            print(f"[!] Previous config retained at {backup} — rollback: cp {backup} {YAML_FILE} && systemctl --user restart litellm")
        print("[i] Diagnose: systemctl --user status litellm --no-pager | journalctl --user -u litellm -n 50 --no-pager")
        return False
    except Exception as e:
        print(f"[!] Restart failed: {e}")
        if backup and os.path.exists(backup):
            print(f"[i] Previous config retained at {backup}")
        print("[i] Diagnose: systemctl --user status litellm.service --no-pager")
        return False


def _is_configured(entry):
    return bool(entry.get("keys") or entry.get("models") or entry.get("endpoints"))


def _provider_flag(db, pid, pdata):
    """Short plain-language flag for the one-line status, or ''.

    Normal mode stays quiet: no news is good news. Only surfaces things
    the user can act on (disabled, known speed, shared quota).
    """
    try:
        if pdata.get("disabled"):
            return " [disabled — skipped, settings kept]"
        creds = list(iter_credentials(pdata))
        domains = {c.get("quota_domain") for c in creds if c.get("quota_domain")}
        deps = [d for d in build_deployments_quiet(db) if d["provider"] == pid]
        cap = estimate_capacity(db, [(d["provider"], d["upstream_model"]) for d in deps])
        bits = []
        if cap["rpm_known"]:
            bits.append(f"~{cap['rpm_known']}/min")
        if len(creds) > 1 and len(domains) == 1:
            bits.append("keys share 1 quota")
        if not bits:
            return ""
        return " (" + ", ".join(bits) + ")"
    except Exception:
        return ""


def build_deployments_quiet(db):
    try:
        deps, _, _, _ = compile_config(db)
        return deps
    except Exception:
        return []


def _plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _status_line(num, p, e, db=None):
    models = ' '.join(e.get('models', [])[:4])
    more = f" +{len(e['models']) - 4} more" if len(e.get('models', [])) > 4 else ""
    flag = _provider_flag(db, p["id"], e) if db is not None else ""
    return (f"  [{num}] {p['name']:<30} {_plural(len(e.get('keys', [])), 'key')}, "
            f"{_plural(len(e.get('models', [])), 'model')}  {models}{more}{flag}").rstrip()


def _custom_status_line(label, e, db, pid):
    models = ' '.join(e.get('models', [])[:4])
    return (f"  [C] {label + ' (custom)':<30} {_plural(len(e.get('keys', [])), 'key')}, "
            f"{_plural(len(e.get('models', [])), 'model')}  {models}"
            f"{_provider_flag(db, pid, e) if db is not None else ''}").rstrip()


def needs_quota_hint(db, pid):
    """True only while project-scoped multi-key grouping is still ambiguous.

    Shows for e.g. 2+ Google keys that were never grouped and whose owner
    hasn't seen the explanation yet. Silent for single keys, grouped keys,
    non-project providers, and reviewed providers.
    """
    meta = PROVIDER_META.get(pid)
    if not meta or meta.get("quota_scope") != "project":
        return False
    pdata = db.get(pid)
    if not isinstance(pdata, dict) or pdata.get("quota_reviewed"):
        return False
    creds = list(iter_credentials(pdata))
    if len(creds) < 2:
        return False
    return all(effective_quota_domain(pid, c).startswith("credential:") for c in creds)


def _quota_hint_line(pid):
    name = next((p["name"].split()[0] for p in PROVIDERS.values() if p["id"] == pid),
                "this provider")
    return (f"      Same {name} project for all keys? "
            "Type 'quota' so the speed math stays honest.")


def print_status(db, verbose=False):
    if verbose:
        print("\nAll providers (detail):")
        for num in sorted(PROVIDERS, key=int):
            p = PROVIDERS[num]
            print(_status_line(num, p, db.get(p["id"], {}), db))
            _print_provider_domains(db, p["id"])
        for pid, label in _custom_entries(db):
            e = db.get(pid, {})
            print(_custom_status_line(label, e, None, pid))
            _print_provider_domains(db, pid)
        _print_alias_summary(db, verbose=True)
        _print_roles_summary(db)
        return
    print("\nYour providers:")
    any_cfg = False
    for num in sorted(PROVIDERS, key=int):
        p = PROVIDERS[num]
        if p["id"] == "custom":
            continue  # template, not a real endpoint
        e = db.get(p["id"], {})
        if _is_configured(e):
            print(_status_line(num, p, e, db))
            if needs_quota_hint(db, p["id"]):
                print(_quota_hint_line(p["id"]))
            any_cfg = True
    for pid, label in _custom_entries(db):
        e = db.get(pid, {})
        if _is_configured(e):
            print(_custom_status_line(label, e, db, pid))
            any_cfg = True
    if not any_cfg:
        print("  (none yet — try: add google)")
    _print_alias_summary(db)
    _print_roles_summary(db)
    _print_diversity_hint(db)


def quota_domains_list(db):
    """Structured quota-domain facts for the TUI dashboard (no secrets).

    One entry per domain per provider: domain id, key count, RPM when
    set, Google project id when known, and the credential ids (masked
    counts only — the raw ids are safe, the secrets never leave the DB).
    """
    domains = []
    for pid, pdata in db.items():
        if pid.startswith("_") or not isinstance(pdata, dict):
            continue
        buckets = {}
        for c in iter_credentials(pdata):
            qd = effective_quota_domain(pid, c)
            buckets.setdefault(qd, []).append(c)
        for qd, creds in sorted(buckets.items()):
            qmeta = (db.get(QUOTA_KEY) or {}).get(qd) or {}
            domains.append({
                "domain_id": qd,
                "provider": pid,
                "key_count": len(creds),
                "rpm": qmeta.get("rpm"),
                "confidence": qmeta.get("confidence", ""),
                "project_id": creds[0].get("project_id", "") if creds else "",
                "keys": [c.get("id", "?") for c in creds[:5]],
            })
    return domains


def _print_provider_domains(db, pid):
    pdata = db.get(pid, {}) if isinstance(db.get(pid), dict) else {}
    buckets = {}
    for c in iter_credentials(pdata):
        buckets.setdefault(c.get("quota_domain", "?"), []).append(c)
    if not buckets:
        return
    if all(qd.startswith("credential:") for qd in buckets):
        n = sum(len(v) for v in buckets.values())
        if n <= 1:
            return
        print(f"      speed limits: {_plural(n, 'key')} counted separately (default). "
              "If some share one provider plan, group them with 'quota'.")
        return
    for qd in sorted(buckets):
        q = (db.get(QUOTA_KEY) or {}).get(qd) or {}
        rpm = q.get("rpm")
        print(f"      speed limit {qd}: {_plural(len(buckets[qd]), 'key')}" +
              (f", about {rpm}/min" if rpm else ", speed unknown"))


def _print_diversity_hint(db):
    try:
        deps = build_deployments_quiet(db)
        provs = {d["provider"] for d in deps}
        if len(deps) >= 8 and len(provs) <= 1:
            print("\n  Tip: almost everything runs through one provider. "
                  "Adding a second provider helps more than adding more keys.")
    except Exception:
        pass


def _print_roles_summary(db):
    roles = db.get(ROLES_KEY) or {}
    if not roles:
        return
    print("\n  Shortcuts (one word tries models in order):")
    try:
        _, pools, _, _ = compile_config(db)
    except Exception:
        pools = {}
    for role in sorted(roles):
        spec = roles[role] if isinstance(roles[role], dict) else {}
        valid = [p for p in (spec.get("pools") or []) if p in pools]
        fb = [p for p in (spec.get("fallback") or []) if p in pools]
        print(f"    {role} = {', '.join(valid) or '(nothing working right now)'}"
              + (f"  (then: {', '.join(fb)})" if fb else ""))


def _pool_labels(db, members):
    """(provider labels, has_stale) for alias members that still exist."""
    labels, stale = [], False
    for mem in members or []:
        if not isinstance(mem, dict):
            stale = True
            continue
        pid, m = mem.get("provider"), mem.get("model")
        pdata = db.get(pid, {}) if pid else {}
        if pid and m and m in (pdata.get("models") or []):
            label = pdata.get("label") or pid
            if label not in labels:
                labels.append(label)
        else:
            stale = True
    return labels, stale


def _print_alias_summary(db, verbose=False):
    aliases = _get_aliases(db)
    if not aliases:
        return
    print("\n  Combined models (one name, served from everywhere listed):")
    try:
        _, pools, _, _ = compile_config(db)
    except Exception:
        pools = {}
    for canonical in sorted(aliases):
        members = aliases[canonical]
        if not isinstance(members, list):
            continue
        labels, stale = _pool_labels(db, members)
        if not labels:
            continue
        line = f"    {canonical} — via {', '.join(labels)}"
        if stale:
            line += "  (one source stopped offering it)"
        if verbose:
            pool_deps = pools.get(canonical, [])
            named = sorted({d.get("quota_domain") for d in pool_deps
                            if d.get("quota_domain") and not d["quota_domain"].startswith("credential:")})
            hstates = [d.get("health") for d in pool_deps]
            if any(h not in ("unknown", None) for h in hstates):
                ok = hstates.count("healthy")
                slow = hstates.count("throttled") + hstates.count("partially-throttled")
                line += f"  [{ok} working, {slow} slow]"
            cap = estimate_capacity(db, [(d["provider"], d["upstream_model"]) for d in pool_deps]) if pool_deps else None
            if cap and cap["rpm_known"]:
                line += f"  [~{cap['rpm_known']}/min]"
            if named:
                line += f"  [shared limits: {', '.join(named)}]"
        print(line)


# ---------- unified alias (cross-provider single model_name) ----------

def _all_model_refs(db):
    """List of (pid, model, label) for every model currently in DB."""
    out = []
    for pid, pdata in db.items():
        if pid == ALIAS_KEY or pid == "_unified":
            continue
        if not isinstance(pdata, dict):
            continue
        for m in pdata.get("models", []) or []:
            label = pdata.get("label") or pid
            out.append((pid, m, label))
    return out


def _alias_stem(m):
    """Version/flash-safe stem: bare id stripped of provider prefix, free markers and date, never strips flash/pro/lite tier."""
    s = (m or "").strip().lower().replace("_", "-").replace(" ", "-")
    s = re.sub(r"-{2,}", "-", s)
    bare = s.rsplit("/", 1)[-1].strip(".-_")
    # Iterative strip: handles combos like -0731:free or -free-0731
    while True:
        orig = bare
        if bare.endswith(":free") or bare.endswith("-free"):
            bare = bare[:-5]
        # date suffix -MMDD / -YYMMDD (digits only after dash, not .3 like 5.3)
        if re.match(r".*-\d{3,4}$", bare) or re.match(r".*-\d{6,8}$", bare):
            bare = re.sub(r"-\d{3,4}$", "", bare)
            bare = re.sub(r"-\d{6,8}$", "", bare)
        bare = re.sub(r"-{2,}", "-", bare).strip("-")
        if bare == orig:
            break
    return bare


def _suggest_alias_groups(db):
    """Safe grouping: {stem: {'members': [...], 'mode': AUTO|SUGGESTED}}.

    AUTO: same stem + compatible known capability tiers.
    SUGGESTED: same stem, plausible but capability-uncertain.
    Incompatible tiers (flash vs pro, lite vs full, reasoning vs not)
    never return AUTO.
    """
    refs = _all_model_refs(db)
    groups = {}
    for pid, m, _ in refs:
        stem = _alias_stem(m)
        groups.setdefault(stem, []).append({"provider": pid, "model": m})
    out = {}
    for stem, members in groups.items():
        uniq = {(d["provider"], d["model"]) for d in members}
        if len(uniq) >= 2:
            members = [dict(t) for t in { (d["provider"], d["model"]): d for d in members }.values()]
            modes = set()
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    modes.add(_group_safety(members[i]["provider"], members[i]["model"],
                                            members[j]["provider"], members[j]["model"]))
            if "MANUAL" in modes and len(modes) == 1:
                continue
            mode = "AUTO" if modes == {"AUTO"} else "SUGGESTED"
            out[stem] = {"members": members, "mode": mode}
    return out


def _auto_unify_stems(db, current_pid, candidate_models):
    """Safe automatic grouping: only AUTO groups merge silently.

    SUGGESTED groups are reported for the user to apply explicitly via the
    alias manager. MANUAL pairs are never touched. Returns stems created.
    """
    # Build combined view including the not-yet-saved candidate for current_pid
    combined_refs = []
    for pid, pdata in db.items():
        if pid == ALIAS_KEY or pid == "_unified":
            continue
        if not isinstance(pdata, dict):
            continue
        models = pdata.get("models", []) or []
        if pid == current_pid:
            models = candidate_models
        for m in models:
            combined_refs.append((pid, m))
    groups = {}
    for pid, m in combined_refs:
        stem = _alias_stem(m)
        if not stem:
            continue
        groups.setdefault(stem, []).append({"provider": pid, "model": m})
    aliases = _get_aliases(db)
    # Ensure db has alias dict if we will create
    changed = []
    for stem, members in groups.items():
        uniq = {}
        for d in members:
            uniq[(d["provider"], d["model"])] = d
        members = list(uniq.values())
        if len(members) < 2:
            continue
        if not stem or not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]*$", stem):
            continue
        # safety gate: all pairs must be AUTO for silent merge
        pair_modes = set()
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair_modes.add(_group_safety(members[i]["provider"], members[i]["model"],
                                             members[j]["provider"], members[j]["model"]))
        if pair_modes != {"AUTO"}:
            print(f"  [?] Suggested pool '{stem}' <- "
                  f"{', '.join(f'{d['provider']}:{d['model']}' for d in members)}")
            print("      (capability-uncertain — apply via 'alias' manager if interchangeable)")
            continue
        existing = aliases.get(stem)
        if existing is None:
            # New alias — create silently
            db.setdefault(ALIAS_KEY, {})[stem] = members
            changed.append(stem)
            print(f"  [+] Auto-grouped '{stem}' <- {', '.join(f'{d['provider']}:{d['model']}' for d in members)}")
        else:
            # Extend existing alias with any missing members (new provider/model added later)
            existing_set = {(d.get("provider"), d.get("model")) for d in existing if isinstance(d, dict)}
            missing = [d for d in members if (d["provider"], d["model"]) not in existing_set]
            if missing:
                # Validate each missing still actually exists in db (candidate already considered)
                db[ALIAS_KEY][stem].extend(missing)
                # dedup just in case
                seen = {}
                for d in db[ALIAS_KEY][stem]:
                    if isinstance(d, dict) and d.get("provider") and d.get("model"):
                        seen[(d["provider"], d["model"])] = d
                db[ALIAS_KEY][stem] = list(seen.values())
                changed.append(stem)
                print(f"  [+] Auto-extended '{stem}' with {', '.join(f'{d['provider']}:{d['model']}' for d in missing)}")
    return changed


def _prompt_alias_canonical(default=""):
    try:
        raw = input(f"Canonical alias [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw:
        return default if default else None
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]*$", raw):
        print("  [!] Alias must start with alnum and contain only a-z 0-9 . _ : -")
        return None
    return raw


def manage_aliases(db):
    """Interactive alias manager loop: list/add/remove/suggest."""
    while True:
        aliases = _get_aliases(db)
        print("\n--- Unified aliases ---")
        if not aliases:
            print("  (none yet — one gateway name can fan out to many providers to dodge rate limits)")
        else:
            for i, (canon, members) in enumerate(sorted(aliases.items()), 1):
                valid = [f"{m.get('provider')}:{m.get('model')}" for m in (members or []) if isinstance(m, dict)]
                print(f"  [{i}] {canon:<30} <- {' , '.join(valid) or '(empty)'}")
        print("\n  [A]dd  [R]emove  [S]uggest  [Q] back")
        print("  Hint: deepseek-v4-flash groups: free/deepseek-v4-flash-0731 (apinex),")
        print("        deepseek/deepseek-v4-flash-free (orcarouter), deepseek-v4-flash:free (tokenharbor)")
        try:
            ch = input("Alias choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if ch in ("q", "quit", "back", "done", "exit", ""):
            break
        if ch in ("s", "suggest"):
            sugg = _suggest_alias_groups(db)
            if not sugg:
                print("  [=] No cross-provider duplicates to suggest (stems all unique).")
                continue
            print("\n  Suggestions (same stem across providers):")
            for stem, info in sorted(sugg.items()):
                members = info["members"] if isinstance(info, dict) else info
                mode = info.get("mode", "?") if isinstance(info, dict) else "?"
                valid = [f"{m['provider']}:{m['model']}" for m in members]
                canon = stem  # suggestion canonical
                # for deepseek keep versioned name: deepseek-v4-flash
                print(f"    [{mode}] {canon:<28} <- {' , '.join(valid)}")
            try:
                ans = input("  Apply a suggestion? Enter canonical to create (e.g. deepseek-v4-flash) or empty to cancel: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if not ans:
                continue
            # need to find members for ans stem
            if ans not in sugg:
                # allow user-typed canonical that is a stem variant: try stem lookup
                stem = _alias_stem(ans)
                info = sugg.get(stem)
                if not info:
                    print(f"  [!] No suggestion for '{ans}' (try Add manually).")
                    continue
                members = info["members"] if isinstance(info, dict) else info
            else:
                info = sugg[ans]
                members = info["members"] if isinstance(info, dict) else info
            # dedup and validate not already aliased
            aliases = _get_aliases(db)
            if ans in aliases:
                print(f"  [!] Alias '{ans}' already exists — remove first or pick new name.")
                continue
            db.setdefault(ALIAS_KEY, {})[ans] = members
            save_db(db)
            n = safe_generate_yaml(db)
            if n is not None:
                print(f"  [+] Created '{ans}' with {len(members)} sources. Routes: {n}")
            continue
        if ch in ("a", "add"):
            refs = _all_model_refs(db)
            if not refs:
                print("  [!] No models in DB yet.")
                continue
            print("\n  Available models (pick numbers to unify under one alias):")
            for i, (pid, m, label) in enumerate(refs, 1):
                print(f"    [{i:2d}] {label:<20} {m}")
            print("  Enter numbers comma/space separated, e.g. 1,2,3  (empty cancels).")
            try:
                raw = input("  Pick: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if not raw:
                continue
            picked = []
            for tok in raw.replace(",", " ").split():
                if tok.isdigit() and 1 <= int(tok) <= len(refs):
                    pid, m, _ = refs[int(tok) - 1]
                    if {"provider": pid, "model": m} not in picked:
                        picked.append({"provider": pid, "model": m})
                else:
                    print(f"  [!] #{tok} out of range")
            if len(picked) < 2:
                print("  [!] Need at least 2 models to unify.")
                continue
            # default canonical: stem of first, or shared stem
            stems = [_alias_stem(d["model"]) for d in picked]
            default = stems[0] if len(set(stems)) == 1 else "my-alias"
            # Special-case deepseek trio: prefer deepseek-v4-flash
            if all("deepseek" in d["model"].lower() for d in picked):
                default = "deepseek-v4-flash"
            canon = _prompt_alias_canonical(default)
            if not canon:
                print("  [-] Cancelled.")
                continue
            aliases = _get_aliases(db)
            if canon in aliases:
                print(f"  [!] Alias '{canon}' already exists.")
                continue
            # warn if canonical collides with bare alias
            bare_aliases = set()
            for pid, pdata in db.items():
                if pid == ALIAS_KEY or pid == "_unified":
                    continue
                for m in pdata.get("models", []) or []:
                    a = m.split("/")[-1] if "/" in m else m
                    # custom_api bare handling matches generate_yaml
                    bare_aliases.add(a)
            if canon in bare_aliases:
                print(f"  [i] '{canon}' already exists as a bare model alias — grouping will replace that single entry with the unified one.")
            db.setdefault(ALIAS_KEY, {})[canon] = picked
            save_db(db)
            n = safe_generate_yaml(db)
            if n is not None:
                print(f"  [+] Unified '{canon}' <- {', '.join(f'{d['provider']}:{d['model']}' for d in picked)}  Routes: {n}")
            continue
        if ch in ("r", "remove", "rm", "delete", "del"):
            aliases = _get_aliases(db)
            if not aliases:
                print("  (nothing to remove)")
                continue
            print("  Remove which? Enter number or canonical name.")
            for i, canon in enumerate(sorted(aliases.keys()), 1):
                print(f"    [{i}] {canon}")
            try:
                raw = input("  Pick #: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if not raw:
                continue
            target = None
            if raw.isdigit() and 1 <= int(raw) <= len(aliases):
                target = sorted(aliases.keys())[int(raw) - 1]
            elif raw in aliases:
                target = raw
            if not target:
                print(f"  [!] No alias '{raw}'")
                continue
            # offer member removal vs whole alias
            members = aliases[target]
            print(f"  Alias '{target}' has {len(members)} members:")
            for i, mem in enumerate(members, 1):
                print(f"    [{i}] {mem.get('provider')}:{mem.get('model')}")
            print("  [A]ll (delete alias) or pick member numbers to remove, empty cancels.")
            try:
                raw2 = input("  Choice: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if not raw2:
                continue
            if raw2 in ("a", "all"):
                del db[ALIAS_KEY][target]
                if not db[ALIAS_KEY]:
                    del db[ALIAS_KEY]
                save_db(db)
                n = safe_generate_yaml(db)
                if n is not None:
                    print(f"  [-] Removed alias '{target}'. Routes: {n}")
            else:
                idxs = []
                for tok in raw2.replace(",", " ").split():
                    if tok.isdigit() and 1 <= int(tok) <= len(members):
                        idxs.append(int(tok) - 1)
                    else:
                        print(f"  [!] #{tok} out of range")
                if not idxs:
                    continue
                for idx in sorted(idxs, reverse=True):
                    del db[ALIAS_KEY][target][idx]
                if not db[ALIAS_KEY][target]:
                    del db[ALIAS_KEY][target]
                    if not db[ALIAS_KEY]:
                        del db[ALIAS_KEY]
                save_db(db)
                n = safe_generate_yaml(db)
                if n is not None:
                    print(f"  [-] Removed {len(idxs)} member(s) from '{target}'. Routes: {n}")
            continue
        print("  Unknown. Use A/R/S/Q.")
    return


PROVIDER_ALIASES = {
    "gemini": ["gemini", "google"],
    "openrouter": ["openrouter", "open_router", "open router", "or"],
    "anthropic": ["anthropic", "claude"],
    "openai": ["openai", "gpt", "chatgpt", "o3"],
    "opencode_zen": ["opencode_zen", "opencode", "zen", "opencode zen"],
    "tokenrouter": ["tokenrouter", "token_router", "token router", "tr"],
    "zai": ["zai", "z.ai", "z ai", "glm", "zhipu", "zhipuai"],
    "ollama_cloud": ["ollama_cloud", "ollama cloud", "remote ollama", "ollama remote",
                     "hosted ollama", "ollama hosted"],
    "ollama_local": ["ollama_local", "ollama local", "local ollama", "local", "localhost"],
    "custom": ["custom", "custom provider", "custom endpoint", "other", "new provider"],
}


def _norm_name(s):
    return s.strip().lower().replace("_", " ").replace(".", " ").replace("-", " ")


def resolve_provider(text, db=None):
    """Fuzzy name -> (num, provider) | (None, [candidate nums]) | (None, []).

    Also matches user-created custom endpoints by label (returns ("C", dyn_dict)).
    """
    q = _norm_name(text)
    # 1. Exact match against builtins or saved custom entries
    hits = []
    for num, p in PROVIDERS.items():
        names = [_norm_name(p["id"]), _norm_name(p["name"])] + PROVIDER_ALIASES.get(p["id"], [])
        if q in names or any(q == a for a in names):
            hits.append(num)
    if len(hits) == 1:
        return hits[0], PROVIDERS[hits[0]]
    if hits:
        return None, hits

    if db:
        for pid, label in _custom_entries(db):
            names = [_norm_name(pid), _norm_name(label),
                     _norm_name(label.replace("(custom)", ""))]
            if q in names or any(q == a for a in names):
                return "C", _custom_provider_dict(pid, db)

    # 2. Substring match against builtins (ignore 2-char aliases to avoid hijacking)
    hits = [num for num, p in PROVIDERS.items()
            if any((q in a or a in q) and len(a) >= 3 for a in
                   ([_norm_name(p["id"]), _norm_name(p["name"])] + PROVIDER_ALIASES.get(p["id"], [])))]
    if len(hits) == 1:
        return hits[0], PROVIDERS[hits[0]]
    if hits:
        return None, hits

    # 3. Substring match against saved custom entries (query as prefix/substring of label)
    if db:
        for pid, label in _custom_entries(db):
            names = [_norm_name(pid), _norm_name(label),
                     _norm_name(label.replace("(custom)", ""))]
            if any(q in a and len(q) >= 3 for a in names):
                return "C", _custom_provider_dict(pid, db)

    return None, []


def _custom_provider_dict(pid, db):
    entry = db.get(pid, {})
    label = entry.get("label") or pid
    return {"id": pid, "name": f"{label} (custom)", "prefix": "openai/",
            "base_url": entry.get("base_url", ""), "type": "custom_api"}


def print_unconfigured(db):
    print("  Not configured yet:")
    for num in sorted(PROVIDERS, key=int):
        p = PROVIDERS[num]
        if not _is_configured(db.get(p["id"], {})):
            print(f"    [{num}] {p['name']}")


# ---------- quota domain management ----------

def manage_quota(db):
    """Interactive speed-limit manager: inspect/assign/create/rename/move."""
    while True:
        qmap = db.get(QUOTA_KEY) or {}
        print("\n--- Speed limits ---")
        print("  Each row is one provider-side limit bucket. Keys in the same")
        print("  bucket share its speed — they don't add up.")
        if not qmap:
            print("  (none yet)")
        auto = sorted(qd for qd in qmap if qd.startswith("credential:"))
        if auto:
            n = sum(len(quota_members(db, qd)) for qd in auto)
            print(f"  default: {_plural(n, 'key')} counted separately "
                  f"({len(auto)} automatic entries)")
        for qd in sorted(qd for qd in qmap if not qd.startswith("credential:")):
            q = qmap[qd] if isinstance(qmap[qd], dict) else {}
            n = len(quota_members(db, qd))
            rpm = q.get("rpm")
            print(f"  {qd:<28} {_plural(n, 'key')}" +
                  (f", about {rpm}/min" if rpm else ", speed unknown"))
            for model, pm in sorted((q.get("per_model") or {}).items()):
                pm_rpm = pm.get("rpm") if isinstance(pm, dict) else None
                if pm_rpm:
                    print(f"      model {model}: about {pm_rpm}/min")
        print("\n  [A] move keys between buckets  [C] new bucket  [R] rename  [L] set speed  [Q] back")
        try:
            ch = input("Quota choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if ch in ("q", "quit", "back", "done", "exit", ""):
            break
        if ch in ("c", "create"):
            try:
                name = input("  New bucket name (e.g. google-project-a): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if not name:
                continue
            ensure_quota_domain(db, name, source="user")
            save_db(db)
            print(f"  [+] Bucket '{name}' created.")
            continue
        if ch in ("r", "rename"):
            try:
                old = input("  Rename which bucket? ").strip()
                new = input("  New name? ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if old in qmap and new and new not in qmap:
                qmap[new] = qmap.pop(old)
                for pid, pdata in db.items():
                    if pid.startswith("_") or not isinstance(pdata, dict):
                        continue
                    for c in pdata.get("credentials", []) or []:
                        if isinstance(c, dict) and c.get("quota_domain") == old:
                            c["quota_domain"] = new
                save_db(db)
                print(f"  [+] Renamed '{old}' -> '{new}'.")
            else:
                print("  [!] Can't rename (name not found, or the new name is taken).")
            continue
        if ch in ("l", "limits"):
            try:
                qd = input("  Which bucket? ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if qd not in qmap:
                print(f"  [!] No bucket '{qd}' (pick one from the list above).")
                continue
            print("  [1] typical value for this provider  [2] I know the real number  "
                  "[3] play it safe (slow)  [4] stop tracking speed\n"
                  "      [5] one specific model (Google free tiers are per project+model)")
            try:
                mode = input("  Choice [1/2/3/4/5]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            q = qmap[qd]
            if mode == "1":
                prov = q.get("provider", "")
                meta = PROVIDER_META.get(prov) or {}
                rpm = (meta.get("default_rpm") or (None,))[0]
                tpm = (meta.get("default_tpm") or (None,))[0]
                q.update({"rpm": rpm, "tpm": tpm, "confidence": "provider_default",
                          "source": "provider default",
                          "updated_at": datetime.datetime.now().isoformat(timespec="seconds")})
                print(f"  [+] '{qd}' set to about {rpm}/min"
                      + (" (typical for this provider — tell me if you know better)" if rpm else " (unknown for this provider)"))
            elif mode == "2":
                try:
                    rpm = input("  Requests per minute (empty = don't know): ").strip()
                    tpm = input("  Tokens per minute (empty = don't know): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    continue
                q.update({"rpm": int(rpm) if rpm else None,
                          "tpm": int(tpm) if tpm else None,
                          "confidence": "manual", "source": "user override",
                          "updated_at": datetime.datetime.now().isoformat(timespec="seconds")})
                print(f"  [+] Saved — your number wins over any default.")
            elif mode == "3":
                q.update({"rpm": 10, "tpm": None, "confidence": "conservative",
                          "source": "conservative preset",
                          "updated_at": datetime.datetime.now().isoformat(timespec="seconds")})
                print("  [+] Playing it safe: about 10/min assumed.")
            elif mode == "4":
                q.update({"rpm": None, "tpm": None, "rpd": None,
                          "confidence": "unknown", "source": "no modeling",
                          "updated_at": datetime.datetime.now().isoformat(timespec="seconds")})
                print("  [+] Speed tracking off for this bucket.")
            elif mode == "5":
                # Google free-tier limits are per project+model: one model
                # can be tighter than the project-wide number.
                try:
                    model = input("  Which model (exact id, e.g. gemini-3.7-flash): ").strip()
                    rpm = input("  Requests/min for this model (empty = don't know): ").strip()
                    tpm = input("  Tokens/min for this model (empty = don't know): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    continue
                if not model:
                    continue
                pm = q.get("per_model") if isinstance(q.get("per_model"), dict) else {}
                pm[model] = {"rpm": int(rpm) if rpm else None,
                             "tpm": int(tpm) if tpm else None,
                             "updated_at": datetime.datetime.now().isoformat(timespec="seconds")}
                q["per_model"] = pm
                print(f"  [+] '{qd}' / {model}: about {pm[model]['rpm'] or '?'}/min"
                      " — beats the bucket-wide number for this model.")
            save_db(db)
            continue
        if ch in ("a", "assign", "move"):
            # pick credential across providers by masked suffix
            allc = []
            for pid, pdata in db.items():
                if pid.startswith("_") or not isinstance(pdata, dict):
                    continue
                for c in iter_credentials(pdata):
                    allc.append((pid, c))
            if not allc:
                print("  (no credentials yet)")
                continue
            for i, (cpid, c) in enumerate(allc, 1):
                qd = c.get("quota_domain", "")
                where = qd if qd and not qd.startswith("credential:") else "separate (default)"
                proj = c.get("project_id") or ""
                proj_str = f" [{proj}]" if proj else ""
                print(f"    [{i:2d}] {cpid:<18} key {_mask_secret(c.get('secret')):<10}{proj_str} in: {where}")
            try:
                raw = input("  Key numbers (empty cancels): ").strip()
                named = sorted({c.get("quota_domain", "") for _, c in allc
                                if c.get("quota_domain") and not c["quota_domain"].startswith("credential:")})
                if named:
                    print(f"  Existing buckets: {', '.join(named)}")
                target = input("  Move into which bucket (existing or new name): ").strip() if raw else ""
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if not raw or not target:
                continue
            ensure_quota_domain(db, target, source="user")
            for tok in raw.replace(",", " ").split():
                if tok.isdigit() and 1 <= int(tok) <= len(allc):
                    cred = allc[int(tok) - 1][1]
                    cred["quota_domain"] = target
                    # project:pid:<id> buckets carry the project identity
                    parts = target.split(":")
                    if len(parts) >= 3 and parts[0] == "project":
                        cred["project_id"] = parts[2]
            save_db(db)
            print(f"  [+] Moved into '{target}'. Shared speed is split fairly when saving.")
            continue
        print("  Unknown. Use A/C/R/L/Q.")
    # Seen the buckets + explanation = reviewed: the status hint retires.
    changed = False
    for pid, pdata in db.items():
        if pid.startswith("_") or not isinstance(pdata, dict):
            continue
        if list(iter_credentials(pdata)) and not pdata.get("quota_reviewed"):
            pdata["quota_reviewed"] = True
            changed = True
    if changed:
        save_db(db)


def quota_grouping_prompt(db, pid, candidate_keys):
    """Bulk quota-grouping workflow after key entry (Google-aware).

    Fast path: single key -> automatic default, no questions. Multi-key on
    project-scoped providers (Google) -> one bulk question, never per-key.
    """
    if len(candidate_keys) <= 1:
        return
    meta = PROVIDER_META.get(pid) or {}
    scope = meta.get("quota_scope", "unknown")
    if scope == "project":
        print(f"\n  One quick question about your {len(candidate_keys)} keys: are they "
              "all from the same Google project?")
        print("  Keys from one project share one speed limit — more keys won't make it faster.")
        try:
            ans = input("  [Y] same project  [S] different projects  [M] I'll sort it later [Y/S/M]: "
                        ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            ans = "s"
        pdata = _provider_entry(db, pid)
        normalize_credentials(pdata)
        creds = [c for c in pdata.get("credentials", []) if str(c.get("secret")) in {str(k) for k in candidate_keys}]
        if not creds:
            # brand-new keys not yet in entry: create temp records
            creds = [{"id": credential_id(k), "secret": k} for k in candidate_keys]
        if ans in ("", "y", "yes"):
            try:
                name = input("  Project nickname (empty = shared:google): ").strip() or f"shared:{pid}"
            except (EOFError, KeyboardInterrupt):
                print()
                name = f"shared:{pid}"
            ensure_quota_domain(db, name, provider=pid, source="bulk grouping")
            for c in creds:
                c["quota_domain"] = name
            print(f"  [+] Got it — all {len(creds)} keys share the '{name}' speed limit.")
        elif ans == "m":
            print("  [i] No problem — type 'quota' later to sort them out.")
        else:
            # different projects: each key can carry its own project id
            print("  [i] OK — each key counts separately.")
            for c in creds:
                try:
                    proj = input(f"  Project ID for key {_mask_secret(c.get('secret'))} "
                                 "(empty = leave separate): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    proj = ""
                if proj:
                    qd = f"project:{pid}:{proj}"
                    c["project_id"] = proj
                    c["quota_domain"] = qd
                    ensure_quota_domain(db, qd, provider=pid, source="per-key project")
        if pid in db:
            # persist assignments onto the real entry
            by_secret = {str(c.get("secret")): c.get("quota_domain") for c in creds}
            proj_secret = {str(c.get("secret")): c.get("project_id") for c in creds
                           if c.get("project_id")}
            for c in db[pid].get("credentials", []) or []:
                if str(c.get("secret")) in by_secret and by_secret[str(c.get("secret"))]:
                    c["quota_domain"] = by_secret[str(c.get("secret"))]
                if str(c.get("secret")) in proj_secret:
                    c["project_id"] = proj_secret[str(c.get("secret"))]
            # the question was answered: never nag about this provider again
            if isinstance(db[pid], dict):
                db[pid]["quota_reviewed"] = True


# ---------- role management ----------

ROLE_PRESETS = {
    "fast": {"pools": [], "fallback": [], "requires": {}},
    "smart": {"pools": [], "fallback": [], "requires": {}},
    "reasoning": {"pools": [], "fallback": [], "requires": {"reasoning": True}},
    "coder": {"pools": [], "fallback": [], "requires": {"tools": True}},
    "vision": {"pools": [], "fallback": [], "requires": {"vision": True}},
    "cheap": {"pools": [], "fallback": [], "requires": {}},
    "fallback": {"pools": [], "fallback": [], "requires": {}},
    "long": {"pools": [], "fallback": [], "requires": {}},
}

# Free-first roles: ordinary work goes to Flash-class (cheap, fast),
# stronger models stay reserved for explicit "smart" calls. Suggested
# only when the needed pools actually exist.
FREE_FAST_TIERS = ("flash", "lite")


def suggest_free_first_roles(db):
    """Suggest google-free-fast / google-free-smart when gemini pools exist.

    Returns dict role -> {pools, note}. fast = flash/lite-tier pools
    (sorted: cheaper & faster first); smart = pro/reasoning-tier pools.
    Suggested only when missing and at least one matching pool exists.
    Never mutates the DB.
    """
    try:
        _deps, pools, _roles, _errors = compile_config(db)
    except Exception:
        return {}
    gemini_pools = []
    for pool, members in pools.items():
        if any(str(d.get("provider") or "") == "gemini" for d in members):
            gemini_pools.append(pool)
    if not gemini_pools:
        return {}
    caps = {}
    for pool, members in pools.items():
        tiers = {str((d.get("capabilities") or {}).get("tier") or "unknown")
                 for d in members if d.get("provider") == "gemini"}
        caps[pool] = tiers
    fast = sorted(p for p in gemini_pools if caps[p] & set(FREE_FAST_TIERS))
    smart = sorted(p for p in gemini_pools if caps[p] & {"pro", "reasoning"})
    roles = db.get(ROLES_KEY) or {}
    out = {}
    if fast and "google-free-fast" not in roles:
        out["google-free-fast"] = {"pools": fast,
                                   "note": "flash-class first (free quota first)"}
    if smart and "google-free-smart" not in roles:
        out["google-free-smart"] = {"pools": smart,
                                    "note": "pro/reasoning for heavy lifts"}
    return out


def apply_free_first_roles(db):
    """Create suggested free-first roles that don't collide with pools.

    Returns (applied, skipped) role names. Only google-free-* names are
    ever written here — user roles are never touched.
    """
    suggestions = suggest_free_first_roles(db)
    roles = db.setdefault(ROLES_KEY, {})
    pools_ok = set()
    try:
        _deps, pools, _r, _errors = compile_config(db)
        pools_ok = set(pools)
    except Exception:
        return [], []
    applied, skipped = [], []
    for role, spec in suggestions.items():
        primaries = [p for p in spec["pools"] if p in pools_ok]
        if not primaries or role in pools_ok:
            skipped.append(role)
            continue
        roles[role] = {"pools": primaries, "fallback": [], "requires": {}}
        applied.append(role)
    return applied, skipped


def manage_roles(db):
    """Interactive role manager: roles point at logical pools with ordered fallback."""
    while True:
        roles = db.get(ROLES_KEY) or {}
        try:
            _, pools, _, _ = compile_config(db)
        except Exception:
            pools = {}
        print("\n--- Roles (app-level names -> model pools) ---")
        if not roles:
            print("  (none yet — e.g. 'fast' -> gemini-3.7-flash, glm-5.3-flash)")
        for role in sorted(roles):
            spec = roles[role] if isinstance(roles[role], dict) else {}
            print(f"  {role:<14} -> {', '.join(spec.get('pools') or []) or '(empty)'}"
                  + (f"  (fallback: {', '.join(spec.get('fallback') or [])})" if spec.get("fallback") else ""))
        print(f"\n  Live pools: {', '.join(sorted(pools)) or '(none)'}")
        free = suggest_free_first_roles(db)
        if free:
            for role, spec in free.items():
                print(f"  [?] '{role}' -> {', '.join(spec['pools'])} ({spec['note']})")
        print("  [A]dd/set  [R]emove  [F]ree-first roles"
              + ("  [Q] back" if free else "  [Q] back"))
        try:
            ch = input("Role choice: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if ch in ("q", "quit", "back", "done", "exit", ""):
            break
        if ch in ("f", "free", "free-first"):
            applied, skipped = apply_free_first_roles(db)
            if applied:
                save_db(db)
                print(f"  [+] Added {', '.join(applied)}.")
                try:
                    n = generate_yaml(db)
                    print(f"  [+] Routes: {n}")
                except ValueError as e:
                    print(f"  [!] {e}")
            else:
                print("  [i] Nothing to add (roles exist or no matching pools).")
            continue
        if ch in ("r", "remove"):
            try:
                name = input("  Role to remove: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if name in roles:
                del roles[name]
                save_db(db)
                print(f"  [-] Role '{name}' removed.")
            continue
        if ch in ("a", "add", "set"):
            try:
                name = input("  Role name (e.g. fast; empty cancels): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if not name:
                continue
            if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]*$", name):
                print("  [!] Role must start with alnum, chars a-z 0-9 . _ : -")
                continue
            print(f"  Pools: {', '.join(sorted(pools)) or '(none)'}")
            try:
                prim = input("  Primary pools, comma/space separated (ordered): ").strip()
                fb = input("  Fallback pools, comma/space separated (empty=none): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            primaries = [p for p in prim.replace(",", " ").split() if p in pools]
            bad = [p for p in prim.replace(",", " ").split() if p and p not in pools]
            if bad:
                print(f"  [!] Unknown pools ignored: {' '.join(bad)}")
            fallbacks = [p for p in fb.replace(",", " ").split() if p in pools and p not in primaries]
            preset = ROLE_PRESETS.get(name, {"requires": {}})
            db.setdefault(ROLES_KEY, {})[name] = {"pools": primaries, "fallback": fallbacks,
                                                  "requires": dict(preset.get("requires", {}))}
            save_db(db)
            try:
                n = generate_yaml(db)
                print(f"  [+] Role '{name}' -> {', '.join(primaries) or '(empty)'}. Routes: {n}")
            except ValueError as e:
                print(f"  [!] {e}")
            continue
        print("  Unknown. Use A/R/Q.")


# ---------- diagnostics / plan ----------

def diagnose(db):
    """Actionable diagnostics: schema, aliases, quota, YAML, service, sync."""
    print("\n=== Diagnostics ===")
    ver = db.get("_schema_version")
    print(f"  [{'OK' if ver == SCHEMA_VERSION else '!'}] DB schema v{ver} (current v{SCHEMA_VERSION})")
    issues = validate_db(db)
    for line in issues[:10]:
        print(f"  [!] {line}")
    if not issues:
        print("  [OK] DB structure sound.")
    cleaned, emptied = normalize_aliases(dict(json.loads(json.dumps(db))))
    if cleaned:
        print(f"  [!] {cleaned} stale alias member(s) would be cleaned" +
              (f" (empties: {', '.join(emptied)})" if emptied else ""))
    else:
        print("  [OK] No stale alias members.")
    # quota sanity: many creds, one domain
    for pid, pdata in db.items():
        if pid.startswith("_") or not isinstance(pdata, dict):
            continue
        creds = list(iter_credentials(pdata))
        doms = {c.get("quota_domain") for c in creds if c.get("quota_domain")}
        if len(creds) >= 4 and len(doms) == 1:
            print(f"  [i] {len(creds)} {pid} credentials share 1 quota domain — "
                  f"correct, but they are NOT {len(creds)} independent pools.")
    try:
        deps, pools, roles, errors = compile_config(db)
        if errors:
            for e in errors[:6]:
                print(f"  [!] {e}")
        else:
            unk = sum(1 for d in deps if not isinstance(d.get("rpm"), (int, float)))
            print(f"  [OK] {len(deps)} deployments / {len(pools)} pools / {len(roles)} roles compile.")
            if unk:
                print(f"  [i] {unk} deployment(s) have unknown RPM (conservative: no limit emitted).")
    except Exception as e:
        print(f"  [!] Compile failed: {e}")
    try:
        with open(YAML_FILE) as f:
            cfg = yaml.safe_load(f) or {}
        print(f"  [OK] config.yaml parses ({len(cfg.get('model_list', []))} routes).")
        rs = cfg.get("router_settings", {})
        bad = [k for k in rs.get("retry_policy", {}) if k not in RETRY_POLICY]
        if bad:
            print(f"  [!] Unsupported retry keys in YAML: {bad}")
        if rs.get("routing_strategy") != ROUTING_STRATEGY:
            print(f"  [!] routing_strategy is {rs.get('routing_strategy')!r}, expected {ROUTING_STRATEGY!r}")
    except Exception as e:
        print(f"  [!] config.yaml issue: {e}")
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", "litellm"],
                           capture_output=True, text=True, timeout=10)
        print(f"  [{'OK' if r.stdout.strip() == 'active' else '!'}] LiteLLM service: {r.stdout.strip() or 'unknown'}")
    except Exception as e:
        print(f"  [i] Service state unknown: {str(e)[:80]}")
    if os.path.exists(SECRET_FILE):
        try:
            mode = oct(os.stat(SECRET_FILE).st_mode & 0o777)
            print(f"  [{'OK' if mode == '0o600' else '!'}] Secret file perms {mode}.")
        except OSError as e:
            print(f"  [!] Secret file: {e}")
    else:
        print("  [i] No secret file yet (created on next run).")
    print("  === End diagnostics ===")


def show_plan(db):
    """Dry-run: show pending changes without modifying anything."""
    try:
        deployments, pools, roles, errors = compile_config(db)
    except Exception as e:
        print(f"  [!] Compile failed: {e}")
        return
    if errors:
        for e in errors[:8]:
            print(f"  [!] {e}")
        return
    print(f"\n  [plan] {len(deployments)} deployments / {len(pools)} pools / {len(roles)} roles")
    for pool in sorted(pools):
        cap = estimate_capacity(db, [(d["provider"], d["upstream_model"]) for d in pools[pool]])
        rpm = f"{cap['rpm_known']} RPM (est.)" if cap["rpm_known"] else "RPM unknown"
        print(f"    {pool}: {len(pools[pool])} deployments, {cap['domains']} domain(s), {rpm}")
    if roles:
        print("  roles:")
        for r, spec in sorted(roles.items()):
            print(f"    {r} -> {', '.join(spec['pools'])}" +
                  (f" (fallback {', '.join(spec['fallback'])})" if spec.get("fallback") else ""))
    print("  (dry run — nothing written)")


def _strip_jsonc_wiz(text):
    """Remove // and /* */ comments (outside strings) + trailing commas."""
    out, i, n, in_str, esc = [], 0, len(text), False, False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            esc = (c == "\\" and not esc)
            if c == '"' and not esc:
                in_str = False
            elif c != "\\":
                esc = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def _gateway_aliases():
    """Unique user-facing aliases from the generated config.

    Includes model_list entries plus role aliases (model_group_alias),
    never raw deployment details. Regex-based, no yaml needed.
    """
    try:
        with open(YAML_FILE) as f:
            text = f.read()
    except OSError:
        return []
    seen, aliases = set(), []
    in_alias_block = False
    for line in text.splitlines():
        if re.match(r"^\s*model_group_alias:\s*$", line):
            in_alias_block = True
            continue
        if in_alias_block:
            m = re.match(r"^\s{2,}(\S+):\s*(\S.*)?$", line)
            if m and not line.strip().startswith("-"):
                name = m.group(1).rstrip(":")
                if name and name not in seen and name not in ("model", "hidden"):
                    seen.add(name)
                    aliases.append(name)
                continue
            if re.match(r"^[a-z_]+:\s*$", line):
                in_alias_block = False
        m = re.match(r"^\s*(?:-\s*)?model_name:\s*(\S+)\s*$", line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            aliases.append(m.group(1))
    return aliases


def _opencode_litellm_models(path):
    """Model keys of the litellm block, or None if missing/unparseable."""
    try:
        with open(path) as f:
            cfg = json.loads(_strip_jsonc_wiz(f.read()))
        return list((cfg.get("provider", {}) or {}).get("litellm", {}).get("models", {}).keys())
    except (OSError, ValueError):
        return None


def maybe_sync_opencode():
    """Q-time offer: sync gateway aliases into opencode.json. Never raises."""
    import datetime
    path = os.environ.get("OPENCODE_JSON", os.path.join(os.path.expanduser("~"), ".config", "opencode", "opencode.json"))
    try:
        if not os.path.exists(path):
            print(f"  [i] No OpenCode config at {path} — skipping sync.")
            return
        aliases = _gateway_aliases()
        if not aliases:
            print("  [!] No gateway aliases found — skipping sync.")
            return
        current = _opencode_litellm_models(path)
        if current is None:
            print(f"  [!] Could not parse {path} — leaving it untouched.")
            return
        if set(current) == set(aliases):
            print(f"  [=] opencode.json already in sync ({len(aliases)} models).")
            return
        try:
            ans = input(f"  opencode.json lists {len(current)}, gateway serves {len(aliases)} — sync to opencode.json? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  [-] Sync skipped.")
            return
        if ans not in ("", "y", "yes"):
            print("  [-] Left opencode.json unchanged (manual: sync-opencode.py).")
            return
        with open(path) as f:
            cfg = json.loads(_strip_jsonc_wiz(f.read()))
        # secret guard: managed block carries alias names only, env placeholder
        block_models = {a: {"name": a} for a in aliases}
        cfg.setdefault("provider", {})["litellm"] = {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Local LiteLLM",
            "options": {"baseURL": "http://localhost:4000/v1",
                        "apiKey": "{env:LITELLM_MASTER_KEY}"},
            "models": block_models,
        }
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = f"{path}.bak-{stamp}"
        import shutil
        shutil.copy2(path, backup)
        _atomic_write_json(path, cfg)
        try:
            with open(path) as f:
                json.load(f)  # strict sanity check
        except ValueError:
            shutil.copy2(backup, path)
            print(f"  [!] Wrote invalid JSON — restored backup {backup}.")
            return
        print(f"  [+] Synced {len(aliases)} models (backup: {backup}). Restart the OpenCode TUI, then /models.")
    except Exception as e:
        print(f"  [!] Sync failed safely ({str(e)[:120]}). Gateway config is unaffected.")


# ---------- jcode target (export/import; JCode stays an output target) ---

JCODE_CONFIG_PATH = os.environ.get(
    "JCODE_CONFIG", os.path.join(os.path.expanduser("~"), ".jcode", "config.toml"))


def _load_jcode_sync():
    """Import sync-jcode.py as a module (single implementation, CLI+engine)."""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "sync-jcode.py"), "sync-jcode.py"):
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("sync_jcode", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("sync-jcode.py not found next to wizard.py")


def sync_jcode_now(db, dry_run=False):
    """CLI: sync gateway logical models into JCode's config.toml."""
    try:
        syncj = _load_jcode_sync()
    except FileNotFoundError as e:
        print(f"  [!] {e}")
        return False
    path = JCODE_CONFIG_PATH
    print(f"\n  JCode target: {path}")
    try:
        result = syncj.sync(path, YAML_FILE, dry_run=dry_run, print_block=False,
                            include_roles=True, set_default=True,
                            overwrite_externally_changed=False)
    except ValueError as e:
        print(f"  [!] {e}")
        if "managed_hash mismatch" in str(e):
            print("  [i] Options: import it ('jcode import') to adopt the "
                  "changes, or run:  python3 sync-jcode.py --overwrite-external")
        return False
    except FileNotFoundError as e:
        print(f"  [!] Nothing to sync yet — {str(e)[:100]}")
        print("  [i] Configure providers in the wizard and press Q to compile first.")
        return False
    except OSError as e:
        print(f"  [!] Sync failed safely ({str(e)[:120]}).")
        return False
    exposed = result["models"]
    print(f"  [+] {len(exposed)} logical model(s): {', '.join(exposed[:8])}"
          + (" ..." if len(exposed) > 8 else ""))
    if not result.get("wrote"):
        print("  [=] Dry run — nothing written.")
        return True
    if result.get("backup"):
        print(f"  [+] Backup: {result['backup']}")
    chg = []
    if result.get("added"):
        chg.append(f"+{len(result['added'])} ({', '.join(result['added'][:4])})")
    if result.get("removed"):
        chg.append(f"-{len(result['removed'])} ({', '.join(result['removed'][:4])})")
    print("  [+] " + ("; ".join(chg) if chg else "no model changes")
          + " — unrelated JCode settings untouched.")
    print("  [i] Try it: jcode --provider-profile llm-proxy-wizard")
    return True


def maybe_sync_jcode():
    """Q-time offer: sync the gateway block into JCode. Never raises."""
    path = JCODE_CONFIG_PATH
    if not os.path.exists(path):
        # JCode not installed: nothing to offer, stay quiet
        return
    try:
        syncj = _load_jcode_sync()
        with open(path) as f:
            profiles = syncj.parse_profiles(f.read())
        prof = profiles.get(syncj.MANAGED_PROFILE)
        aliases = _gateway_aliases()
        if not aliases:
            return
        if prof and [m.get("id") for m in prof.get("models", [])] == list(aliases):
            return  # already in sync — stay quiet
    except Exception:  # noqa: BLE001 -- the Q-time offer must never break quitting
        return
    try:
        ans = input("  Also sync these models to JCode (~/.jcode/config.toml)? [Y/n]: "
                    ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  [-] JCode sync skipped.")
        return
    if ans not in ("", "y", "yes"):
        print("  [-] Left JCode unchanged (manual: sync-jcode.py).")
        return
    db = load_db()
    sync_jcode_now(db)


def import_opencode_now(db):
    """CLI: import provider/model config from opencode.json into the DB."""
    path = os.environ.get(
        "OPENCODE_JSON",
        os.path.join(os.path.expanduser("~"), ".config", "opencode", "opencode.json"))
    print(f"\n--- Import from OpenCode ({path}) ---")
    if not os.path.exists(path):
        print("  [!] No OpenCode config found — nothing to import.")
        return
    # reuse the engine facade (same logic the TUI uses)
    try:
        import engine as _engine
        result = _engine.import_opencode(db)
    except Exception as e:  # noqa: BLE001 -- report, don't crash the shell
        print(f"  [!] Import failed: {str(e)[:140]}")
        return
    if not result.get("ok"):
        print(f"  [!] {result.get('note', 'import failed')}")
        return
    if result.get("gateway_models"):
        print(f"  [=] OpenCode already uses the wizard's LiteLLM gateway "
              f"({len(result['gateway_models'])} logical model(s) — nothing to import).")
    imported = result.get("providers", [])
    if not imported:
        print("  [i] No direct (non-gateway) providers with models found.")
        return
    for r in imported:
        tag = "new" if r.get("created") else "matched"
        print(f"  [+] {r.get('name', '?')}: {tag} endpoint {r.get('base_url')} "
              f"(+{len(r.get('new_models', []))} model(s), "
              f"{r.get('total_models', 0)} total)")
    save_db(db)
    print("  [+] Imported. Secrets (if any) are stored as credentials — never printed.")
    print("  [i] Validate keys from the provider screen, then Q to compile.")


def import_jcode_now(db):
    """CLI: import provider profiles from JCode's config.toml into the DB."""
    path = JCODE_CONFIG_PATH
    print(f"\n--- Import from JCode ({path}) ---")
    if not os.path.exists(path):
        print("  [!] No JCode config found — nothing to import.")
        return
    try:
        import engine as _engine
        result = _engine.import_jcode(db)
    except Exception as e:  # noqa: BLE001 -- report, don't crash the shell
        print(f"  [!] Import failed: {str(e)[:140]}")
        return
    if not result.get("ok"):
        print(f"  [!] {result.get('note', 'import failed')}")
        return
    if result.get("managed_models"):
        print(f"  [=] JCode already uses the wizard's gateway profile "
              f"({len(result['managed_models'])} logical model(s) — not re-imported).")
    imported = result.get("providers", [])
    if not imported:
        print("  [i] No custom provider profiles with models found.")
        return
    for r in imported:
        tag = "new" if r.get("created") else "matched"
        print(f"  [+] {r.get('name', '?')}: {tag} endpoint {r.get('base_url')} "
              f"(+{len(r.get('new_models', []))} model(s), "
              f"{r.get('total_models', 0)} total)")
    save_db(db)
    print("  [+] Imported. Validate keys from the provider screen, then Q to compile.")


def _resolve_pid(db, text):
    num, res = resolve_provider(text, db)
    if num == "C" and isinstance(res, dict):
        return res["id"]
    if num is not None and num in PROVIDERS:
        return PROVIDERS[num]["id"]
    if text in db and isinstance(db.get(text), dict):
        return text
    return None


def _set_disabled(db, text, disabled):
    pid = _resolve_pid(db, text)
    if not pid or not isinstance(db.get(pid), dict):
        print(f"  [!] No provider '{text}'")
        return
    db[pid]["disabled"] = disabled
    save_db(db)
    verb = "Disabled" if disabled else "Enabled"
    print(f"  [{'+' if not disabled else '-'}] {verb} provider {pid} (config kept, excluded from YAML).")
    safe_generate_yaml(db)


def _quarantine_credential(db, text, quar):
    parts = text.split()
    pid = _resolve_pid(db, parts[0]) if parts else None
    if not pid:
        print(f"  [!] Usage: {'quarantine' if quar else 'unquarantine'} <provider> <suffix>")
        return
    suffix = parts[1] if len(parts) > 1 else ""
    pdata = db.get(pid, {})
    hits = [c for c in iter_credentials(pdata)
            if not suffix or str(c.get("secret", "")).endswith(suffix)]
    if not hits:
        print("  [!] No matching credential.")
        return
    for c in hits:
        c["quarantined"] = quar
    save_db(db)
    print(f"  [{'!' if quar else '+'}] {'Quarantined' if quar else 'Released'} {len(hits)} credential(s) on {pid}.")
    safe_generate_yaml(db)


def _remove_provider(db, text):
    pid = _resolve_pid(db, text)
    if not pid or pid not in db:
        print(f"  [!] No provider '{text}'")
        return
    try:
        ans = input(f"  Remove provider '{pid}' entirely (keys+models kept out of YAML, DB entry deleted)? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if ans in ("y", "yes"):
        del db[pid]
        # drop alias members referencing it (visible, via normalizer)
        cleaned, _ = normalize_aliases(db)
        save_db(db)
        safe_generate_yaml(db)
        print(f"  [-] Removed '{pid}' (cleaned {cleaned} stale alias member(s)).")


def _choose_validation_mode(db):
    print("  Validation modes: FAST (1 cred/model, cheapest) / STRICT (every cred x model) / SAMPLE (N creds/model).")
    try:
        m = input(f"  Mode [{db.get(SETTINGS_KEY, {}).get('validation_mode', 'FAST')}]: ").strip().upper()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if m in VALIDATION_MODES:
        db.setdefault(SETTINGS_KEY, {})["validation_mode"] = m
        if m == "SAMPLE":
            try:
                n = input(f"  Sample size [{db[SETTINGS_KEY].get('sample_size', 2)}]: ").strip()
                if n:
                    db[SETTINGS_KEY]["sample_size"] = max(1, int(n))
            except (EOFError, KeyboardInterrupt, ValueError):
                print()
        save_db(db)
        print(f"  [+] Validation mode: {m}")
    elif m:
        print("  [!] Unknown mode.")


def _yaml_hash():
    try:
        with open(YAML_FILE, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return ""


def show_help():
    print("""
  What this program does: it collects your AI keys and model choices, then
  writes the gateway's settings file (config.yaml) and restarts the gateway.
  You talk to the gateway at http://localhost:4000; it forwards to providers.

  Everyday commands:
    1-10        change a provider (keys, models)
    add <name>  add a provider, e.g. add google   (bare 'add' lists missing ones)
    alias       use one name for the same model from several places
    T           try every model name through the gateway (1 quick call each)
    Q           save everything, restart the gateway, quit

  Less often:
    P           try every single key+model connection directly (thorough, slower)
    F           full check: settings, connections, gateway, everything
    quota       speed limits: group keys that share one provider-side limit
    role        shortcuts: one word like 'fast' that tries models in order
    plan        preview what Q would change, without changing anything
    diagnose    find problems in your setup and how to fix them
    mode        how thoroughly new models are tested (FAST is fine usually)
    all         show the detailed view / back to the short view
    help        this text

  Clients (OpenCode / JCode):
    jcode           sync logical models into ~/.jcode/config.toml
    jcode import    adopt an existing JCode provider config into the wizard
    import          adopt an existing OpenCode provider config into the wizard
    opencode        sync logical models into opencode.json
    # vault is source of truth — no direct opencode→jcode; harnesses import from vault

  Words you'll see:
    key       one API key you pasted in.
    quota     the provider's speed limit bucket. Several keys can share one
              (e.g. all Google keys from the same Google project). Sharing
              keys does NOT multiply your speed — 'quota' shows the truth.
    combined model  one name served from several places, so a slow/busy one
              doesn't stop you. Made with 'alias'.
    shortcut  a nickname for a list of models in order (made with 'role').
""")


def main():
    if "--version" in sys.argv or "-v" in sys.argv:
        print(__version__)
        return
    if "--diagnose" in sys.argv or "diagnose" in sys.argv[1:]:
        diagnose(load_db())
        return
    # explicit sync/import targets (non-interactive one-shots)
    if "--sync-jcode" in sys.argv:
        db = load_db()
        ok = sync_jcode_now(db, dry_run="--dry-run" in sys.argv)
        sys.exit(0 if ok else 1)
    if "--sync-opencode" in sys.argv:
        maybe_sync_opencode()
        sys.exit(0)
    if "--import-opencode" in sys.argv:
        db = load_db()
        import_opencode_now(db)
        sys.exit(0)
    if "--import-jcode" in sys.argv:
        db = load_db()
        import_jcode_now(db)
        sys.exit(0)
    db = load_db()
    cleaned, emptied = normalize_aliases(db)
    if cleaned:
        print(f"Dropped {cleaned} outdated entr{'y' if cleaned == 1 else 'ies'} pointing at "
              "models you removed" + (f" ({', '.join(emptied)} now empty)" if emptied else "") + ".")
        save_db(db)
    print(f"=== LLM Proxy Wizard v{__version__} ===")
    show_all = False
    while True:
        print_status(db, verbose=show_all)
        print("\n[number] change provider | add <name> | alias (combine models) | T (try all) | Q (save & quit)")
        print("Type help for everything else (quota, roles, tests, checks).")
        try:
            ch = input("Choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting (no final restart). Your per-provider settings are already saved.")
            break
        low = ch.lower()
        if low in ("help", "h", "?"):
            show_help()
            continue
        if low in ("q", "0", "done", "quit", "exit"):
            save_db(db)
            before = _yaml_hash()
            before_aliases = set(_gateway_aliases())
            try:
                n = generate_yaml(db)
            except ValueError as e:
                print(f"[!] {e}")
                print("[i] Fix with 'diagnose', then Q again. Config.yaml left untouched.")
                continue
            print(f"[+] Saved. The gateway now serves {n} model connections.")
            after_aliases = set(_gateway_aliases())
            if before and before == _yaml_hash():
                print("[=] Nothing actually changed — no restart needed.")
            else:
                if before_aliases != after_aliases:
                    maybe_sync_opencode()
                restart_proxy()
            if before_aliases != after_aliases:
                maybe_sync_jcode()
            print("[+] Bye. (Tip: T tries everything through the gateway.)")
            break
        if low == "t":
            proxy_smoke_test()
            continue
        if low == "p":
            pool_health_test(db)
            continue
        if low == "f":
            full_sweep(db)
            db = load_db()
            continue
        if low in ("all", "list", "ls"):
            show_all = not show_all
            continue
        if low in ("quota", "quotas", "qd"):
            manage_quota(db)
            db = load_db()
            continue
        if low in ("role", "roles"):
            manage_roles(db)
            db = load_db()
            continue
        if low in ("diagnose", "diag", "doctor"):
            diagnose(db)
            continue
        if low in ("jcode",):
            sync_jcode_now(db)
            db = load_db()
            continue
        if low in ("sync jcode", "jcode sync"):
            sync_jcode_now(db)
            db = load_db()
            continue
        if low in ("jcode import", "import jcode"):
            import_jcode_now(db)
            db = load_db()
            continue
        if low in ("import opencode", "opencode import", "import"):
            import_opencode_now(db)
            db = load_db()
            continue
        if low in ("sync opencode", "opencode sync", "opencode"):
            maybe_sync_opencode()
            continue
        if low in ("plan", "dry-run", "dryrun", "preview", "diff"):
            show_plan(db)
            continue
        if low in ("pools", "pool", "deployments"):
            show_plan(db)
            continue
        if low in ("health",):
            pool_health_test(db)
            continue
        if low in ("status", "st"):
            print_status(db, verbose=True)
            continue
        if low.startswith("disable "):
            _set_disabled(db, low[8:].strip(), True)
            db = load_db()
            continue
        if low.startswith("enable "):
            _set_disabled(db, low[7:].strip(), False)
            db = load_db()
            continue
        if low.startswith("quarantine "):
            _quarantine_credential(db, low[11:].strip(), True)
            db = load_db()
            continue
        if low.startswith("unquarantine "):
            _quarantine_credential(db, low[13:].strip(), False)
            db = load_db()
            continue
        if low.startswith("remove ") or low.startswith("rm "):
            _remove_provider(db, (low.split(None, 1) + [""])[1])
            db = load_db()
            continue
        if low in ("mode", "validation", "validation-mode"):
            _choose_validation_mode(db)
            db = load_db()
            continue
        if low in ("alias", "aliases", "group", "groups", "unify", "unified"):
            manage_aliases(db)
            db = load_db()
            continue
        if low.startswith("alias ") or low.startswith("group ") or low.startswith("unify "):
            # shortcut: alias deepseek-v4-flash  -> jump straight into add flow
            manage_aliases(db)
            db = load_db()
            continue
        dyn = None  # dynamic custom-endpoint dict when resolved/created below
        if low == "add":
            print_unconfigured(db)
            try:
                name = input("Provider name/number to add: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if name in PROVIDERS and name != "10":
                ch = name
            elif name == "10":
                dyn = create_custom_provider(db)
                if dyn is None:
                    continue
            else:
                num, res = resolve_provider(name, db)
                if num is None:
                    if res:
                        print("  Ambiguous — did you mean:")
                        for n in res:
                            print(f"    [{n}] {PROVIDERS[n]['name']}")
                        continue
                    try:
                        ans = input(f"  No provider matches '{name}'. Create a custom endpoint named '{name}'? [y/N]: ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        continue
                    if ans in ("y", "yes"):
                        dyn = _named_custom_provider(db, name)
                        if dyn is None:
                            continue
                        try:
                            configure_provider(db, dyn)
                            db = load_db()  # re-read (configure saves)
                        except (KeyboardInterrupt, EOFError):
                            print("\n[-] Provider step cancelled.")
                            db = load_db()
                    continue
                if num == "C":
                    dyn = res
                elif num == "10":
                    dyn = create_custom_provider(db)
                    if dyn is None:
                        continue
                else:
                    ch = num
        elif low.startswith("add "):
            num, res = resolve_provider(ch[4:], db)
            if num is None:
                if res:
                    print("  Ambiguous — did you mean:")
                    for n in res:
                        print(f"    [{n}] {PROVIDERS[n]['name']}")
                    continue
                try:
                    ans = input(f"  No provider matches '{ch[4:]}'. Create a custom endpoint named '{ch[4:]}'? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    continue
                if ans in ("y", "yes"):
                    dyn = _named_custom_provider(db, ch[4:])
                    if dyn is None:
                        continue
                    try:
                        configure_provider(db, dyn)
                        db = load_db()  # re-read (configure saves)
                    except (KeyboardInterrupt, EOFError):
                        print("\n[-] Provider step cancelled.")
                        db = load_db()
                continue
            if num == "C":
                dyn = res
            elif num == "10":
                dyn = create_custom_provider(db)
                if dyn is None:
                    continue
            else:
                ch = num
        elif ch.strip() == "10":
            dyn = create_custom_provider(db)
            if dyn is None:
                continue
        if dyn is not None:
            try:
                configure_provider(db, dyn)
                db = load_db()  # re-read (configure saves)
            except (KeyboardInterrupt, EOFError):
                print("\n[-] Provider step cancelled.")
                db = load_db()
            continue
        if ch not in PROVIDERS:
            print("Not sure what you mean. Try a number, add <name>, alias, T, Q — or help.")
            continue
        try:
            configure_provider(db, PROVIDERS[ch])
            db = load_db()  # re-read (configure saves)
        except (KeyboardInterrupt, EOFError):
            print("\n[-] Provider step cancelled.")
            db = load_db()


if __name__ == "__main__":
    main()
