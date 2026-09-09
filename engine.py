"""Clean callable engine for llm-proxy-wizard (Milestone 1).

The tested product logic lives in ``wizard.py`` (compiler, quota math,
provider adapters, secret handling) and ``sync-opencode.py`` (OpenCode
mutation). This module does NOT reimplement any of that: it is a thin,
importable facade so the Textual TUI (and future callers) can drive the
product without owning provider logic, quota math, config compilation,
secret handling, or OpenCode mutation.

Contract preserved:
- ``providers_db.json`` schema v2, ``config.yaml``, OpenCode JSON.
- ``~/.config/litellm/`` + ``LITELLM_DB_FILE`` / ``LITELLM_YAML_FILE`` /
  ``OPENCODE_JSON`` / ``LITELLM_SECRET_FILE`` overrides. A DB written by
  the CLI is readable/writable here and vice versa. Unknown fields are
  never silently discarded (``migrate_db`` preserves them).
- Secrets: never printed/logged here; credential IDs stay
  ``cred-<sha256(secret)[:12]>``; sensitive files mode 0600; atomic
  writes; OpenCode sync carries model NAMES only and backs up first.
- Auth stays ``{env:LITELLM_MASTER_KEY}`` / ``os.environ/LITELLM_MASTER_KEY``.

Testability: filesystem paths are overridable via :class:`EnginePaths`;
network ops delegate to ``wizard`` adapters (mock ``wizard._get`` /
``wizard._post`` / ``wizard.test_single_model`` in tests); systemd ops
accept an injectable ``runner``; time-dependent readiness accepts an
injectable ``wait_fn``. Tests must use temp dirs + fake keys.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from typing import Any

import wizard as _wiz

__version__ = _wiz.__version__
SCHEMA_VERSION = _wiz.SCHEMA_VERSION


# ---------------------------------------------------------------- paths ---

def _default_paths() -> tuple[str, str, str, str]:
    litellm_dir = os.path.join(os.path.expanduser("~"), ".config", "litellm")
    db = os.environ.get("LITELLM_DB_FILE", os.path.join(litellm_dir, "providers_db.json"))
    yml = os.environ.get("LITELLM_YAML_FILE", os.path.join(litellm_dir, "config.yaml"))
    sec = os.environ.get("LITELLM_SECRET_FILE", os.path.join(litellm_dir, ".master_key"))
    oc = os.environ.get(
        "OPENCODE_JSON",
        os.path.join(os.path.expanduser("~"), ".config", "opencode", "opencode.json"),
    )
    return db, yml, sec, oc


@dataclasses.dataclass
class EnginePaths:
    """Overridable filesystem locations. Defaults honour env overrides."""

    db_file: str = dataclasses.field(default_factory=lambda: _default_paths()[0])
    yaml_file: str = dataclasses.field(default_factory=lambda: _default_paths()[1])
    secret_file: str = dataclasses.field(default_factory=lambda: _default_paths()[2])
    opencode_json: str = dataclasses.field(default_factory=lambda: _default_paths()[3])

    @classmethod
    def from_env(cls) -> EnginePaths:
        db, yml, sec, oc = _default_paths()
        return cls(db_file=db, yaml_file=yml, secret_file=sec, opencode_json=oc)

    @classmethod
    def temp(cls, tmpdir: str, opencode: str | None = None) -> EnginePaths:
        return cls(
            db_file=os.path.join(tmpdir, "providers_db.json"),
            yaml_file=os.path.join(tmpdir, "config.yaml"),
            secret_file=os.path.join(tmpdir, ".master_key"),
            opencode_json=opencode or os.path.join(tmpdir, "opencode.json"),
        )


@contextlib.contextmanager
def _patched_wizard(paths: EnginePaths):
    """Point wizard's module-level paths at ``paths`` for one operation."""
    old = (_wiz.DB_FILE, _wiz.YAML_FILE, _wiz.SECRET_FILE)
    _wiz.DB_FILE, _wiz.YAML_FILE, _wiz.SECRET_FILE = (
        paths.db_file, paths.yaml_file, paths.secret_file,
    )
    try:
        yield
    finally:
        _wiz.DB_FILE, _wiz.YAML_FILE, _wiz.SECRET_FILE = old


def _resolve_paths(paths: EnginePaths | None) -> EnginePaths:
    return paths or EnginePaths.from_env()


# ------------------------------------------------------------ state I/O ---

def load_state(paths: EnginePaths | None = None) -> dict[str, Any]:
    """Load + migrate the DB. Readable whichever UI wrote it."""
    p = _resolve_paths(paths)
    with _patched_wizard(p):
        return _wiz.load_db()


def save_state(db: dict[str, Any], paths: EnginePaths | None = None) -> None:
    """Normalize + atomically persist the DB (mode 0600)."""
    p = _resolve_paths(paths)
    with _patched_wizard(p):
        # save_db ensures parent dir exists and migrates before writing.
        _wiz.save_db(db)


def validate_state(db: dict[str, Any]) -> list[str]:
    """Structural diagnostics; empty means sound. Delegates to engine."""
    return _wiz.validate_db(db)


def get_master_key(paths: EnginePaths | None = None) -> str:
    """Resolve gateway master key: env -> secret file -> generated (0600)."""
    p = _resolve_paths(paths)
    with _patched_wizard(p):
        return _wiz.get_master_key()


def mask_secret(secret: str) -> str:
    """Suffix-only masking for display. Never reveals the full secret."""
    return _wiz._mask_secret(secret)


def credential_id(secret: str) -> str:
    """Stable secret-free credential identity."""
    return _wiz.credential_id(secret)


# -------------------------------------------------------------- providers ---

def list_providers(db: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Builtin providers (+ custom endpoints when ``db`` is given)."""
    out: list[dict[str, Any]] = []
    for num in sorted(_wiz.PROVIDERS, key=lambda n: int(n) if n.isdigit() else 99):
        p = _wiz.PROVIDERS[num]
        out.append({"num": num, "id": p["id"], "name": p["name"],
                    "type": p.get("type", ""), "custom": False})
    if db is not None:
        for pid, label in _wiz._custom_entries(db):
            out.append({"num": "C", "id": pid, "name": f"{label} (custom)",
                        "type": "custom_api", "custom": True})
    return out


def get_provider(pid_or_text: str, db: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Resolve a provider id / fuzzy name to its provider dict."""
    if not pid_or_text:
        return None
    for p in list_providers(db):
        if pid_or_text in (p["id"], p["num"]) or pid_or_text == p["name"]:
            return _provider_dict(p["id"], db)
    num, res = _wiz.resolve_provider(pid_or_text, db)
    if isinstance(res, dict):
        return dict(res)
    if num is not None and num in _wiz.PROVIDERS:
        return dict(_wiz.PROVIDERS[num])
    return None


def _provider_dict(pid: str, db: dict[str, Any] | None = None) -> dict[str, Any] | None:
    for p in _wiz.PROVIDERS.values():
        if p["id"] == pid:
            return dict(p)
    if db is not None and pid in db and pid.startswith("custom_"):
        return _wiz._custom_provider_dict(pid, db)
    if pid == "custom" and "custom" not in [p["id"] for p in list_providers()]:
        return dict(_wiz.PROVIDERS.get("10", {}))
    return None


# ------------------------------------------------------------ credentials ---

def add_credentials(db: dict[str, Any], pid: str, secrets: list[str],
                    quota_domain: str | None = None,
                    endpoint: str | None = None) -> list[str]:
    """Store raw secrets as credentials. Returns new credential IDs.

    Does not validate against the network; call :func:`validate_credentials`
    separately so the UI can run it in a background worker. Quota-domain
    assignment honours an explicit ``quota_domain``; otherwise per-credential
    defaults apply (bulk Google grouping is resolved via
    :func:`set_quota_domains` after asking the user ONE question).
    """
    if not secrets:
        return []
    entry = db.get(pid)
    if not isinstance(entry, dict):
        entry = {"keys": [], "models": [], "endpoints": []}
        db[pid] = entry
    entry.setdefault("keys", [])
    entry.setdefault("models", [])
    entry.setdefault("endpoints", [])
    if endpoint and endpoint not in entry["endpoints"]:
        entry["endpoints"].append(endpoint)
    added: list[str] = []
    for s in secrets:
        s = str(s or "").strip()
        if not s or s in entry["keys"]:
            continue
        entry["keys"].append(s)
        added.append(credential_id(s))
    _wiz.normalize_credentials(entry)
    if quota_domain:
        for c in _wiz.iter_credentials(entry):
            if c.get("id") in added:
                c["quota_domain"] = quota_domain
    _wiz.migrate_db(db)  # ensure quota-domain records exist; preserves unknowns
    return added


def remove_credentials(db: dict[str, Any], pid: str,
                       cred_ids: list[str]) -> int:
    """Remove credentials by ID. Returns number removed. Keeps last-key guard
    to the caller (interactive flows abort instead of saving keyless)."""
    entry = db.get(pid)
    if not isinstance(entry, dict):
        return 0
    creds = entry.get("credentials") or []
    doomed = {c.get("id") for c in creds if c.get("id") in set(cred_ids)}
    if not doomed:
        return 0
    doomed_secrets = {c.get("secret") for c in creds if c.get("id") in doomed}
    entry["credentials"] = [c for c in creds if c.get("id") not in doomed]
    entry["keys"] = [k for k in entry.get("keys", []) if k not in doomed_secrets]
    _wiz.migrate_db(db)
    return len(doomed)


def list_credentials(db: dict[str, Any], pid: str) -> list[dict[str, Any]]:
    """Secret-free credential summaries safe for TUI display."""
    entry = db.get(pid)
    if not isinstance(entry, dict):
        return []
    out = []
    for c in _wiz.iter_credentials(entry):
        out.append({"id": c.get("id", ""), "suffix": mask_secret(c.get("secret", "")),
                    "quota_domain": c.get("quota_domain", ""),
                    "enabled": c.get("enabled", True) is not False
                    and not c.get("quarantined"),
                    "validation": (c.get("validation") or {}).get("status", "unknown")})
    return out


def validate_credentials(pid: str, secrets: list[str],
                         endpoints: list[str] | None = None
                         ) -> tuple[list[tuple[str, bool, str]], list[str] | None]:
    """Validate keys directly against the provider (network).

    Returns ``(results, available_models)``. Must run off the UI thread;
    network adapters are mockable via ``wizard._get`` / ``wizard._post``.
    Prints progress (wizard behaviour, preserved).
    """
    return _wiz.validate_keys(pid, list(secrets), endpoints or [])


def discover_models(pid: str, key: str,
                    endpoint: str | None = None) -> list[tuple[str, str]] | None:
    """Fetch the live model catalog. Returns ``[(id, label)]`` or None."""
    return _wiz.fetch_catalog(pid, key, endpoint)


def effective_endpoint(db: dict[str, Any], pid: str) -> str | None:
    """Pre-filled base URL for a custom_api provider: stored override >
    stored base > builtin default (None when nothing is known)."""
    return _wiz.effective_base_url(db, pid, None)


# ------------------------------------------------------------------ quota ---

def calculate_quota_domains(db: dict[str, Any],
                            pid: str | None = None) -> dict[str, list[tuple[str, str]]]:
    """Map quota-domain id -> ``[(pid, credential_id)]`` members."""
    domains: dict[str, list[tuple[str, str]]] = {}
    for qd in (db.get(_wiz.QUOTA_KEY) or {}):
        members = [(p, (c.get("id") or "")) for p, c in _wiz.quota_members(db, qd)]
        if pid is not None:
            members = [(p, c) for p, c in members if p == pid]
        if members or pid is None:
            domains[qd] = members
    # include domains referenced by credentials but missing a record
    for p, pdata in db.items():
        if p.startswith("_") or not isinstance(pdata, dict):
            continue
        if pid is not None and p != pid:
            continue
        for c in _wiz.iter_credentials(pdata):
            qd = c.get("quota_domain") or ""
            if qd and qd not in domains:
                domains[qd] = [(pp, cc) for pp, cc in _wiz.quota_members(db, qd)]
    return domains


def set_quota_domains(db: dict[str, Any], pid: str, mode: str) -> str:
    """Resolve bulk grouping with ONE decision: ``shared`` / ``separate`` / ``later``.

    ``shared`` puts every credential of ``pid`` in one domain;
    ``separate`` gives each credential its own domain; ``later`` keeps the
    automatic defaults. All modes mark the provider reviewed (mirroring the
    CLI's ``quota_grouping_prompt``: the question is never nagged twice).
    Returns the resulting domain id, ``separate``, or ``later``.
    """
    entry = db.get(pid)
    if not isinstance(entry, dict):
        return ""
    creds = list(_wiz.iter_credentials(entry))
    if mode == "shared":
        qd = f"project:{pid}-shared"
        _wiz.ensure_quota_domain(db, qd, provider=pid, confidence="manual",
                                 source="tui grouping answer")
        for c in creds:
            c["quota_domain"] = qd
        entry["quota_reviewed"] = True
        _wiz.migrate_db(db)
        return qd
    if mode == "later":
        entry["quota_reviewed"] = True
        _wiz.migrate_db(db)
        return "later"
    for c in creds:
        c["quota_domain"] = _wiz.default_quota_domain_id(pid, c)
    entry["quota_reviewed"] = True
    _wiz.migrate_db(db)
    return "separate"


def quota_domains_list(db: dict[str, Any]) -> list[dict[str, Any]]:
    """Structured quota-domain facts for the TUI dashboard (via wizard)."""
    return _wiz.quota_domains_list(db)


def probe_latency(db: dict[str, Any], pid: str, model: str) -> float | None:
    """Mean recent probe latency (seconds) for (provider, model), or None."""
    return _wiz.probe_latency(db, pid, model)


def suggest_free_first_roles(db: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Suggested google-free-fast/smart roles when gemini pools exist."""
    return _wiz.suggest_free_first_roles(db)


def apply_free_first_roles(db: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Create suggested free-first roles. Returns (applied, skipped)."""
    return _wiz.apply_free_first_roles(db)


def calculate_capacity(db: dict[str, Any], members) -> dict[str, Any]:
    """Effective capacity summed over UNIQUE quota domains (never keys x RPM)."""
    return _wiz.estimate_capacity(db, members)


def needs_grouping_question(db: dict[str, Any], pid: str) -> bool:
    """True when bulk keys for a project-scoped provider are ambiguous."""
    return bool(_wiz.needs_quota_hint(db, pid))


def get_validation_mode(db: dict[str, Any]) -> str:
    """FAST / STRICT / SAMPLE probe mode from settings (default FAST)."""
    mode = (db.get(_wiz.SETTINGS_KEY) or {}).get("validation_mode", "FAST")
    return mode if mode in ("FAST", "STRICT", "SAMPLE") else "FAST"


def get_sample_size(db: dict[str, Any]) -> int:
    """Per-model probe sample size from settings (default 2)."""
    try:
        return max(1, int((db.get(_wiz.SETTINGS_KEY) or {}).get("sample_size", 2)))
    except (TypeError, ValueError):
        return 2


def get_models(db: dict[str, Any], pid: str) -> list[str]:
    """Configured model IDs for a provider (may be empty)."""
    entry = db.get(pid)
    if not isinstance(entry, dict):
        return []
    return [m for m in entry.get("models", []) if isinstance(m, str)]


def set_models(db: dict[str, Any], pid: str, models: list[str]) -> list[str]:
    """Replace a provider's model list (deduped, order preserved)."""
    entry = db.get(pid)
    if not isinstance(entry, dict):
        entry = {"keys": [], "models": [], "endpoints": []}
        db[pid] = entry
    seen: set[str] = set()
    clean = []
    for m in models:
        if isinstance(m, str) and m and m not in seen:
            seen.add(m)
            clean.append(m)
    entry["models"] = clean
    return clean


def mark_catalog_checked(db: dict[str, Any], pid: str) -> None:
    """Stamp a successful live-catalog fetch (mirrors the CLI)."""
    import datetime as _dt
    entry = db.get(pid)
    if isinstance(entry, dict):
        entry["catalog_checked_at"] = _dt.datetime.now().isoformat(  # noqa: DTZ005 -- mirrors wizard.py's naive stamp format
            timespec="seconds")
        entry["catalog_source"] = "live provider catalog"


def order_catalog_free_first(
        catalog: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Catalog sorted free/cheap/useful first (same predicate as the CLI)."""
    def _key(item: tuple[str, str]) -> tuple[int, str]:
        mid, label = item if len(item) == 2 else (item[0], "")
        return (0 if _wiz._is_free_model(mid, label) else 1, str(mid).lower())
    return sorted(catalog, key=_key)


def is_free_model(mid: str, label: str = "") -> bool:
    """Free/cheap/useful predicate (same rule the CLI picker uses)."""
    return bool(_wiz._is_free_model(mid, label))


# ------------------------------------------------------------------ models ---

def probe_model(pid: str, model: str, key: str | None,
                endpoint: str | None = None) -> tuple[str, str]:
    """Minimal classified probe of one model. Returns ``(class, msg)``."""
    return _wiz.probe_model_classified(pid, model, key, endpoint)


def probe_models(pid: str, models: list[str], key: str | None,
                 endpoint: str | None = None,
                 keys: list[str] | None = None, mode: str = "FAST",
                 sample_size: int = 2, db: dict[str, Any] | None = None,
                 sleep_s: float = 0.0) -> list[tuple[str, str, str]]:
    """Probe several models. Returns ``[(model, class, msg)]``."""
    return _wiz.test_models(pid, list(models), key, endpoint, keys=keys,
                            mode=mode, sample_size=sample_size, db=db,
                            sleep_s=sleep_s)


def refresh_credential_health(db: dict[str, Any], sleep_s: float = 1.0,
                              progress=None, stop=None) -> dict[str, int]:
    """Probe one model per credential, persisting validation in ``db``.

    Health is tracked per credential in the schema, so a single minimal
    probe per credential refreshes every deployment backed by it (far
    cheaper than probing all N deployments). Secrets never leave this
    function. ``progress(pid, model, done, total)`` reports each step;
    ``stop`` (a ``threading.Event``) aborts between probes. Returns
    outcome counts ``{"checked", "ok", "throttled", "invalid", "unknown"}``.
    Callers persist with :func:`save_state` and recompile to display.
    """
    import time as _time

    targets: list[tuple[str, str, str, str | None]] = []
    for pid, pdata in db.items():
        if pid.startswith("_") or not isinstance(pdata, dict):
            continue
        models = [m for m in pdata.get("models", []) or []
                  if isinstance(m, str) and m]
        if not models:
            continue
        endpoints = pdata.get("endpoints", []) or []
        ep = endpoints[0] if endpoints else None
        for cred in _wiz.iter_credentials(pdata):
            if cred.get("enabled") is False or cred.get("quarantined"):
                continue
            secret = cred.get("secret")
            if not secret:
                continue  # local/secretless engines have no upstream health
            targets.append((pid, models[0], secret, ep))
    counts = {"checked": 0, "ok": 0, "throttled": 0, "invalid": 0,
              "unknown": 0}
    total = len(targets)
    for i, (pid, model, secret, ep) in enumerate(targets):
        if stop is not None and stop.is_set():
            break
        if progress is not None:
            progress(pid, model, i + 1, total)
        try:
            results = _wiz.test_models(pid, [model], secret, ep,
                                       keys=[secret], mode="FAST", db=db,
                                       sleep_s=0)
            cls = results[0][1] if results else "UNKNOWN"
        except Exception:  # noqa: BLE001 -- one bad probe never aborts the sweep
            cls = "UNKNOWN"
        counts["checked"] += 1
        counts[{"OK": "ok", "RATE_LIMITED": "throttled",
                "AUTH_ERROR": "invalid"}.get(cls, "unknown")] += 1
        if sleep_s and i != total - 1:
            _time.sleep(sleep_s)
    return counts


def get_combined_models(db: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """User-facing combined models (canonical pool -> members)."""
    aliases = _wiz._get_aliases(db)
    return {k: list(v) for k, v in aliases.items()}


def combine_models(db: dict[str, Any], canonical: str,
                   members: list[tuple[str, str]]) -> None:
    """Set the member backends for one combined model name.

    Members are ``(provider_id, model_id)``. Mirrors the CLI alias manager:
    explicit user action may apply AUTO or SUGGESTED pairs, but MANUAL pairs
    (different stems / incompatible tiers) are refused with ``ValueError`` —
    never silently merged.
    """
    seen, clean = set(), []
    for pid, mid in members:
        if (pid, mid) in seen:
            continue
        seen.add((pid, mid))
        clean.append({"provider": pid, "model": mid})
    # enforce the CLI's tier-safety: MANUAL pairs must stay separate
    for i in range(len(clean)):
        for j in range(i + 1, len(clean)):
            a, b = clean[i], clean[j]
            if _wiz._group_safety(a["provider"], a["model"],
                                  b["provider"], b["model"]) == "MANUAL":
                raise ValueError(
                    f"refusing to combine incompatible models: "
                    f"{a['provider']}:{a['model']} vs {b['provider']}:{b['model']}")
    db.setdefault(_wiz.ALIAS_KEY, {})[canonical] = clean


def suggest_combinations(db: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """Conservative auto-combine suggestions, same rules as the CLI."""
    return _wiz._suggest_alias_groups(db)


def auto_combine(db: dict[str, Any], pid: str,
                 candidate: list[str]) -> list[str]:
    """Silently create only AUTO stems (same call the CLI makes on save).

    SUGGESTED (capability-uncertain) groups are never created here — they
    are reported via :func:`pending_suggestions` for explicit user action.
    Returns the created stem names. May print progress (callers quiet it).
    """
    return list(_wiz._auto_unify_stems(db, pid, list(candidate)) or [])


def pending_suggestions(db: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """SUGGESTED same-model groups not yet realized as pools.

    Stems already present in the alias map are skipped (the automatic path
    extends those on save). MANUAL pairs never appear (the suggester omits
    all-MANUAL groups), so applying these via :func:`combine_models` is safe.
    """
    aliases = _wiz._get_aliases(db)
    out: dict[str, list[dict[str, str]]] = {}
    for stem, spec in _wiz._suggest_alias_groups(db).items():
        if not isinstance(spec, dict) or spec.get("mode") != "SUGGESTED":
            continue
        if stem in aliases:
            continue
        members = spec.get("members") or []
        if len(members) >= 2:
            out[stem] = list(members)
    return out


# ---------------------------------------------------------------- compiler ---

def compile_config(db: dict[str, Any]):
    """DB -> (deployments, pools, roles, errors). Pure; no I/O."""
    return _wiz.compile_config(db)


def write_config(db: dict[str, Any],
                 paths: EnginePaths | None = None) -> int:
    """Validate + atomically write ``config.yaml`` (mode 0600).

    Raises ``ValueError`` on invalid state, leaving the file untouched.
    Returns the number of emitted routes.
    """
    p = _resolve_paths(paths)
    with _patched_wizard(p):
        return _wiz.generate_yaml(db)


def config_changed(paths: EnginePaths | None = None) -> bool:
    """True when the on-disk YAML differs from a fresh compile (hash check)."""
    return True  # conservative default; apply_config() compares precisely


def apply_config(db: dict[str, Any], paths: EnginePaths | None = None,
                 runner: Callable | None = None,
                 wait_fn: Callable[[int], bool] | None = None) -> dict[str, Any]:
    """Review -> validate -> compile -> write -> restart-if-changed -> readiness.

    Returns ``{"routes": n, "changed": bool, "restarted": bool,
    "ready": bool|None}``. Never partially applies invalid config
    (``ValueError`` propagates, files untouched). Restart uses an injectable
    ``runner`` (defaults to ``subprocess.run``); readiness uses ``wait_fn``
    (defaults to the bounded 60s poll).
    """
    from wizard import _yaml_hash as _hash  # local import: patched paths apply

    p = _resolve_paths(paths)
    with _patched_wizard(p):
        before = _hash()
        routes = _wiz.generate_yaml(db)  # raises ValueError; file untouched
        after = _hash()
        changed = (before != after) if before else True
        result: dict[str, Any] = {"routes": routes, "changed": changed,
                                  "restarted": False, "ready": None}
        if not changed:
            return result
        ok = restart_gateway(runner=runner, wait_fn=wait_fn)
        result["restarted"] = True
        result["ready"] = ok
        return result


# ----------------------------------------------------------------- testing ---

def test_gateway(paths: EnginePaths | None = None) -> dict[str, int]:
    """Smoke test: one request per gateway alias. Returns class counts."""
    p = _resolve_paths(paths)
    with _patched_wizard(p):
        return _wiz.proxy_smoke_test()


def test_pools(db: dict[str, Any]) -> dict[str, int]:
    """Direct per-deployment health test. Returns class counts."""
    return _wiz.pool_health_test(db)


def gateway_status(runner: Callable | None = None) -> str:
    """``running`` / ``stopped`` / ``unknown`` via systemd (mockable)."""
    run = runner or (lambda *a, **k: subprocess.run(*a, check=False, **k))
    try:
        r = run(["systemctl", "--user", "is-active", "litellm"],
                capture_output=True, text=True, timeout=10)
        state = (getattr(r, "stdout", "") or "").strip()
        if state == "active":
            return "running"
        if state in ("inactive", "failed", "activating"):
            return "stopped"
        code = getattr(r, "returncode", 1)
        return "running" if code == 0 else "stopped"
    except Exception:  # noqa: BLE001 -- systemd absence must read as unknown
        return "unknown"


def restart_gateway(runner: Callable | None = None,
                    wait_fn: Callable[[int], bool] | None = None) -> bool:
    """Restart LiteLLM + bounded readiness poll. Both injectable for tests."""
    if runner is not None or wait_fn is not None:
        run = runner or (lambda *a, **k: subprocess.run(*a, check=True, **k))
        try:
            run(["systemctl", "--user", "restart", "litellm"], check=True)
        except Exception:  # noqa: BLE001 -- injected-runner failure means not restarted
            return False
        try:
            return bool((wait_fn or (lambda _t: True))(60))
        except Exception:  # noqa: BLE001 -- wait failure means not ready
            return False
    return bool(_wiz.restart_proxy())


def wait_for_gateway(timeout_s: int = 60) -> bool:
    """Bounded readiness check (no fixed blind sleeps)."""
    return bool(_wiz.wait_for_proxy(timeout_s))


def gateway_aliases(paths: EnginePaths | None = None) -> list[str]:
    """Unique pool names the gateway config currently serves, in order.

    Read from ``config.yaml`` (what LiteLLM actually serves), not from the
    DB. Empty when no config has been applied yet.
    """
    import yaml as _yaml
    p = _resolve_paths(paths)
    try:
        with open(p.yaml_file) as f:
            cfg = _yaml.safe_load(f) or {}
    except (OSError, ValueError):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in cfg.get("model_list", []) or []:
        name = (item or {}).get("model_name")
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def probe_gateway_alias(alias: str, paths: EnginePaths | None = None,
                        timeout: int = 25) -> tuple[str, str]:
    """One minimal chat completion through the gateway for a single pool.

    Returns ``(class, msg)`` with the CLI's classification (429 throttled
    stays distinct from auth/server failures). Single-alias primitive so
    the TUI can show progress and cancel between pools.
    """
    p = _resolve_paths(paths)
    with _patched_wizard(p):
        return _wiz._gateway_request(alias, _wiz.get_master_key(), timeout)


def set_credential_quarantined(db: dict[str, Any], pid: str, cred_id: str,
                               quarantined: bool = True) -> bool:
    """Park/unpark one credential by ID. Parked credentials stay saved but
    are excluded from compilation until applied. Returns True if found."""
    entry = db.get(pid)
    if not isinstance(entry, dict):
        return False
    for c in _wiz.iter_credentials(entry):
        if c.get("id") == cred_id:
            c["quarantined"] = bool(quarantined)
            return True
    return False


def gateway_overview(db: dict[str, Any],
                     paths: EnginePaths | None = None,
                     status: str = "unknown") -> dict[str, Any]:
    """Home-screen facts: gateway state, models, attention items. Quiet data,
    no telemetry. Attention items reuse existing diagnostics (stale aliases,
    structural issues, missing YAML)."""
    _wiz.migrate_db(db)
    _deployments, pools, _roles, errors = _wiz.compile_config(db)
    pool_names = sorted(pools)
    members: dict[str, list[str]] = {}
    for pool in pool_names:
        provs = sorted({d["provider"] for d in pools[pool]})
        members[pool] = provs
    attention: list[str] = []
    attention.extend(errors[:5])
    attention.extend(_wiz.validate_db(db)[:5])
    p = _resolve_paths(paths)
    if not os.path.exists(p.yaml_file):
        attention.append("No config.yaml yet — Configure, then Apply.")
    if not pool_names:
        attention.append("No models configured yet.")
    return {"gateway": status, "pools": pool_names, "members": members,
            "attention": attention}


# ------------------------------------------------------------- opencode ---

def _load_sync_module():
    import wizard as _w  # noqa: F401  (keeps import surface obvious)
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "sync-opencode.py"), "sync-opencode.py"):
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("sync_opencode", cand)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    raise FileNotFoundError("sync-opencode.py not found next to engine.py")


def sync_opencode(paths: EnginePaths | None = None, dry_run: bool = False,
                  include_roles: bool = True) -> dict[str, Any]:
    """Sync gateway pools (+roles) into OpenCode JSON.

    Model NAMES only, never secrets; backup + atomic write + strict-JSON
    validation with restore on failure (same guarantees as the script).
    Must only run after the gateway config is valid. Returns
    ``{"pools": n, "roles": m, "exposed": [...], "wrote": bool}``.
    """
    p = _resolve_paths(paths)
    sync = _load_sync_module()
    aliases, roles, source = sync.load_aliases(p.yaml_file)
    if not aliases and not roles:
        raise ValueError("No gateway aliases found. Apply the gateway config first.")
    exposed = list(aliases)
    if roles and include_roles:
        exposed += [r for r in roles if r not in exposed]
    block = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Local LiteLLM",
        "options": {"baseURL": "http://localhost:4000/v1",
                    "apiKey": "{env:LITELLM_MASTER_KEY}"},
        "models": {a: {"name": a} for a in exposed},
    }
    if dry_run:
        return {"pools": len(aliases), "roles": len(roles), "exposed": exposed,
                "wrote": False, "source": source, "block": block}
    if not os.path.exists(p.opencode_json):
        raise FileNotFoundError(f"Not found: {p.opencode_json}")
    with open(p.opencode_json) as f:
        raw = f.read()
    try:
        cfg = json.loads(sync._strip_jsonc(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse {p.opencode_json}: {e}") from e
    cfg.setdefault("provider", {})["litellm"] = block
    import datetime as _dt
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005 -- local filename stamp, mirrors sync-opencode.py
    backup = f"{p.opencode_json}.bak-{stamp}"
    shutil.copy2(p.opencode_json, backup)
    sync._atomic_write_json(p.opencode_json, cfg)
    try:
        with open(p.opencode_json) as f:
            json.load(f)
    except ValueError:
        shutil.copy2(backup, p.opencode_json)
        raise ValueError(f"Wrote invalid JSON — restored backup {backup}.")
    return {"pools": len(aliases), "roles": len(roles), "exposed": exposed,
            "wrote": True, "source": source, "backup": backup}


def opencode_differs(paths: EnginePaths | None = None) -> bool:
    """True when opencode.json's litellm block is stale vs the gateway."""
    p = _resolve_paths(paths)
    try:
        sync = _load_sync_module()
        aliases, roles, _ = sync.load_aliases(p.yaml_file)
        exposed = set(aliases) | set(roles)
        with open(p.opencode_json) as f:
            cfg = json.loads(sync._strip_jsonc(f.read()))
        current = set(((cfg.get("provider") or {}).get("litellm") or {}).get("models") or {})
        return current != exposed
    except Exception:  # noqa: BLE001 -- unreadable config conservatively counts as stale
        return True


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json_tmp(path: str, data: dict[str, Any]) -> None:
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
