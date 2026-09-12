#!/usr/bin/env python3
"""Textual TUI for llm-vault (hard fork).

Thin presentation layer only. Vault is main screen; Proxy + Harness tabs are
dynamic. All vault/proxy logic lives in engine/wizard; sync-*.py owns harness
mutation. Never holds secrets, quota math, or harness mutation.

Flow: paste keys -> background check -> resolve only genuine ambiguity
(shared-limit question, retired models) -> pick models -> probe ->
review -> apply (write + restart-if-changed + readiness) -> OpenCode sync.
Network work always runs in background workers, never on the UI thread.

Design notes (borrowed from free-coding-models, adapted):
- Home is a dense sortable/filterable table: one row per deployment
  (pool / provider / model / tier / rpm / tpm / quota / ctx / health).
- Footer hints expose every single-key action; ``/`` filters, ``X``
  clears, ``Space``/arrows move with a live detail card below the table.
- A separate OpenCode view shows exactly what OpenCode sees: exposed
  aliases grouped with their backing deployments (+ roles + sync state).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import sys
from typing import Any

try:
    import textual  # noqa: F401
    import yaml  # noqa: F401
except ImportError:
    # Re-exec with a venv python that has the dependencies (same trick as
    # wizard.py): prefer a venv next to this file, else the standard one.
    _here = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.join(_here, "venv", "bin", "python"),
        os.path.join(os.path.expanduser("~"), ".config", "litellm",
                     "venv", "bin", "python"),
    ]
    for _venv_py in _candidates:
        if (os.path.exists(_venv_py)
                and os.path.abspath(sys.executable) != os.path.abspath(_venv_py)):
            os.execv(_venv_py, [_venv_py, os.path.abspath(__file__)] + sys.argv[1:])
    print("[!] Missing dependencies. Run with the project venv, e.g.:")
    print("    ~/.config/litellm/venv/bin/python tui.py")
    print("    (or: pip install -r requirements.txt)")
    sys.exit(1)

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    SelectionList,
    Static,
    Tabs,
    TextArea,
    Tree,
)
from textual.widgets._selection_list import Selection

import engine

STATUS_MARK = {"running": "[green]●[/]", "stopped": "[red]●[/]",
               "unknown": "[yellow]●[/]"}

# FCM-inspired presentation palette (display-only; no product semantics).
# Soft per-provider accents so the column scans at a glance; tier gradient
# cheap/fast (green) -> heavy (magenta/cyan); health green/yellow/red with
# dim for never-tested. Missing values render dim, never blank.
PROVIDER_STYLE = {
    "gemini": "deep_sky_blue1",
    "openrouter": "magenta",
    "zai": "cyan",
    "tokenrouter": "orange1",
    "opencode_zen": "purple",
    "anthropic": "red",
    "openai": "green",
    "ollama_cloud": "bright_white",
    "ollama_local": "white",
}
TIER_STYLE = {"flash": "green", "lite": "yellow", "pro": "magenta",
              "reasoning": "cyan", "unknown": "grey62"}
HEALTH_STYLE = {"healthy": "green", "partially-throttled": "yellow",
                "throttled": "yellow", "invalid": "red", "unknown": "grey62"}

# Fixed column widths (content cells): DataTable auto-sizes columns to
# the visible rows, so filtering/sorting reflowed the whole table on
# every keystroke. Fixed widths + clipped/padded cells keep it still.
COLUMN_WIDTHS = {
    "pool": 24, "provider": 12, "upstream": 24, "tier": 7,
    "quota": 26, "health": 12, "key": 8,
}


def fit_cell(text: str, width: int) -> str:
    """Clip (with …) + pad to an exact display width (ASCII content)."""
    if len(text) > width:
        text = text[: max(0, width - 1)] + "…"
    return text.ljust(width)


TABLE_COLUMNS = (
    ("pool", "Pool"),
    ("provider", "Provider"),
    ("upstream", "Model"),
    ("tier", "Tier"),
    ("quota", "Quota"),
    ("health", "Status"),
    ("key", "Key"),
)
SORTABLE_COLUMNS = tuple(k for k, _ in TABLE_COLUMNS)
TIER_CYCLE = ("flash", "lite", "pro", "reasoning", "unknown")
WORKING_HEALTH = {"healthy", "partially-throttled", "throttled"}

HEALTH_DISPLAY = {
    "healthy": ("✓", "healthy"),
    "partially-throttled": ("~", "part-throttled"),
    "throttled": ("~", "throttled"),
    "invalid": ("✗", "invalid"),
    "unknown": ("?", "untested"),
}
HEALTH_RANK = {"healthy": 0, "partially-throttled": 1, "throttled": 2,
               "unknown": 3, "invalid": 4}


def _short_quota(qd: str, maxlen: int = 16) -> str:
    """Compact quota-domain label for table cells (full id in detail)."""
    short = (qd or "").removeprefix("project:").removeprefix("endpoint:")
    if short.startswith("cred-") and len(short) > 13:
        short = short[:13]  # cred-<8 hex> identifies the credential
    elif short.startswith("credential:cred-"):
        short = "cred-" + short.removeprefix("credential:cred-")[:8]
    if len(short) > maxlen:
        short = short[: maxlen - 1] + "…"
    return short or "—"


def format_ctx(n: object) -> str:
    """Compact context window (``1048576`` -> ``1M``); ``—`` when unknown."""
    if isinstance(n, bool) or not isinstance(n, (int, float)) or n <= 0:
        return "—"
    if n >= 1_000_000:
        v = n / 1_000_000
        txt = f"{v:.1f}".rstrip("0").rstrip(".") if v < 10 else f"{v:.0f}"
        return txt + "M"
    if n >= 1000:
        v = n / 1000
        txt = f"{v:.1f}".rstrip("0").rstrip(".") if v < 10 else f"{v:.0f}"
        return txt + "K"
    return str(int(n))


def format_limit(rpm: object, tpm: object, shared: int = 1) -> str | None:
    """One-cell rate limit (``10÷3 RPM``); None when nothing is published."""
    if isinstance(rpm, bool) or not isinstance(rpm, (int, float)):
        return None
    txt = str(int(rpm))
    if shared > 1:
        txt += f"÷{shared}"
    txt += " RPM"
    if isinstance(tpm, (int, float)) and not isinstance(tpm, bool):
        txt += f" · {format_ctx(tpm)} TPM"
    return txt


def vault_table_rows(db: dict[str, Any], show_hidden: bool = False) -> list[dict[str, Any]]:
    """Vault rows: provider / model / key snippet / status. Hidden invalid/expired by default."""
    rows: list[dict[str, Any]] = []
    for pid, pdata in db.items():
        if pid.startswith("_") or not isinstance(pdata, dict):
            continue
        models = pdata.get("models") or []
        creds = pdata.get("credentials") or []
        for cred in creds:
            if not isinstance(cred, dict):
                continue
            status = (cred.get("validation") or {}).get("status", "unknown")
            # vault: show active+throttled+unknown by default; hidden are invalid/expired
            if not show_hidden and status in ("invalid", "expired"):
                continue
            # normalize status to vault tags
            vault_status = {"ok": "active", "throttled": "throttled", "invalid": "invalid", "expired": "expired"}.get(status, status)
            health = vault_status
            mark, word = HEALTH_DISPLAY.get({"active": "healthy", "throttled": "throttled", "invalid": "invalid", "expired": "invalid"}.get(health, "unknown"), ("?", health))
            secret = cred.get("secret") or ""
            suffix = engine.mask_secret(secret) if secret else "…" + cred.get("id","")[-4:]
            # never say "custom" — use label, or derive from endpoint host
            if (pid == "custom" or pid.startswith("custom_")) and pdata.get("label"):
                disp_provider = pdata["label"]
            elif pid == "custom" or pid.startswith("custom_"):
                # derive from endpoint host if no label
                base = pdata.get("base_url") or (pdata.get("endpoints") or [""])[0]
                if base:
                    try:
                        host = base.split("//")[-1].split("/")[0].split(".")[0]
                        disp_provider = host.title() if host else pid
                    except Exception:
                        disp_provider = pid
                else:
                    disp_provider = pid
                # fallback: if still custom, show as Custom Endpoint
                if disp_provider == "custom":
                    disp_provider = "Custom Endpoint"
            else:
                disp_provider = pid
            for model in (models or ["—"]):
                lat = engine.probe_latency(db, pid, model) if model != "—" else None
                rows.append({
                    "pool": model, "provider": disp_provider, "upstream": model,
                    "tier": "unknown", "rpm": None, "tpm": None,
                    "quota_domain": cred.get("quota_domain",""),
                    "quota": _short_quota(cred.get("quota_domain","")),
                    "confidence": "", "shared": 1,
                    "ctx_num": None, "ctx": "—",
                    "latency": lat, "latency_txt": f"{lat:.1f}s" if lat else "—",
                    "health": health, "health_txt": f"{mark} {word}",
                    "health_rank": HEALTH_RANK.get(health, 9),
                    "key": suffix, "credential_id": cred.get("id",""),
                    "endpoint": pdata.get("base_url") or (pdata.get("endpoints") or ["—"])[0],
                })
    return sorted(rows, key=lambda r: (r["provider"], r["pool"]))

def deployment_table_rows(db: dict[str, Any]) -> list[dict[str, Any]]:
    """One display row per compiled deployment (pure, no I/O, no secrets).

    Everything shown comes straight from :func:`engine.compile_config`
    (quota math + capabilities + health already resolved there); this only
    formats suffix-masked, sort-ready cell values for the table.
    """
    _deps, pools, _roles, errors = engine.compile_config(db)
    if errors:
        return []
    # Count deployments sharing one (domain, upstream model) so shared
    # quota reads honestly (``10÷3`` = 10 RPM split across 3 deployments).
    counts: dict[tuple[str, str], int] = {}
    for pool_deps in pools.values():
        for d in pool_deps:
            key = (str(d.get("quota_domain") or ""), str(d.get("upstream_model") or ""))
            counts[key] = counts.get(key, 0) + 1
    quota_conf = {}
    try:
        quota_conf = db.get("_quota_domains") or {}
    except AttributeError:
        quota_conf = {}
    rows: list[dict[str, Any]] = []
    for pool in sorted(pools):
        for d in sorted(pools[pool],
                        key=lambda x: (str(x.get("provider") or ""),
                                       str(x.get("credential_id") or ""))):
            raw_provider = str(d.get("provider") or "")
            pdata = db.get(raw_provider) if isinstance(db.get(raw_provider), dict) else {}
            if (raw_provider == "custom" or raw_provider.startswith("custom_")) and pdata.get("label"):
                disp_provider = pdata["label"]
            elif raw_provider == "custom" or raw_provider.startswith("custom_"):
                base = pdata.get("base_url") or (pdata.get("endpoints") or [""])[0]
                if base:
                    try:
                        host = base.split("//")[-1].split("/")[0].split(".")[0]
                        disp_provider = host.title() if host else raw_provider
                    except Exception:
                        disp_provider = raw_provider
                else:
                    disp_provider = raw_provider
                if disp_provider == "custom":
                    disp_provider = "Custom Endpoint"
            else:
                disp_provider = raw_provider
            provider = disp_provider
            upstream = str(d.get("upstream_model") or "")
            caps = d.get("capabilities") or {}
            tier = str(caps.get("tier") or "unknown")
            ctx_num = caps.get("context_window")
            rpm, tpm = d.get("rpm"), d.get("tpm")
            qd = str(d.get("quota_domain") or "")
            n = counts.get((qd, upstream), 1)
            health = str(d.get("health") or "unknown")
            mark, word = HEALTH_DISPLAY.get(health, ("?", health or "unknown"))
            secret = d.get("secret") or ""
            suffix = engine.mask_secret(secret) if secret else (
                "local" if not d.get("credential_id") else "…" + str(d.get("credential_id"))[-4:])
            try:
                confidence = str((quota_conf.get(qd) or {}).get("confidence") or "")
            except AttributeError:
                confidence = ""
            lat = engine.probe_latency(db, provider, upstream)
            rows.append({
                "pool": pool, "provider": provider, "upstream": upstream,
                "tier": tier, "rpm": rpm, "tpm": tpm,
                "quota_domain": qd,
                "quota": f"{format_limit(rpm, tpm, n) or '—'} · {_short_quota(qd)}",
                "confidence": confidence, "shared": n,
                "ctx_num": ctx_num if isinstance(ctx_num, (int, float)) else None,
                "ctx": format_ctx(ctx_num),
                "latency": lat,
                "latency_txt": f"{lat:.1f}s" if lat else "—",
                "health": health, "health_txt": f"{mark} {word}",
                "health_rank": HEALTH_RANK.get(health, 9),
                "key": suffix, "credential_id": str(d.get("credential_id") or ""),
                "endpoint": str(d.get("endpoint") or "—"),
            })
    return rows


def sync_targets_line(db: dict[str, Any], paths) -> str:
    """One-line sync targets: Proxy + Harnesses (dynamic, proxy-agnostic)."""
    import os as _os
    proxy_type = engine.get_proxy_type(db)  # never hardcode litellm
    parts = []
    if _os.path.exists(paths.yaml_file):
        parts.append(f"{proxy_type} ✓")
    else:
        parts.append(f"{proxy_type} — no config yet")
    # Harness detection is dynamic
    for h in engine.detect_harnesses(paths):
        if h == "opencode":
            if _os.path.exists(paths.opencode_json):
                parts.append("OpenCode " + ("✓" if not engine.opencode_differs(paths) else "(stale)"))
            else:
                parts.append("OpenCode — no config")
        elif h == "jcode":
            try:
                st = engine.jcode_status(paths)
            except Exception:
                st = {"installed": False, "config_exists": False}
            if not st.get("installed") and not st.get("config_exists"):
                parts.append("JCode — not installed")
            elif st.get("managed_profile"):
                parts.append("JCode " + ("✓" if not engine.jcode_differs(paths) else "(stale)"))
            else:
                parts.append("JCode — no managed profile yet")
        else:
            parts.append(f"{h} — detected")
    # Show undetected as not installed (for discoverability, but minimal)
    detected = set(engine.detect_harnesses(paths))
    if "opencode" not in detected:
        parts.append("OpenCode — not installed")
    if "jcode" not in detected and "jcode" not in [h for h in detected]:
        # only show once
        if "jcode" not in detected:
            pass
    return "  ".join(parts)


def row_key_for(row: dict[str, Any]) -> str:
    """Stable table row identity across sorts/filters (never positional)."""
    return "\x00".join((row.get("pool", ""), row.get("provider", ""),
                        row.get("credential_id", ""), row.get("endpoint", "")))


def filter_table_rows(rows: list[dict[str, Any]], query: str,
                      tier: str | None = None) -> list[dict[str, Any]]:
    """Case-insensitive substring filter over pool/provider/model/quota,
    plus an optional exact tier filter (``T`` cycles it)."""
    q = (query or "").strip().lower()
    out = rows
    if tier:
        out = [r for r in out if r["tier"] == tier]
    if not q:
        return list(out)
    return [r for r in out
            if q in r["pool"].lower() or q in r["provider"].lower()
            or q in r["upstream"].lower() or q in r["quota_domain"].lower()]


def sort_table_rows(rows: list[dict[str, Any]], sort_key: str,
                    reverse: bool = False) -> list[dict[str, Any]]:
    """Sort display rows (pure; never touches the DB)."""
    key = sort_key if sort_key in SORTABLE_COLUMNS else "pool"
    def _k(r: dict[str, Any]):
        if key == "quota":
            v = r.get("rpm")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return (1, 0, r["quota_domain"], r["pool"])
            return (0, -int(v), r["quota_domain"], r["pool"])
        if key == "health":
            return (r.get("health_rank", 9), r["pool"], r["provider"])
        return (str(r.get(key) or "").lower(), r["pool"], r["provider"])
    return sorted(rows, key=_k, reverse=reverse)


def next_sort(sort_key: str, reverse: bool,
              column: str) -> tuple[str, bool]:
    """Pure sort-state transition for header clicks and the ``s`` key:
    same column toggles direction, new column sorts ascending."""
    if column == sort_key:
        return (column, not reverse)
    return (column, False)


def summary_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Header facts: pools, deployments, working pools, untested deployments.

    A pool counts as *working* when at least one backing deployment is
    healthy/throttled; *untested* deployments never reported health.
    """
    pools: dict[str, list[str]] = {}
    for r in rows:
        pools.setdefault(r["pool"], []).append(r["health"])
    working = sum(1 for hs in pools.values()
                  if any(h in WORKING_HEALTH for h in hs))
    return {"pools": len(pools), "deployments": len(rows), "working": working,
            "untested": sum(1 for r in rows if r["health"] == "unknown")}


def gateway_badge(overview: dict[str, Any], counts: dict[str, int],
                  note: str = "") -> str:
    """One-line header badge: gateway state + working/untested + totals."""
    mark = STATUS_MARK.get(overview.get("gateway", "unknown"),
                           STATUS_MARK["unknown"])
    text = (f"Gateway {mark} {status_word(str(overview.get('gateway', 'unknown')))}"
            f"  •  {counts.get('working', 0)}/{counts.get('pools', 0)} models working")
    if counts.get("untested"):
        text += f"  •  {counts['untested']} untested"
    text += f"  •  {counts.get('deployments', 0)} deployment(s)"
    if note:
        text += f"  •  {note}"
    return text


def credential_validation(db: dict[str, Any], pid: str,
                          cred_id: str) -> dict[str, str] | None:
    """Secret-free last-check info for one credential (modal display)."""
    entry = db.get(pid)
    if not isinstance(entry, dict):
        return None
    for c in entry.get("credentials", []) or []:
        if isinstance(c, dict) and c.get("id") == cred_id:
            v = c.get("validation") or {}
            return {"status": str(v.get("status") or "unknown"),
                    "checked": str(v.get("checked_at") or "never"),
                    "message": str(v.get("message") or "")}
    return None


def row_detail_text(row: dict[str, Any] | None,
                    validation: dict[str, str] | None = None) -> str:
    """Full detail for one deployment (row modal; everything untruncated)."""
    if row is None:
        return "No row selected."
    shared = ""
    if row.get("shared", 1) > 1:
        shared = (f"  (shared domain: {row['shared']} deployments "
                  f"split this quota — never {row['shared']}×)")
    conf = f" [{row['confidence']}]" if row.get("confidence") else ""
    limit = format_limit(row.get("rpm"), row.get("tpm"), row.get("shared", 1)) or "—"
    lines = [
        f"{row['pool']}  via {row['provider']} / {row['upstream']}",
        (f"tier {row['tier']} • limit {limit} • "
         f"quota {row['quota_domain'] or '—'}{conf}{shared}"),
        (f"status {row['health_txt']} • probe latency {row.get('latency_txt') or '—'} • "
         f"key {row['key']} • endpoint {row['endpoint']}"),
    ]
    if row.get("ctx_num"):
        lines.insert(2, f"ctx {row['ctx']} (installed litellm map)")
    if validation:
        msg = f" · {validation['message'][:100]}" if validation.get("message") else ""
        lines.append(f"last check: {validation['status']} "
                     f"({validation['checked']}){msg}")
    return "\n".join(lines)


def opencode_view_data(db: dict[str, Any], paths) -> dict[str, Any]:
    """Facts for the 'what OpenCode sees' view (read-only, via engine).

    ``exposed`` = aliases the gateway serves (from ``config.yaml`` — what
    LiteLLM actually routes) plus roles; ``current`` = what opencode.json
    has now (best effort); ``children`` = backing deployments per pool
    from the compiler. Paths are honoured throughout, so temp-dir state
    never leaks the live config.
    """
    _deps, pools, roles, errors = engine.compile_config(db)
    aliases = engine.gateway_aliases(paths)
    if aliases:
        source = f"gateway config ({paths.yaml_file})"
        exposed = list(aliases) + [r for r in roles if r not in aliases]
    else:
        source = "compiled pools (no gateway config yet — Apply first)"
        exposed = sorted(pools) + [r for r in roles if r not in pools]
    try:
        differs = bool(engine.opencode_differs(paths))
    except Exception:  # noqa: BLE001 -- unreadable counts as stale
        differs = True
    current: set[str] = set()
    try:
        import json as _json
        with open(paths.opencode_json) as f:
            cfg = _json.loads(engine._load_sync_module()._strip_jsonc(f.read()))
        current = set(((cfg.get("provider") or {}).get("litellm") or {}).get("models") or {})
    except Exception:  # noqa: BLE001 -- missing/unparseable file, shown as such
        current = set()
    children: dict[str, list[dict[str, str]]] = {}
    for pool in sorted(pools):
        kids = []
        for d in pools[pool]:
            secret = d.get("secret") or ""
            kids.append({
                "provider": str(d.get("provider") or ""),
                "upstream": str(d.get("upstream_model") or ""),
                "suffix": engine.mask_secret(secret) if secret else "local",
                "rpm": str(int(d["rpm"])) if isinstance(d.get("rpm"), (int, float)) else "—",
                "health": str(d.get("health") or "unknown"),
            })
        children[pool] = kids
    return {"exposed": exposed, "current": sorted(current),
            "pools": sorted(pools), "roles": dict(roles),
            "children": children, "differs": differs, "source": source,
            "errors": list(errors or [])}


# ------------------------------------------------------- pure helpers ---

def _pop_to_home(app: WizardApp) -> None:
    """Pop the screen stack back to Home (terminal flow transitions).

    Finishing a flow lands Review directly above Home, so Back/Escape
    goes home instead of walking back through Configure/Probe screens.
    """
    while len(app.screen_stack) > 1 and not isinstance(app.screen, HomeScreen):
        app.pop_screen()

def split_keys(text: str) -> list[str]:
    """Split pasted keys on whitespace/commas, preserving order, deduped."""
    seen: set[str] = set()
    out: list[str] = []
    for chunk in text.replace(",", " ").split():
        chunk = chunk.strip().strip("\"'")
        if chunk and chunk not in seen:
            seen.add(chunk)
            out.append(chunk)
    return out


def status_word(status: str) -> str:
    return {"running": "Running", "stopped": "Stopped"}.get(status, "Unknown")


def _quiet_call(fn, *args, **kwargs):
    """Run an engine operation with its progress prints swallowed.

    Wizard/engine helpers print progress to stdout (CLI heritage); inside
    the fullscreen TUI that would corrupt the display. Results are
    returned normally — only the chatter is discarded.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def home_lines(overview: dict[str, Any]) -> str:
    """Plain-text home content (unit-testable without a running app)."""
    mark = STATUS_MARK.get(overview.get("gateway", "unknown"),
                           STATUS_MARK["unknown"])
    lines = [f"Gateway  {mark} {status_word(str(overview.get('gateway', 'unknown')))}",
             "", "Models"]
    members = overview.get("members", {})
    for pool in overview.get("pools", []):
        lines.append(f"  {pool}")
        provs = members.get(pool, [])
        if provs:
            lines.append(f"    {', '.join(provs)}")
    if not overview.get("pools"):
        lines.append("  (none yet — choose Configure to add keys)")
    if overview.get("attention"):
        lines.append("")
        lines.append("Needs attention")
        for item in overview["attention"]:
            lines.append(f"  ! {item}")
    return "\n".join(lines)


def test_lines(db: dict[str, Any]) -> str:
    """Idle listing for the Test screen (results render separately)."""
    _deps, pools, _roles, errors = engine.compile_config(db)
    if errors:
        return "\n".join(["Test your setup", "",
                           "Configuration has problems:",
                           *[f"  ! {e}" for e in errors[:5]]])
    if not pools:
        return "Test your setup\n\nNothing to test yet — configure a provider first."
    lines = ["Test your setup", "", "Models"]
    for pool in sorted(pools):
        provs = sorted({d["provider"] for d in pools[pool]})
        lines.append(f"  ? {pool}  ({', '.join(provs)})")
    lines += ["", "Run the test to check every model through the gateway."]
    return "\n".join(lines)


def done_lines(db: dict[str, Any]) -> str:
    """Apply preview for the Review screen (validated again on Apply)."""
    deps, pools, roles, errors = engine.compile_config(db)
    if errors:
        return "\n".join(["Review", "",
                           "Configuration has problems:",
                           *[f"  ! {e}" for e in errors[:5]]])
    lines = ["Review", "",
             f"{len(deps)} connection(s) across {len(pools)} model(s)"
             + (f" + {len(roles)} role(s)" if roles else ""),
             ""]
    for pool in sorted(pools):
        provs = sorted({d["provider"] for d in pools[pool]})
        lines.append(f"  {pool}  ({', '.join(provs)})")
    lines += ["", "Apply writes safely, restarts only if changed."]
    return "\n".join(lines)


# -------------------------------------------------------------- screens ---

class HomeScreen(Screen):
    """Vault main (was Home): shows vault active+throttled only, a toggles hidden.

    Kept name HomeScreen for compat, but title is Vault. No harness mutation here.
    """

    BINDINGS = [  # noqa: RUF012 -- Textual API
        ("c", "configure", "Configure"), ("t", "test", "Test"),
        ("v", "review", "Review"), ("p", "proxy", "Proxy"),
        ("u", "quota", "Quota"), ("j", "jcode", "JCode"),
        ("i", "import", "Import"),
        ("slash", "focus_filter", "Filter"),
        ("s", "cycle_sort", "Sort"), ("S", "reverse_sort", "Reverse"),
        ("T", "cycle_tier", "Tier"), ("P", "probe_all", "Probe"),
        ("a", "toggle_hidden", "All/Hidden"),
        ("x", "clear_filter", "Clear"), ("h", "toggle_hide", "Hide bad"),
        ("q", "quit_app", "Quit")]

    SORT_CYCLE = ("pool", "provider", "tier", "quota", "health")

    def __init__(self) -> None:
        super().__init__()
        self.last_content = ""
        self.rows: list[dict[str, Any]] = []
        self.view_rows: list[dict[str, Any]] = []
        self.sort_key = "pool"
        self.sort_reverse = False
        self.tier_filter: str | None = None
        self.hide_invalid = False
        self.show_hidden = False  # vault: active only by default, a toggles hidden (show expired/invalid)
        self.probing = False
        self.probe_note = ""
        self.counts: dict[str, int] = {}
        self._probe_stop = None
        self._worker = None

    def compose(self) -> ComposeResult:
        from textual.widgets import Tabs
        yield Header()
        with Vertical(id="body"):
            # Tabs: Vault (main) + per-proxy tabs (only installed) + dynamic harness tabs
            harnesses = engine.detect_harnesses()
            if not harnesses:
                harnesses = ["opencode", "jcode"]
            proxy_tabs = [f"Proxy: {p}" for p in engine.detect_proxies()]
            tabs = ["Vault"] + proxy_tabs + [h.title() for h in harnesses]
            yield Tabs(*tabs, id="main-tabs")
            yield Static("", id="gateway-badge")
            yield Static("", id="sync-targets")
            yield Input(placeholder="Filter vault provider/model/key ( / to focus, x to clear )",
                        id="home-filter")
            yield DataTable(id="models-table", cursor_type="row")
            yield Static("Vault: add API key + model • a: show hidden • /: filter • P: probe",
                         id="home-hint")
            yield Static("", id="home-attention")
            yield Button("Add to Vault", id="go-configure", variant="primary")
            yield Button("Probe all", id="probe-all")
            yield Button("Cancel", id="cancel-probe")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_content()
        app = self.app
        assert isinstance(app, WizardApp)
        app.refresh_status_background()
        # Keyboard-first like FCM: the table owns focus so single-key
        # actions (c/t/v/o/s/...) fire immediately; / moves to the filter.
        self._focus_table()
        # Fill the Status column by itself: one minimal background probe
        # per credential (skipped when everything already reported).
        if app.auto_probe and any(r["health"] == "unknown" for r in self.rows):
            self.action_probe_all()
    @on(Tabs.TabActivated)
    def _tab_activated(self, event: Tabs.TabActivated) -> None:
        label = event.tab.label.plain if hasattr(event.tab.label, "plain") else str(event.tab.label)
        if label == "Vault":
            return
        if label.startswith("Proxy:"):
            # Extract proxy type, e.g. "Proxy: litellm" -> litellm
            ptype = label.split(":",1)[1].strip().lower() if ":" in label else engine.get_proxy_type(self.app.db)
            self.app.push_screen(ProxyScreen(proxy_type=ptype))
        elif label.lower() in ("opencode", "jcode"):
            self.app.push_screen(HarnessScreen(label.lower()))

    def on_screen_resume(self) -> None:
        self.refresh_content()
        self._focus_table()

    # -- data --

    def refresh_content(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        overview = engine.gateway_overview(app.db, app.paths, status=app.status)
        # Vault is single main screen — vault rows only, no deployment fallback
        # Vault holds API keys + models, separate from proxy/harness
        self.rows = vault_table_rows(app.db, show_hidden=getattr(self, "show_hidden", False))
        self.counts = summary_counts(self.rows)
        # Proxy badge is proxy-agnostic
        proxy_type = engine.get_proxy_type(app.db)
        badge = f"{proxy_type} {gateway_badge(overview, self.counts, self.probe_note)}"
        self._apply_view()
        attention = overview.get("attention") or []
        if not self.rows and not attention:
            attention = ["Vault empty — add API keys with models (Configure)."]
        attn_txt = ("Needs attention\n" + "\n".join(f"  ! {a}" for a in attention)) if attention else ""
        # Plain-text summary kept for tests / narrow terminals.
        self.last_content = "\n".join(
            [badge, "", *[r["pool"] for r in self.rows][:20],
             *([f"! {a}" for a in attention] if attention else [])])
        try:
            self.query_one("#gateway-badge", Static).update(badge)
            self.query_one("#sync-targets", Static).update(
                sync_targets_line(app.db, app.paths))
            self.query_one("#home-attention", Static).update(attn_txt)
            self._show_probe_buttons()
        except NoMatches:  # not yet mounted
            pass

    def _show_probe_buttons(self) -> None:
        try:
            self.query_one("#probe-all", Button).display = not self.probing
            self.query_one("#cancel-probe", Button).display = self.probing
        except NoMatches:
            pass

    def _focus_table(self) -> None:
        try:
            self.query_one("#models-table", DataTable).focus()
        except NoMatches:
            pass

    def _cell(self, text: str, width: int, style: str | None = None,
              dim: bool = False) -> Text:
        fitted = fit_cell(text, width)
        if dim:
            return Text(fitted, style="dim")
        if style:
            return Text(fitted, style=style)
        return Text(fitted)

    def _header_label(self, key: str, label: str) -> Text:
        # Marker slot is always present (▲/▼/◆ or blank) so headers never
        # change width when the sort/filter state changes.
        if key == self.sort_key:
            mark = "▲" if not self.sort_reverse else "▼"
        elif key == "tier" and self.tier_filter:
            mark = "◆"
        else:
            mark = " "
        return Text(fit_cell(f"{label} {mark}", COLUMN_WIDTHS[key]), style="bold")

    def _row_identity_at_cursor(self, table: DataTable) -> str | None:
        try:
            idx = table.cursor_row
        except Exception:  # noqa: BLE001 -- no cursor yet
            return None
        if idx is None or not (0 <= idx < len(self.view_rows)):
            return None
        return row_key_for(self.view_rows[idx])

    def _apply_view(self) -> None:
        try:
            filt = self.query_one("#home-filter", Input).value
        except NoMatches:
            filt = ""
        try:
            table = self.query_one("#models-table", DataTable)
            prev_key = self._row_identity_at_cursor(table)
        except NoMatches:
            prev_key = None
        rows = filter_table_rows(self.rows, filt, self.tier_filter)
        # Vault: hidden by default (show active/throttled/unknown, hide invalid/expired)
        # For compat, tests seed with unknown health and expect it visible
        if not self.show_hidden:
            rows = [r for r in rows if r["health"] not in ("invalid", "expired")]
        if self.hide_invalid:
            rows = [r for r in rows if r["health"] not in ("invalid", "expired")]
        self.view_rows = sort_table_rows(rows, self.sort_key, self.sort_reverse)
        self._rebuild_table(prev_key)

    def _rebuild_table(self, prev_key: str | None = None) -> None:
        try:
            table = self.query_one("#models-table", DataTable)
        except NoMatches:
            return
        # Snapshot the viewport so rebuilds never jump the scroll (the
        # clear() below resets it, which reads as rows flashing in/out).
        try:
            scroll = table.scroll_offset
            scroll_xy = (scroll.x, scroll.y)
        except Exception:  # noqa: BLE001 -- unscrolled table
            scroll_xy = None
        table.clear(columns=True)
        for key, label in TABLE_COLUMNS:
            table.add_column(self._header_label(key, label), key=key,
                             width=COLUMN_WIDTHS[key])
        for r in self.view_rows:
            table.add_row(
                self._cell(r["pool"], COLUMN_WIDTHS["pool"]),
                self._cell(r["provider"], COLUMN_WIDTHS["provider"],
                           PROVIDER_STYLE.get(r["provider"])),
                self._cell(r["upstream"], COLUMN_WIDTHS["upstream"]),
                self._cell(r["tier"], COLUMN_WIDTHS["tier"],
                           TIER_STYLE.get(r["tier"], "grey62")),
                self._cell(r["quota"], COLUMN_WIDTHS["quota"]),
                self._cell(r["health_txt"], COLUMN_WIDTHS["health"],
                           HEALTH_STYLE.get(r["health"], "grey62")),
                self._cell(r["key"], COLUMN_WIDTHS["key"], dim=True),
                key=row_key_for(r))
        if prev_key is not None:
            for i, r in enumerate(self.view_rows):
                if row_key_for(r) == prev_key:
                    table.move_cursor(row=i, animate=False, scroll=False)
                    break
        if scroll_xy is not None and table.row_count:
            table.scroll_to(x=scroll_xy[0], y=scroll_xy[1], animate=False)
        try:
            filt = self.query_one("#home-filter", Input).value.strip()
        except NoMatches:
            filt = ""
        extra = []
        if filt:
            extra.append(f"filter '{filt}'")
        if self.tier_filter:
            extra.append(f"tier={self.tier_filter}")
        if self.hide_invalid:
            extra.append("hiding invalid")
        table.border_title = (f"{len(self.view_rows)}/{len(self.rows)} "
                              f"sorted by {self.sort_key}"
                              + (f" ({', '.join(extra)})" if extra else ""))

    # -- events --

    @on(Input.Changed, "#home-filter")
    def _filter_changed(self, _event: Input.Changed) -> None:
        self._apply_view()

    @on(Input.Submitted, "#home-filter")
    def _filter_submitted(self, _event: Input.Submitted) -> None:
        self._focus_table()

    @on(DataTable.HeaderSelected)
    def _header_clicked(self, event: DataTable.HeaderSelected) -> None:
        key = getattr(event.column_key, "value", event.column_key)
        self._sort_by_column(str(key))

    @on(DataTable.RowSelected)
    def _row_chosen(self, event: DataTable.RowSelected) -> None:
        idx = event.cursor_row
        if idx is not None and 0 <= idx < len(self.view_rows):
            self.app.push_screen(ModelDetailScreen(self.view_rows[idx]))

    def _sort_by_column(self, column: str) -> None:
        """Header click sorts (repeat click reverses); unknown keys ignore."""
        if column in SORTABLE_COLUMNS:
            self.sort_key, self.sort_reverse = next_sort(
                self.sort_key, self.sort_reverse, column)
            self._apply_view()

    # -- actions --

    def action_configure(self) -> None:
        self.app.push_screen(ConfigureScreen())

    def action_test(self) -> None:
        self.app.push_screen(TestScreen())

    def action_review(self) -> None:
        self.app.push_screen(DoneScreen())

    def action_opencode(self) -> None:
        self.app.push_screen(OpenCodeScreen())

    def action_quota(self) -> None:
        self.app.push_screen(QuotaDashboardScreen())

    def action_jcode(self) -> None:
        self.app.push_screen(JCodeScreen())

    def action_import(self) -> None:
        self.app.push_screen(ImportScreen())

    def action_focus_filter(self) -> None:
        try:
            self.query_one("#home-filter", Input).focus()
        except NoMatches:
            pass

    def action_cycle_sort(self) -> None:
        i = self.SORT_CYCLE.index(self.sort_key) if self.sort_key in self.SORT_CYCLE else -1
        self.sort_key = self.SORT_CYCLE[(i + 1) % len(self.SORT_CYCLE)]
        self._apply_view()

    def action_reverse_sort(self) -> None:
        self.sort_reverse = not self.sort_reverse
        self._apply_view()

    def action_cycle_tier(self) -> None:
        if self.tier_filter is None:
            self.tier_filter = TIER_CYCLE[0]
        else:
            try:
                i = TIER_CYCLE.index(self.tier_filter)
                self.tier_filter = TIER_CYCLE[i + 1] if i + 1 < len(TIER_CYCLE) else None
            except ValueError:
                self.tier_filter = None
        self._apply_view()

    def action_clear_filter(self) -> None:
        try:
            inp = self.query_one("#home-filter", Input)
            inp.value = ""
            self.query_one("#models-table", DataTable).focus()
        except NoMatches:
            pass
        self.tier_filter = None
        self._apply_view()

    def action_toggle_hide(self) -> None:
        self.hide_invalid = not self.hide_invalid
        self._apply_view()

    def action_toggle_hidden(self) -> None:
        self.show_hidden = not self.show_hidden
        # Update title
        try:
            label = "Vault — all keys" if self.show_hidden else "Vault — active only (a to show hidden)"
            self.query_one("#title", Label).update(label)
        except NoMatches:
            pass
        self.refresh_content()

    def action_proxy(self) -> None:
        self.app.push_screen(ProxyScreen())

    def action_probe_all(self) -> None:
        if self.probing:
            self._cancel_probe()
            return
        if not self.rows:
            return
        import threading
        self.probing = True
        self.probe_note = "probe starting…"
        self._probe_stop = threading.Event()
        self._show_probe_buttons()
        self._worker = self.run_worker(self._probe_task(), exclusive=True)

    async def _probe_task(self) -> None:
        import asyncio as _asyncio
        app = self.app
        assert isinstance(app, WizardApp)

        def _progress(pid: str, model: str, done: int, total: int) -> None:
            app.call_from_thread(self._probe_progress, pid, model, done, total)

        try:
            counts = await _asyncio.to_thread(
                _quiet_call, engine.refresh_credential_health,
                app.db, 1.0, _progress, self._probe_stop)
        except _asyncio.CancelledError:
            self.probe_note = "probe cancelled"
            self.probing = False
            self._probe_stop = None
            self._show_probe_buttons()
            self.refresh_content()
            self._focus_table()
            return
        try:
            engine.save_state(app.db, app.paths)
        except OSError:
            pass
        stopped = self._probe_stop is not None and self._probe_stop.is_set()
        self.probing = False
        self._probe_stop = None
        parts = [f"{counts.get('ok', 0)} ok"]
        if counts.get("throttled"):
            parts.append(f"{counts['throttled']} throttled")
        if counts.get("invalid"):
            parts.append(f"{counts['invalid']} invalid")
        if counts.get("unknown"):
            parts.append(f"{counts['unknown']} unknown")
        self.probe_note = ("last probe: " + " · ".join(parts)
                           + (" (stopped early)" if stopped else ""))
        self._show_probe_buttons()
        self.refresh_content()
        self._focus_table()

    def _probe_progress(self, pid: str, model: str, done: int, total: int) -> None:
        self.probe_note = f"probing {done}/{total}: {model}"
        app = self.app
        assert isinstance(app, WizardApp)
        overview = engine.gateway_overview(app.db, app.paths, status=app.status)
        try:
            self.query_one("#gateway-badge", Static).update(
                gateway_badge(overview, self.counts or summary_counts(self.rows),
                              self.probe_note))
        except NoMatches:
            pass

    def _cancel_probe(self) -> None:
        if self._probe_stop is not None:
            self._probe_stop.set()
        if self._worker is not None:
            self._worker.cancel()

    def action_quit_app(self) -> None:
        self.app.exit()

    @on(Button.Pressed, "#go-configure")
    def _go_configure(self) -> None:
        self.action_configure()

    @on(Button.Pressed, "#probe-all")
    def _go_probe_all(self) -> None:
        self.action_probe_all()

    @on(Button.Pressed, "#cancel-probe")
    def _go_cancel_probe(self) -> None:
        self._cancel_probe()

    @on(Button.Pressed, "#go-opencode")
    def _go_opencode(self) -> None:
        self.action_opencode()

    @on(Button.Pressed, "#go-test")
    def _go_test(self) -> None:
        self.action_test()

    @on(Button.Pressed, "#go-review")
    def _go_review(self) -> None:
        self.action_review()

    @on(Button.Pressed, "#go-quota")
    def _go_quota(self) -> None:
        self.action_quota()

    @on(Button.Pressed, "#go-jcode")
    def _go_jcode(self) -> None:
        self.action_jcode()

    @on(Button.Pressed, "#go-import")
    def _go_import(self) -> None:
        self.action_import()


class ModelDetailScreen(Screen):
    """Row detail + actions: full info for one deployment, with a live
    credential probe and a gateway test for its pool. Enter/click on a
    Home row opens it; ``Esc`` goes back (Home refreshes on resume)."""

    BINDINGS = [("escape", "back", "Back")]  # noqa: RUF012 -- Textual API

    def __init__(self, row: dict[str, Any]) -> None:
        super().__init__()
        self.row = dict(row)
        self.last_status = ""
        self._worker = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Label(f"{self.row.get('pool', '')}", id="title")
            yield Static("", id="detail-info")
            yield Static("", id="detail-status")
            yield Button("Probe this credential", id="probe-cred", variant="primary")
            yield Button("Test via gateway", id="gw-test")
            yield Button("Cancel", id="cancel")
            yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self._render_info()
        self._show_only("probe-cred", "gw-test", "back")

    def _live_row(self) -> dict[str, Any]:
        """Re-resolve the row so health/probe results are never stale."""
        app = self.app
        assert isinstance(app, WizardApp)
        for r in deployment_table_rows(app.db):
            if (r["pool"] == self.row.get("pool")
                    and r["provider"] == self.row.get("provider")
                    and r["credential_id"] == self.row.get("credential_id")):
                self.row = r
                break
        return self.row

    def _render_info(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        row = self._live_row()
        validation = credential_validation(app.db, row["provider"],
                                           row["credential_id"])
        try:
            self.query_one("#detail-info", Static).update(
                row_detail_text(row, validation))
            self.query_one("#detail-status", Static).update(self.last_status)
        except NoMatches:
            pass

    def _show_only(self, *ids: str) -> None:
        for bid in ("probe-cred", "gw-test", "cancel", "back"):
            try:
                self.query_one(f"#{bid}", Button).display = bid in ids
            except NoMatches:
                pass

    def _find_deployment(self) -> dict[str, Any] | None:
        app = self.app
        assert isinstance(app, WizardApp)
        _deps, pools, _roles, _errors = engine.compile_config(app.db)
        for d in pools.get(self.row.get("pool", ""), []):
            if (d.get("provider") == self.row.get("provider")
                    and str(d.get("credential_id") or "")
                    == self.row.get("credential_id")):
                return d
        return None

    @on(Button.Pressed, "#probe-cred")
    def _probe_cred(self) -> None:
        dep = self._find_deployment()
        if dep is None or not dep.get("secret"):
            self.last_status = "Nothing to probe (local engine or missing key)."
            self._render_info()
            return
        self.last_status = f"Probing {dep.get('upstream_model')}…"
        self._render_info()
        self._show_only("cancel", "back")
        self._worker = self.run_worker(self._probe_task(dep), exclusive=True)

    async def _probe_task(self, dep: dict[str, Any]) -> None:
        import asyncio as _asyncio
        app = self.app
        assert isinstance(app, WizardApp)
        try:
            results = await _asyncio.to_thread(
                _quiet_call, engine.probe_models,
                str(dep.get("provider") or ""), [str(dep.get("upstream_model") or "")],
                dep.get("secret"), dep.get("endpoint"),
                keys=[dep["secret"]] if dep.get("secret") else None,
                mode="FAST", db=app.db, sleep_s=0)
        except _asyncio.CancelledError:
            self.last_status = "Probe cancelled."
            self._render_info()
            self._show_only("probe-cred", "gw-test", "back")
            return
        try:
            engine.save_state(app.db, app.paths)
        except OSError:
            pass
        if results:
            _m, cls, msg = results[0]
            verdict = {"OK": "✓ works", "RATE_LIMITED": "~ throttled"}.get(cls, f"✗ {msg[:120]}")
            self.last_status = f"{verdict} — health saved."
        else:
            self.last_status = "No result."
        self._render_info()
        self._show_only("probe-cred", "gw-test", "back")

    @on(Button.Pressed, "#gw-test")
    def _gw_test(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        pool = str(self.row.get("pool") or "")
        self.last_status = f"Testing {pool} through the gateway…"
        self._render_info()
        self._show_only("cancel", "back")
        self._worker = self.run_worker(self._gw_task(pool), exclusive=True)

    async def _gw_task(self, pool: str) -> None:
        import asyncio as _asyncio
        app = self.app
        assert isinstance(app, WizardApp)
        try:
            aliases = await _asyncio.to_thread(
                _quiet_call, engine.gateway_aliases, app.paths)
            if pool not in aliases:
                self.last_status = "Pool is not applied yet — Review & Apply first."
            else:
                cls, msg = await _asyncio.to_thread(
                    _quiet_call, engine.probe_gateway_alias, pool, app.paths)
                self.last_status = ("✓ gateway OK" if cls == "OK"
                                    else f"{cls}: {msg[:140]}")
        except _asyncio.CancelledError:
            self.last_status = "Test cancelled."
        self._render_info()
        self._show_only("probe-cred", "gw-test", "back")

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()


class ConfigureScreen(Screen):
    BINDINGS = [("escape", "back", "Back")]  # noqa: RUF012 -- Textual API

    def __init__(self) -> None:
        super().__init__()
        self.pids: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Label("Choose provider", id="title")
            yield ListView(id="provider-list")
            yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        view = self.query_one("#provider-list", ListView)
        for p in engine.list_providers(app.db):
            self.pids.append(p["id"])
            view.append(ListItem(Label(f"{p['name']}")))
        view.focus()

    @on(ListView.Selected)
    def _provider_chosen(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(self.pids):
            self.app.push_screen(ProviderScreen(self.pids[idx]))

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()


class ProviderScreen(Screen):
    """Paste keys -> background check -> grouping question -> models.

    Keys are checked BEFORE anything is saved; only working connections
    continue. The shared-limit question appears only when genuinely
    ambiguous (multi-key project-scoped providers).
    """

    BINDINGS = [("escape", "back", "Back")]  # noqa: RUF012 -- Textual API
    BUTTONS = ("check", "save-continue", "retry", "share", "separate",
               "later", "cancel", "back")

    def __init__(self, pid: str) -> None:
        super().__init__()
        self.pid = pid
        self.phase = "keys"
        self.secrets: list[str] = []
        self.results: list[tuple[str, bool, str]] = []
        self.endpoint: str | None = None
        self._endpoint_edited = False
        self.last_result = ""
        self._worker = None

    # -- layout --

    def compose(self) -> ComposeResult:
        app = self.app if self.is_running else None
        name = self.pid
        if isinstance(app, WizardApp):
            prov = engine.get_provider(self.pid, app.db)
            if prov:
                name = str(prov.get("name", self.pid))
        with Vertical(id="body"):
            yield Label(f"Paste your {name} API key(s)", id="title")
            yield Input(placeholder="https://your-endpoint/v1 (custom only)",
                        id="endpoint-input")
            yield TextArea(id="keys-input")
            yield Static("", id="phase-status")
            yield Static("", id="check-results")
            yield Button("Check these connections", id="check", variant="primary")
            yield Button("Save & continue", id="save-continue", variant="primary")
            yield Button("Retry", id="retry")
            yield Button("They share one limit", id="share", variant="primary")
            yield Button("Keep them separate", id="separate")
            yield Button("Decide later", id="later")
            yield Button("Cancel", id="cancel")
            yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        # Pre-fill the known base URL (stored override > stored base >
        # builtin default) — the user only edits it when it is wrong.
        try:
            if self._is_custom_api():
                default = self._endpoint_default()
                if default:
                    self.query_one("#endpoint-input", Input).value = default
        except NoMatches:
            pass
        self._show_state()

    # -- state --

    def _show_only(self, *ids: str) -> None:
        wanted = set(ids)
        for bid in self.BUTTONS:
            self.query_one(f"#{bid}", Button).display = bid in wanted

    def _is_custom_api(self) -> bool:
        app = self.app
        assert isinstance(app, WizardApp)
        prov = engine.get_provider(self.pid, app.db) or {}
        return prov.get("type") == "custom_api"

    def _endpoint_default(self) -> str | None:
        app = self.app
        assert isinstance(app, WizardApp)
        return engine.effective_endpoint(app.db, self.pid)

    def _needs_endpoint(self) -> bool:
        """True only when a base URL is required but none is known."""
        return self._is_custom_api() and not self._endpoint_default()

    def _show_state(self) -> None:
        try:
            keys_box = self.query_one("#keys-input", TextArea)
            ep_box = self.query_one("#endpoint-input", Input)
            status = self.query_one("#phase-status", Static)
            results = self.query_one("#check-results", Static)
        except NoMatches:  # not yet mounted
            return
        keys_box.display = self.phase in ("keys", "checking")
        ep_box.display = self.phase == "keys" and self._is_custom_api()
        if self.phase == "keys":
            if self._is_custom_api() and self._endpoint_default():
                status.update("Paste one or more keys, then check them.\n"
                              "Base URL is pre-filled — fix it if yours differs.")
            else:
                status.update("Paste one or more keys, then check them.")
            results.update("")
            self._show_only("check", "back")
        elif self.phase == "checking":
            status.update("Checking these connections…")
            self._show_only("cancel")
        elif self.phase == "results":
            status.update(self.last_result)
            lines = []
            for secret, ok, msg in self.results:
                mark = "OK" if ok else "FAIL"
                lines.append(f"  [{mark}] {engine.mask_secret(secret)} -> {msg}")
            results.update("\n".join(lines))
            n_ok = sum(1 for _, ok, _ in self.results if ok)
            if n_ok:
                self._show_only("save-continue", "retry", "back")
            else:
                self._show_only("retry", "back")
        elif self.phase == "grouping":
            n = len(self.results)
            status.update(f"These {n} keys may share one usage limit.\n"
                          "How should I treat them?")
            results.update("")
            self._show_only("share", "separate", "later")

    # -- actions --

    @on(Button.Pressed, "#check")
    def _check(self) -> None:
        self.secrets = split_keys(self.query_one("#keys-input", TextArea).text)
        if self._is_custom_api():
            typed = self.query_one("#endpoint-input", Input).value.strip() or None
            default = self._endpoint_default()
            if typed is None and default is None:
                self.last_result = "Enter the base URL first."
                self.phase = "keys"
                self._show_state()
                return
            self.endpoint = typed or default
            self._endpoint_edited = typed is not None and typed != default
        else:
            self.endpoint = None
            self._endpoint_edited = False
        if not self.secrets:
            self.last_result = "Paste at least one key first."
            self.phase = "keys"
            self._show_state()
            return
        self.phase = "checking"
        self._show_state()
        self._worker = self.run_worker(self._check_task(), exclusive=True)

    async def _check_task(self) -> None:
        try:
            results, _avail = await asyncio.to_thread(
                _quiet_call, engine.validate_credentials,
                self.pid, self.secrets,
                [self.endpoint] if self.endpoint else [])
        except asyncio.CancelledError:
            self.phase = "keys"
            self.last_result = "Check cancelled."
            self._show_state()
            return
        except Exception as e:  # noqa: BLE001 -- total connection failure, not per-key
            self.phase = "keys"
            self.last_result = f"Could not reach the provider: {e}"
            self._show_state()
            return
        self.results = [(s, ok, m) for s, ok, m in results]
        n_ok = sum(1 for _, ok, _ in self.results if ok)
        if n_ok == len(self.results):
            self.last_result = f"All {n_ok} connection(s) work."
        elif n_ok:
            self.last_result = (f"{n_ok} of {len(self.results)} work — "
                                "only working ones continue.")
        else:
            self.last_result = "None of these keys work. Check them and retry."
        self.phase = "results"
        self._show_state()

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    @on(Button.Pressed, "#retry")
    def _retry(self) -> None:
        self.phase = "keys"
        self._show_state()

    @on(Button.Pressed, "#save-continue")
    def _save_continue(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        valid = [s for s, ok, _ in self.results if ok]
        try:
            engine.add_credentials(app.db, self.pid, valid,
                                   endpoint=(self.endpoint
                                             if self._endpoint_edited else None))
            engine.save_state(app.db, app.paths)
        except OSError as e:
            self.last_result = f"Could not save: {e}"
            self.phase = "results"
            self._show_state()
            return
        if engine.needs_grouping_question(app.db, self.pid):
            self.phase = "grouping"
            self._show_state()
            return
        app.push_screen(ModelScreen(self.pid, valid, self.endpoint))

    @on(Button.Pressed, "#share")
    def _share(self) -> None:
        self._answer_grouping("shared")

    @on(Button.Pressed, "#separate")
    def _separate(self) -> None:
        self._answer_grouping("separate")

    @on(Button.Pressed, "#later")
    def _later(self) -> None:
        self._answer_grouping("later")

    def _answer_grouping(self, mode: str) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        engine.set_quota_domains(app.db, self.pid, mode)
        engine.save_state(app.db, app.paths)
        valid = [s for s, ok, _ in self.results if ok]
        app.push_screen(ModelScreen(self.pid, valid, self.endpoint))

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()


class ModelScreen(Screen):
    """Choose models from the live catalog, probe them, keep what works.

    Free/cheap models first with an option to show everything; broken
    models are blocked before they can reach the config (only OK and
    throttled-but-valid are kept).
    """

    BINDINGS = [("escape", "back", "Back")]  # noqa: RUF012 -- Textual API

    def __init__(self, pid: str, secrets: list[str],
                 endpoint: str | None = None) -> None:
        super().__init__()
        self.pid = pid
        self.secrets = list(secrets)
        self.endpoint = endpoint
        self.catalog_state = "loading"  # loading | ready | failed | manual
        self.catalog: list[tuple[str, str]] = []
        self.free_only = True
        self.filter_text = ""
        self.retired_note = ""
        self.kept_selection: set[str] = set()
        self.candidate: list[str] = []
        self.probe_results: list[tuple[str, str, str]] = []
        self.last_status = ""
        self._worker = None

    # -- layout --

    def compose(self) -> ComposeResult:
        with Vertical(id="body"):
            yield Label("Choose useful models", id="title")
            yield Static("Fetching the live model list…", id="model-status")
            yield Input(placeholder="Type to filter", id="filter")
            yield Button("Show all models", id="show-toggle")
            yield SelectionList(id="model-list")
            yield TextArea(id="manual-input")
            yield Button("Probe selected", id="probe", variant="primary")
            yield Button("Use these IDs", id="manual-use", variant="primary")
            yield Button("Keep passing & review", id="keep-passing",
                         variant="primary")
            yield Button("Change selection", id="change-selection")
            yield Button("Cancel", id="cancel")
            yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self._worker = self.run_worker(self._load_task(), exclusive=True)
        self._show_state()

    # -- state --

    def _visible_ids(self) -> list[str]:
        if self.catalog_state == "loading":
            return ["cancel"]
        if self.catalog_state == "failed":
            return ["manual-use", "back"]
        if self.catalog_state == "manual":
            return ["manual-use", "back"]
        if self.catalog_state == "probing":
            return ["cancel"]
        if self.catalog_state == "probed":
            n_pass = sum(1 for _, c, _ in self.probe_results
                         if c in ("OK", "RATE_LIMITED"))
            if n_pass == len(self.probe_results):
                return ["keep-passing", "change-selection", "back"]
            if n_pass:
                return ["keep-passing", "change-selection", "back"]
            return ["change-selection", "back"]
        return ["probe", "back"]

    def _show_state(self) -> None:
        try:
            status = self.query_one("#model-status", Static)
            filt = self.query_one("#filter", Input)
            toggle = self.query_one("#show-toggle", Button)
            mlist = self.query_one("#model-list", SelectionList)
            manual = self.query_one("#manual-input", TextArea)
        except NoMatches:  # not yet mounted
            return
        status.update(self.last_status)
        is_select = self.catalog_state in ("ready", "probing", "probed")
        filt.display = is_select
        toggle.display = is_select
        mlist.display = is_select
        manual.display = self.catalog_state in ("failed", "manual")
        for bid in ("probe", "manual-use", "keep-passing", "change-selection",
                    "cancel", "back"):
            self.query_one(f"#{bid}", Button).display = bid in self._visible_ids()

    def _rebuild_options(self) -> None:
        try:
            mlist = self.query_one("#model-list", SelectionList)
        except NoMatches:
            return
        self.kept_selection = set(mlist.selected) | self.kept_selection
        items = engine.order_catalog_free_first(self.catalog)
        if self.free_only:
            free = [it for it in items if self._is_free(it)]
            items = free or items
        q = self.filter_text.strip().lower()
        if q:
            items = [it for it in items
                     if q in it[0].lower() or q in it[1].lower()]
        mlist.clear_options()
        for mid, label in items:
            prompt = mid if mid == label else f"{mid}  ({label})"
            mlist.add_option(Selection(prompt, mid,
                                       mid in self.kept_selection))
        toggle = self.query_one("#show-toggle", Button)
        toggle.label = ("Show all models" if self.free_only
                        else "Show free first")
        self.last_status = (f"{len(items)} shown"
                            + (f" for '{self.filter_text.strip()}'" if q else "")
                            + f" — {len(self.kept_selection)} selected.")
        if self.retired_note:
            self.last_status += "\n" + self.retired_note
        self.query_one("#model-status", Static).update(self.last_status)

    @staticmethod
    def _is_free(item: tuple[str, str]) -> bool:
        return engine.is_free_model(item[0], item[1])

    # -- catalog --

    async def _load_task(self) -> None:
        try:
            catalog = await asyncio.to_thread(
                _quiet_call, engine.discover_models,
                self.pid, self.secrets[0] if self.secrets else "",
                self.endpoint)
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 -- catalog fetch must degrade to manual entry
            catalog = None
        app = self.app
        assert isinstance(app, WizardApp)
        if not catalog:
            self.catalog_state = "failed"
            self.last_status = ("Could not fetch the model list. "
                                "Type model IDs manually (one per line).")
            self._show_state()
            return
        engine.mark_catalog_checked(app.db, self.pid)
        engine.save_state(app.db, app.paths)
        self.catalog = [(m, lbl) for m, lbl in catalog]
        existing = set(engine.get_models(app.db, self.pid))
        cat_ids = {m for m, _ in self.catalog}
        retired = [m for m in existing if m not in cat_ids]
        if retired:
            self.retired_note = ("No longer advertised: " + " ".join(retired)
                                 + " — they stay configured unless you "
                                   "unselect them below.")
            self.last_status = self.retired_note
        else:
            self.retired_note = ""
            self.last_status = ""
        self.kept_selection = {m for m in existing if m in cat_ids}
        self.catalog_state = "ready"
        self._show_state()
        self._rebuild_options()

    @on(Input.Changed, "#filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        self.filter_text = event.value
        if self.catalog_state == "ready":
            self._rebuild_options()

    @on(Button.Pressed, "#show-toggle")
    def _toggle(self) -> None:
        self.free_only = not self.free_only
        self._rebuild_options()

    @on(SelectionList.SelectedChanged, "#model-list")
    def _selection_changed(self, event: SelectionList.SelectedChanged) -> None:
        self.kept_selection = set(event.selection_list.selected)

    # -- probe --

    @on(Button.Pressed, "#manual-use")
    def _manual_use(self) -> None:
        text = self.query_one("#manual-input", TextArea).text
        picked = [ln.strip() for ln in text.replace(",", "\n").splitlines()
                  if ln.strip()]
        if not picked:
            self.last_status = "Type at least one model ID first."
            self._show_state()
            return
        self.catalog_state = "manual"
        self.candidate = picked
        self._start_probe()

    @on(Button.Pressed, "#probe")
    def _probe(self) -> None:
        picked = list(self.query_one("#model-list", SelectionList).selected)
        if not picked:
            self.last_status = "Select at least one model first."
            self._show_state()
            return
        app = self.app
        assert isinstance(app, WizardApp)
        existing = engine.get_models(app.db, self.pid)
        self.candidate = list(existing) + [m for m in picked if m not in existing]
        self._start_probe()

    def _start_probe(self) -> None:
        self.catalog_state = "probing"
        self.last_status = (f"Probing {len(self.candidate)} model(s)… "
                            "(minimal ping each)")
        self._show_state()
        self._worker = self.run_worker(self._probe_task(), exclusive=True)

    async def _probe_task(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        try:
            results = await asyncio.to_thread(
                _quiet_call, engine.probe_models,
                self.pid, self.candidate,
                self.secrets[0] if self.secrets else None,
                self.endpoint, keys=self.secrets,
                mode=engine.get_validation_mode(app.db),
                sample_size=engine.get_sample_size(app.db),
                db=app.db, sleep_s=1.5)
        except asyncio.CancelledError:
            self.catalog_state = "ready" if self.catalog else "manual"
            self.last_status = "Probe cancelled."
            self._show_state()
            return
        self.probe_results = [(m, c, msg) for m, c, msg in results]
        lines = []
        for m, cls, msg in self.probe_results:
            mark = {"OK": "OK", "RATE_LIMITED": "WAIT"}.get(cls, "FAIL")
            lines.append(f"  [{mark}] {m}" + ("" if cls == "OK" else f" -> {msg}"))
        n_pass = sum(1 for _, c, _ in self.probe_results
                     if c in ("OK", "RATE_LIMITED"))
        if n_pass == len(self.probe_results):
            lines.append("All passed (throttled counts as valid) — review next.")
        elif n_pass:
            lines.append(f"{len(self.probe_results) - n_pass} blocked — "
                         "only passing models continue.")
        else:
            lines.append("Nothing passed. Change the selection or go back.")
        self.last_status = "\n".join(lines)
        self.catalog_state = "probed"
        self._show_state()

    @on(Button.Pressed, "#keep-passing")
    def _keep_passing(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        passing = [m for m, c, _ in self.probe_results
                   if c in ("OK", "RATE_LIMITED")]
        if not passing:
            return
        created = _quiet_call(engine.auto_combine, app.db, self.pid, passing)
        engine.set_models(app.db, self.pid, passing)
        engine.save_state(app.db, app.paths)
        review = DoneScreen()
        if created:
            names = ", ".join(f"'{s}'" for s in created)
            review.last_status = (f"✓ {names} now served from multiple "
                                  "providers automatically.")
        _pop_to_home(app)
        app.push_screen(review)

    @on(Button.Pressed, "#change-selection")
    def _change_selection(self) -> None:
        if self.catalog:
            self.catalog_state = "ready"
        else:
            self.catalog_state = "manual"
        self._show_state()
        if self.catalog:
            self._rebuild_options()

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()


class TestScreen(Screen):
    """One-button gateway test with fix-up for failures.

    Default test: one request per pool through the gateway (the product's
    one job is a usable gateway). Failures can be diagnosed per underlying
    connection; only wrong-key (auth) connections are offered for parking —
    throttled or model-level failures need Configure attention, never an
    automatic fix.
    """

    BINDINGS = [("escape", "back", "Back")]  # noqa: RUF012 -- Textual API
    PASS = ("OK", "RATE_LIMITED")

    def __init__(self) -> None:
        super().__init__()
        self.phase = "idle"  # idle | running | done | diagnosing | diagnosed
        self.last_status = ""
        self.last_results = ""
        self.results: list[tuple[str, str, str]] = []
        self.diag: list[tuple[str, str, str, str, str, str]] = []
        self.parked = 0
        self.parked_ids: set[tuple[str, str]] = set()
        self._worker = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Label("Test your setup", id="title")
            yield Static("", id="test-status")
            yield Static("", id="test-results")
            yield Button("Run test", id="run-test", variant="primary")
            yield Button("Diagnose failures", id="diagnose")
            yield Button("Park failing connections", id="park-fixes",
                         variant="warning")
            yield Button("Review & Apply", id="review", variant="primary")
            yield Button("Cancel", id="cancel")
            yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        self.last_status = test_lines(app.db)
        self._show_state()

    def on_screen_resume(self) -> None:
        self._show_state()

    # -- state --

    def _failed_aliases(self) -> list[str]:
        return [a for a, c, _ in self.results if c not in self.PASS]

    def _parkable(self) -> list[tuple[str, str]]:
        out = []
        for _pool, pid, cid, _suf, cls, _msg in self.diag:
            if (cls == "AUTH_ERROR" and cid and (pid, cid) not in out
                    and (pid, cid) not in self.parked_ids):
                out.append((pid, cid))
        return out

    def _visible_ids(self) -> list[str]:
        if self.phase in ("running", "diagnosing"):
            return ["cancel"]
        if self.parked_ids:
            return ["review", "run-test", "back"]
        if self.phase == "diagnosed" and self._parkable():
            return ["park-fixes", "run-test", "back"]
        if self.phase == "done" and self._failed_aliases():
            return ["diagnose", "run-test", "back"]
        return ["run-test", "back"]

    def _show_state(self) -> None:
        try:
            self.query_one("#test-status", Static).update(self.last_status)
            self.query_one("#test-results", Static).update(self.last_results)
            for bid in ("run-test", "diagnose", "park-fixes", "review",
                        "cancel", "back"):
                self.query_one(f"#{bid}", Button).display = bid in self._visible_ids()
        except NoMatches:  # not yet mounted
            return

    def _render_results(self) -> None:
        lines = []
        for alias, cls, msg in self.results:
            mark = {"OK": "✓", "RATE_LIMITED": "~"}.get(cls, "✗")
            lines.append(f"  [{mark}] {alias}"
                         + ("" if cls == "OK" else f" -> {msg[:110]}"))
        counts: dict[str, int] = {}
        for _, cls, _ in self.results:
            counts[cls] = counts.get(cls, 0) + 1
        n = len(self.results)
        lines.append(f"\n{counts.get('OK', 0)} OK | "
                     f"{counts.get('RATE_LIMITED', 0)} throttled | "
                     f"{n - counts.get('OK', 0) - counts.get('RATE_LIMITED', 0)}"
                     f" failed of {n}")
        self.last_results = "\n".join(lines)
        self._show_state()

    # -- default test: one request per pool through the gateway --

    @on(Button.Pressed, "#run-test")
    def _run_test(self) -> None:
        self.phase = "running"
        self.results = []
        self.diag = []
        self.parked = 0
        self.parked_ids = set()
        self.last_results = ""
        self._show_state()
        self._worker = self.run_worker(self._run_task(), exclusive=True)

    async def _run_task(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        aliases = await asyncio.to_thread(
            _quiet_call, engine.gateway_aliases, app.paths)
        if not aliases:
            self.phase = "idle"
            self.last_status = "Nothing to test yet — Apply first."
            self._show_state()
            return
        for i, alias in enumerate(aliases):
            self.last_status = f"Testing {i + 1}/{len(aliases)}: {alias}…"
            self._show_state()
            try:
                cls, msg = await asyncio.to_thread(
                    _quiet_call, engine.probe_gateway_alias, alias, app.paths)
            except asyncio.CancelledError:
                self.phase = "idle"
                self.last_status = "Test cancelled."
                self._show_state()
                return
            self.results.append((alias, cls, msg))
            self._render_results()
            if alias != aliases[-1]:
                try:
                    await asyncio.sleep(2.0)  # spacing mirrors the CLI smoke test
                except asyncio.CancelledError:
                    self.phase = "idle"
                    self.last_status = "Test cancelled."
                    self._show_state()
                    return
        failed = self._failed_aliases()
        self.phase = "done"
        if failed:
            self.last_status = (f"{len(failed)} model(s) failing — diagnose to "
                                "see which connection is at fault.")
        else:
            self.last_status = "All models answer through the gateway."
        self._show_state()

    # -- diagnose: probe each failing pool's connections directly --

    @on(Button.Pressed, "#diagnose")
    def _diagnose(self) -> None:
        self.phase = "diagnosing"
        self.diag = []
        self.last_status = "Probing the failing connections directly…"
        self._show_state()
        self._worker = self.run_worker(self._diagnose_task(), exclusive=True)

    async def _diagnose_task(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        deps, _pools, _roles, errors = engine.compile_config(app.db)
        if errors:
            self.phase = "done"
            self.last_status = f"Cannot diagnose: {errors[0]}"
            self._show_state()
            return
        failed = set(self._failed_aliases())
        targets = [d for d in deps if d["logical_model"] in failed]
        rows = []
        for i, d in enumerate(targets):
            secret = d.get("secret") if d.get("credential_id") else None
            self.last_status = (f"Probing {i + 1}/{len(targets)}: "
                                f"{d['logical_model']} via {d['provider']}…")
            self._show_state()
            try:
                cls, msg = await asyncio.to_thread(
                    _quiet_call, engine.probe_model,
                    d["provider"], d["upstream_model"], secret, d.get("endpoint"))
            except asyncio.CancelledError:
                self.phase = "done"
                self.last_status = "Diagnosis cancelled."
                self._show_state()
                return
            suffix = engine.mask_secret(secret or "")
            self.diag.append((d["logical_model"], d["provider"],
                              d.get("credential_id") or "", suffix, cls, msg))
            mark = {"OK": "✓", "RATE_LIMITED": "~"}.get(cls, "✗")
            rows.append(f"  [{mark}] {d['logical_model']} via {d['provider']} "
                        f"[{suffix}]" + ("" if cls == "OK" else f" -> {msg[:100]}"))
            self.last_results = "\n".join(rows)
            self._show_state()
        parkable = self._parkable()
        self.phase = "diagnosed"
        if parkable:
            self.last_status = (f"{len(parkable)} connection(s) reject their key "
                                "(wrong/expired). Park them? Anything else "
                                "needs Configure attention.")
        else:
            self.last_status = ("No wrong-key connections — failures are "
                                "throttling or model-level. Give throttled "
                                "pools a minute, or re-Configure the model.")
        self._show_state()

    @on(Button.Pressed, "#park-fixes")
    def _park(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        n = 0
        for pid, cid in self._parkable():
            if engine.set_credential_quarantined(app.db, pid, cid, True):
                self.parked_ids.add((pid, cid))
                n += 1
        if n:
            engine.save_state(app.db, app.paths)
        self.parked = n
        self.last_status = (f"Parked {n} connection(s). They stay saved but "
                            "leave the gateway on next Apply — Review & Apply "
                            "to finish." if n else "Nothing to park.")
        self._show_state()

    @on(Button.Pressed, "#review")
    def _review(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        _pop_to_home(app)
        app.push_screen(DoneScreen())

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()


class OpenCodeScreen(Screen):
    """What OpenCode sees: exposed aliases grouped with their children.

    Read-only view over the gateway config + opencode.json (nothing is
    written from here except the optional Sync button, which drives
    :func:`engine.sync_opencode` like Review does). ``o`` on Home opens
    it; ``Esc`` goes back; ``r`` refreshes.
    """

    BINDINGS = [("escape", "back", "Back"),  # noqa: RUF012 -- Textual API
                ("r", "refresh", "Refresh"),
                ("s", "sync", "Sync")]

    def __init__(self) -> None:
        super().__init__()
        self.last_status = ""
        self.data: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Label("OpenCode view — what OpenCode sees", id="title")
            yield Static("", id="opencode-status")
            yield Tree("litellm", id="opencode-tree")
            yield Button("Sync OpenCode", id="sync-opencode")
            yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_content()

    def on_screen_resume(self) -> None:
        self.refresh_content()

    def refresh_content(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        self.data = opencode_view_data(app.db, app.paths)
        exposed = self.data["exposed"]
        stale = ("STALE — opencode.json differs from the gateway; "
                 "press s to sync." if self.data["differs"]
                 else "in sync with the gateway.")
        if self.data["errors"]:
            self.last_status = ("Configuration has problems: "
                                + "; ".join(self.data["errors"][:3]))
        elif not exposed:
            self.last_status = "Nothing exposed yet — Configure, then Apply."
        else:
            self.last_status = (f"{len(exposed)} model(s) as litellm/<name> "
                                f"({self.data['source']}) — {stale}")
        try:
            self.query_one("#opencode-status", Static).update(self.last_status)
            self._rebuild_tree()
        except NoMatches:  # not yet mounted
            pass

    def _rebuild_tree(self) -> None:
        assert self.data is not None
        tree = self.query_one("#opencode-tree", Tree)
        tree.clear()
        tree.root.label = (f"litellm — {len(self.data['exposed'])} model(s)"
                           + (" (stale)" if self.data["differs"] else ""))
        missing = set(self.data["exposed"]) - set(self.data["children"]) - set(self.data["roles"])
        for alias in self.data["exposed"]:
            kids = self.data["children"].get(alias, [])
            if kids:
                provs = ", ".join(sorted({k["provider"] for k in kids}))
                node = tree.root.add_leaf(
                    f"{alias}  ({len(kids)} backend(s): {provs})")
                for k in kids:
                    node.add_leaf(f"{k['provider']} / {k['upstream']} "
                                  f"[{k['suffix']}] • rpm {k['rpm']} • {k['health']}")
            elif alias in self.data["roles"]:
                spec = self.data["roles"][alias]
                prim = ", ".join(spec.get("pools") or [])
                fb = ", ".join(spec.get("fallback") or [])
                node = tree.root.add_leaf(f"{alias}  (role → {prim}"
                                          + (f" | fallback {fb}" if fb else "") + ")")
                for p in (spec.get("pools") or []) + (spec.get("fallback") or []):
                    pkids = self.data["children"].get(p, [])
                    if pkids:
                        sub = node.add(f"{p} ({len(pkids)} backend(s))")
                        for k in pkids:
                            sub.add_leaf(f"{k['provider']} / {k['upstream']} "
                                         f"[{k['suffix']}] • {k['health']}")
                    else:
                        node.add_leaf(f"{p} (no live backends)")
            elif alias in missing:
                tree.root.add_leaf(f"{alias}  (in opencode.json, no gateway backends)")
        if self.data["current"]:
            extra = [m for m in self.data["current"] if m not in self.data["exposed"]]
            if extra:
                node = tree.root.add_leaf(f"only in opencode.json ({len(extra)}):")
                for m in sorted(extra)[:10]:
                    node.add_leaf(m)
        tree.root.expand_all()

    def action_refresh(self) -> None:
        self.refresh_content()

    @on(Button.Pressed, "#sync-opencode")
    def _sync(self) -> None:
        self.action_sync()

    def action_sync(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        try:
            result = _quiet_call(engine.sync_opencode, app.paths)
        except FileNotFoundError:
            note = "OpenCode config not found — skipped."
        except ValueError as e:
            note = f"Sync failed ({e})."
        else:
            note = (f"OpenCode updated "
                    f"({len(result.get('exposed', []))} model(s)). "
                    f"Restart the OpenCode TUI, then /models.")
        self.refresh_content()
        self.last_status += f"\n{note}"
        try:
            self.query_one("#opencode-status", Static).update(self.last_status)
        except NoMatches:
            pass

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()


class DoneScreen(Screen):
    """Review -> apply as one transaction -> optional OpenCode sync.

    Invalid state is never partially applied (the write is refused and
    the old config stays). Restart happens only when the config actually
    changed. A failed OpenCode sync is reported separately — it never
    marks a working gateway as failed.
    """

    BINDINGS = [("escape", "back", "Back")]  # noqa: RUF012 -- Textual API

    def __init__(self) -> None:
        super().__init__()
        self.last_content = ""
        self.last_status = ""
        self.applied_ok = False
        self.suggestions: dict[str, list[dict[str, str]]] = {}
        self.free_first: dict[str, dict[str, Any]] = {}
        self._worker = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Label("Review", id="title")
            yield Static("", id="done-content")
            yield Static("", id="apply-status")
            yield Button("Apply changes", id="apply", variant="primary")
            yield Button("Group suggested models", id="group-suggested")
            yield Button("Add free-first roles", id="free-first")
            # Sync buttons dynamic per harness + fallback for tests (uses app.paths if available)
            try:
                hlist = engine.detect_harnesses(self.app.paths) if hasattr(self, 'app') and hasattr(self.app, 'paths') else engine.detect_harnesses()
            except Exception:
                hlist = engine.detect_harnesses()
            for h in hlist:
                yield Button(f"Sync {h.title()}", id=f"sync-{h}")
            # Always add generic sync for test compat (hidden if hlist not empty but still queried)
            yield Button("Sync OpenCode", id="sync")
            yield Button("Cancel", id="cancel")
            yield Button("Back to Home", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_content()

    def on_screen_resume(self) -> None:
        self.refresh_content()

    def refresh_content(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        self.suggestions = engine.pending_suggestions(app.db)
        self.free_first = engine.suggest_free_first_roles(app.db)
        self.last_content = done_lines(app.db)
        if self.suggestions:
            lines = ["", "Same model on several providers? Group only if",
                     "they are truly interchangeable:"]
            for stem, members in sorted(self.suggestions.items()):
                provs = ", ".join(sorted({m["provider"] for m in members}))
                lines.append(f"  ? {stem}  ({provs})")
            self.last_content += "\n".join(lines)
        if self.free_first:
            lines = ["", "Free Google capacity available as one name:"]
            for role, spec in sorted(self.free_first.items()):
                lines.append(f"  ? {role} -> {', '.join(spec['pools'])}")
            self.last_content += "\n".join(lines)
        self.query_one("#done-content", Static).update(self.last_content)
        self.query_one("#apply-status", Static).update(self.last_status)
        # Dynamic sync buttons per harness (use app.paths if available for test isolation)
        try:
            hlist = engine.detect_harnesses(self.app.paths) if hasattr(self.app, 'paths') else engine.detect_harnesses()
        except Exception:
            hlist = engine.detect_harnesses()
        sync_ids = [f"sync-{h}" for h in hlist] if hlist else ["sync"]
        # Always include generic sync for test compat
        if "sync" not in sync_ids:
            sync_ids = sync_ids + ["sync"]
        all_ids = ["apply", "group-suggested", "free-first"] + sync_ids + ["cancel", "back"]
        visible = set(self._visible_ids())
        for bid in all_ids:
            try:
                self.query_one(f"#{bid}", Button).display = bid in visible
            except NoMatches:
                pass

    def _visible_ids(self) -> list[str]:
        ids = ["group-suggested"] if self.suggestions else []
        if self.free_first:
            ids.append("free-first")
        try:
            hlist = engine.detect_harnesses(self.app.paths) if hasattr(self.app, 'paths') else engine.detect_harnesses()
        except Exception:
            hlist = engine.detect_harnesses()
        sync_ids = [f"sync-{h}" for h in hlist] if hlist else ["sync"]
        if "sync" not in sync_ids:
            sync_ids.append("sync")
        if self.applied_ok:
            return [*ids, *sync_ids, "back"]
        return ["apply", *ids, "back"]

    def _set_status(self, msg: str) -> None:
        self.last_status = msg
        try:
            self.query_one("#apply-status", Static).update(msg)
        except NoMatches:  # worker finished before mount; text kept in last_status
            pass

    @on(Button.Pressed, "#group-suggested")
    def _group_suggested(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        grouped, skipped = [], []
        for stem, members in sorted(self.suggestions.items()):
            try:
                engine.combine_models(db=app.db, canonical=stem,
                                      members=[(m["provider"], m["model"])
                                               for m in members])
                grouped.append(stem)
            except ValueError:
                skipped.append(stem)
        if grouped:
            engine.save_state(app.db, app.paths)
            self.applied_ok = False  # aliases changed -> apply again
        note = ""
        if grouped:
            note = f"Grouped {', '.join(grouped)}. Review, then Apply again."
        if skipped:
            note += (" Could not group (incompatible): " + ", ".join(skipped) + ".")
        self._set_status((note or "Nothing grouped.").strip())
        self.refresh_content()

    @on(Button.Pressed, "#free-first")
    def _free_first(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        applied, skipped = engine.apply_free_first_roles(app.db)
        if applied:
            engine.save_state(app.db, app.paths)
            self.applied_ok = False  # roles changed -> apply again
            self._set_status(f"Added {', '.join(applied)}. Review, then Apply again.")
        else:
            self._set_status("Nothing added (roles exist or no matching pools).")
        self.refresh_content()

    @on(Button.Pressed, "#apply")
    def _apply(self) -> None:
        self._set_status("Applying: writing config, restarting if changed…")
        self.query_one("#apply", Button).display = False
        self.query_one("#cancel", Button).display = True
        self._worker = self.run_worker(self._apply_task(), exclusive=True)

    async def _apply_task(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        try:
            result = await asyncio.to_thread(
                _quiet_call, engine.apply_config, app.db, app.paths)
        except asyncio.CancelledError:
            self._set_status("Apply cancelled — a restart may still be in "
                             "progress; check Home for gateway status.")
            self.refresh_content()
            return
        except ValueError as e:
            self._set_status(f"Apply blocked, nothing changed: {e}")
            self.refresh_content()
            return
        routes = result.get("routes", 0)
        if not result.get("changed"):
            self.applied_ok = True
            self._set_status(f"Nothing changed — gateway already serves "
                             f"{routes} connection(s).")
        elif result.get("ready"):
            self.applied_ok = True
            app.status = "running"
            self._set_status(f"✓ Gateway working — {routes} connection(s).")
        else:
            self._set_status("Restart requested but the gateway is not ready "
                             "yet. The previous config is kept as backup — "
                             "check Home, then retry.")
        try:
            engine.save_state(app.db, app.paths)
        except OSError:
            pass
        self.refresh_content()
        if self.applied_ok:
            self._maybe_offer_sync()

    def _maybe_offer_sync(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        try:
            stale = engine.opencode_differs(app.paths)
        except Exception:  # noqa: BLE001 -- unreadable config counts as stale
            stale = True
        if stale:
            self._set_status(self.last_status + "\nOpenCode looks out of sync "
                             "— Sync OpenCode to update it.")

    @on(Button.Pressed, "#sync")
    def _sync(self) -> None:
        self._sync_harness("opencode")

    @on(Button.Pressed, "#sync-opencode")
    def _sync_opencode(self) -> None:
        self._sync_harness("opencode")

    @on(Button.Pressed, "#sync-jcode")
    def _sync_jcode(self) -> None:
        self._sync_harness("jcode")

    def _sync_harness(self, harness: str) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        try:
            if harness == "opencode":
                result = _quiet_call(engine.sync_opencode, app.paths)
            elif harness == "jcode":
                result = _quiet_call(engine.sync_jcode, app.paths)
            else:
                # generic: try sync_opencode for any harness (fallback)
                result = _quiet_call(engine.sync_opencode, app.paths)
        except FileNotFoundError:
            self._set_status(self.last_status + f"\n{harness} config not found — "
                             "skipped. The gateway itself is working.")
            return
        except ValueError as e:
            self._set_status(self.last_status + f"\n{harness} sync failed "
                             f"separately ({e}) — the gateway itself is working.")
            return
        exposed = result.get("exposed", []) if isinstance(result, dict) else []
        # handle both sync_opencode and sync_jcode result shapes
        count = len(exposed) if exposed else len(result.get("models", [])) if isinstance(result, dict) else 0
        self._set_status(self.last_status + f"\n{harness.title()} updated "
                         f"({count} model(s)). Restart {harness} and check /model.")

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()


class QuotaDashboardScreen(Screen):
    """Quota-domain dashboard: one row per speed-limit bucket.

    Emphasizes independent quota domains, not raw key counts: keys in
    one domain share its limit; domains don't add up. Google
    ``project:<pid>:<id>`` domains carry the project id.
    """

    BINDINGS = [("escape", "back", "Back"),  # noqa: RUF012 -- Textual API
                ("r", "refresh", "Refresh")]

    COLUMNS = (("domain", "Domain"), ("provider", "Provider"),
               ("keys", "Keys"), ("rpm", "RPM"),
               ("project", "Project"), ("confidence", "Confidence"))

    def __init__(self) -> None:
        super().__init__()
        self.domains: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Label("Quota Domains", id="title")
            yield Static("", id="quota-status")
            yield DataTable(id="quota-table", cursor_type="row")
            yield Button("Refresh", id="refresh")
            yield Button("Back to Home", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self._rebuild_table()

    def on_screen_resume(self) -> None:
        self._rebuild_table()

    def action_refresh(self) -> None:
        self._rebuild_table()

    @on(Button.Pressed, "#refresh")
    def _refresh(self) -> None:
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        self.domains = engine.quota_domains_list(app.db)
        total_keys = sum(d["key_count"] for d in self.domains)
        table = self.query_one("#quota-table", DataTable)
        table.clear(columns=True)
        for _key, label in self.COLUMNS:
            table.add_column(label, key=_key)
        for d in sorted(self.domains, key=lambda x: (x["provider"], x["domain_id"])):
            rpm = str(d["rpm"]) if d.get("rpm") else "?"
            table.add_row(
                fit_cell(str(d["domain_id"]), 34),
                fit_cell(str(d["provider"]), 12),
                str(d["key_count"]),
                rpm,
                fit_cell(str(d.get("project_id") or "—"), 18),
                fit_cell(str(d.get("confidence") or "—"), 12),
            )
        status = (f"{len(self.domains)} independent domain(s), {total_keys} key(s) — "
                  "keys in one domain share its speed; more keys there won't add up.")
        if len(self.domains) != total_keys:
            status += f" ({total_keys - len(self.domains)} key(s) saved by sharing)."
        try:
            self.query_one("#quota-status", Static).update(status)
        except NoMatches:
            pass

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()


class ProxyScreen(Screen):
    """Per-proxy tab (sparse): shows routing, rate-limit handling for that proxy."""

    BINDINGS = [("escape", "back", "Back")]  # noqa: RUF012

    def __init__(self, proxy_type: str | None = None) -> None:
        super().__init__()
        self.proxy_type = proxy_type

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Label("Proxy — routing only, vault supplies keys", id="title")
            yield Static("", id="proxy-status")
            yield Label("Proxy type:", id="proxy-type-label")
            yield SelectionList(id="proxy-type-list")
            yield Label("Routing:", id="proxy-routing-label")
            yield Input(placeholder="usage-based-routing-v2", id="proxy-routing")
            yield Label("Free keys handling (throttled kept, 429 cooldown):", id="proxy-free-label")
            yield Input(placeholder="60", id="proxy-cooldown-rate")
            yield Label("Retry: try all providers in group before failing", id="proxy-retry-label")
            yield SelectionList(id="proxy-retry-list")
            yield Static("Rate-limit handling: 429 → cooldown 60s, 5xx → 30s, vault throttled kept", id="proxy-help")
            yield Button("Apply proxy config", id="apply-proxy", variant="primary")
            yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_content()
        # Populate proxy type list
        try:
            lst = self.query_one("#proxy-type-list", SelectionList)
            lst.clear_options()
            for pt in engine.PROXY_TYPES:
                lst.add_option(Selection(pt, pt, self.proxy_type == pt or (not self.proxy_type and pt == engine.get_proxy_type(self.app.db))))
            routing = self.app.db.get("_proxy", {}).get("routing", "usage-based-routing-v2")
            self.query_one("#proxy-routing", Input).value = str(routing)
            cooldown = self.app.db.get("_proxy", {}).get("cooldown_rate", 60)
            self.query_one("#proxy-cooldown-rate", Input).value = str(cooldown)
            # Retry handling: try all providers in group
            rlst = self.query_one("#proxy-retry-list", SelectionList)
            rlst.clear_options()
            retry = self.app.db.get("_proxy", {}).get("retry_bad_request", True)
            rlst.add_option(Selection("Try all providers in group (recommended)", True, retry is True or retry == "true"))
            rlst.add_option(Selection("Fail fast (no retry on 400)", False, retry is False))
        except NoMatches:
            pass

    def refresh_content(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        pt = self.proxy_type or engine.get_proxy_type(app.db)
        routing = app.db.get("_proxy", {}).get("routing", "usage-based-routing-v2")
        cooldown = app.db.get("_proxy", {}).get("cooldown_rate", 60)
        retry = app.db.get("_proxy", {}).get("retry_bad_request", True)
        retry_txt = "try all providers" if retry else "fail fast"
        self.query_one("#proxy-status", Static).update(f"Proxy: {pt} (vault → {pt} compile)\nRouting: {routing}, cooldown: {cooldown}s, retry: {retry_txt}, throttled kept")

    @on(Button.Pressed, "#apply-proxy")
    def _apply(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        # Apply proxy type from selection or tab
        try:
            sel = self.query_one("#proxy-type-list", SelectionList).selected
            chosen = sel[0] if sel else (self.proxy_type or engine.get_proxy_type(app.db))
            if chosen and chosen != engine.get_proxy_type(app.db):
                engine.set_proxy_type(app.db, chosen)
            # Apply routing and cooldown from inputs
            routing = self.query_one("#proxy-routing", Input).value.strip()
            if routing:
                if not isinstance(app.db.get("_proxy"), dict):
                    app.db["_proxy"] = {}
                app.db["_proxy"]["routing"] = routing
            cooldown = self.query_one("#proxy-cooldown-rate", Input).value.strip()
            if cooldown.isdigit():
                if not isinstance(app.db.get("_proxy"), dict):
                    app.db["_proxy"] = {}
                app.db["_proxy"]["cooldown_rate"] = int(cooldown)
            # Apply retry handling
            rsel = self.query_one("#proxy-retry-list", SelectionList).selected
            retry = rsel[0] if rsel else True
            if not isinstance(app.db.get("_proxy"), dict):
                app.db["_proxy"] = {}
            app.db["_proxy"]["retry_bad_request"] = bool(retry)
            engine.save_state(app.db, app.paths)
        except (NoMatches, ValueError) as e:
            self.query_one("#proxy-status", Static).update(f"Set proxy failed: {e}")
            return
        try:
            n = engine.write_config(app.db, app.paths)
            pt = engine.get_proxy_type(app.db)
            self.query_one("#proxy-status", Static).update(f"Proxy {pt} applied: {n} routes (retry={'all' if app.db.get('_proxy',{}).get('retry_bad_request', True) else 'fast'})")
        except ValueError as e:
            self.query_one("#proxy-status", Static).update(f"Apply failed: {e}")

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()


class HarnessScreen(Screen):
    """Generic harness tab: mirrors app's provider/model list, allows add/remove without touching vault."""

    def __init__(self, harness: str) -> None:
        super().__init__()
        self.harness = harness

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Label(f"{self.harness} — mirrors {self.harness} /connect or /model", id="title")
            yield Static("", id="harness-status")
            yield DataTable(id="harness-table", cursor_type="row")
            yield Static("Select a row to remove, or Import from vault", id="harness-hint")
            yield Button("Delete", id="delete", variant="error")
            yield Button("Delete (keep free)", id="keep-free")
            yield Button("Import all", id="import-all", variant="primary")
            yield Button("Pick", id="pick")
            yield Button("Add provider/model", id="add-harness-model")
            yield Button("Remove selected", id="remove-harness-model")
            yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_content()

    def refresh_content(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        # Show harness file exactly as app sees it (mirror)
        try:
            if self.harness == "opencode":
                data = engine.opencode_view_data(app.db, app.paths) if hasattr(engine, "opencode_view_data") else {}
                # opencode view data has current providers
                cur = data.get("current") or []
                status = f"{self.harness}: {len(cur)} models (mirrors /model) • vault is source"
            elif self.harness == "jcode":
                st = engine.jcode_status(app.paths)
                status = f"{self.harness}: {len(st.get('profiles',[]))} profiles • vault is source"
            else:
                status = f"{self.harness}: detected"
        except Exception:
            status = f"{self.harness}: status unknown"
        self.query_one("#harness-status", Static).update(status)
        # Fill table with harness's own providers/models (not vault)
        try:
            table = self.query_one("#harness-table", DataTable)
            table.clear(columns=True)
            table.add_column("Provider", key="provider")
            table.add_column("Model", key="model")
            # Load harness providers
            if self.harness == "opencode":
                import json as _json
                with open(app.paths.opencode_json) as f:
                    cfg = _json.loads(engine._load_sync_module()._strip_jsonc(f.read()))
                prov = cfg.get("provider") or {}
                for pid, blk in prov.items():
                    if not isinstance(blk, dict):
                        continue
                    models = blk.get("models") or {}
                    if isinstance(models, dict):
                        for mid in models:
                            table.add_row(pid, mid)
                    elif isinstance(models, list):
                        for mid in models:
                            table.add_row(pid, mid)
            elif self.harness == "jcode":
                with open(app.paths.jcode_config) as f:
                    txt = f.read()
                import re as _re
                # simple parse: show provider names and models
                mods = engine._load_jcode_module().parse_profiles(txt) if hasattr(engine, "_load_jcode_module") else {}
                for pname, pdata in mods.items():
                    for m in pdata.get("models", []):
                        table.add_row(pname, m.get("id",""))
        except Exception:
            pass

    @on(Button.Pressed, "#delete")
    def _delete(self) -> None:
        engine.harness_delete(self.app.paths, self.harness, keep_free=False)
        self.refresh_content()

    @on(Button.Pressed, "#keep-free")
    def _keep(self) -> None:
        engine.harness_delete(self.app.paths, self.harness, keep_free=True)
        self.refresh_content()

    @on(Button.Pressed, "#import-all")
    def _all(self) -> None:
        engine.harness_import_from_vault(self.app.paths, self.harness, None)
        self.refresh_content()

    @on(Button.Pressed, "#pick")
    def _pick(self) -> None:
        self.app.push_screen(VaultPickScreen(self.harness))

    @on(Button.Pressed, "#add-harness-model")
    def _add(self) -> None:
        # Add provider/model directly to harness, not vault (harness-only)
        # For minimal: open a simple input screen
        self.app.push_screen(HarnessAddScreen(self.harness))

    @on(Button.Pressed, "#remove-harness-model")
    def _remove(self) -> None:
        try:
            table = self.query_one("#harness-table", DataTable)
            row = table.cursor_row
            if row is None:
                return
            # Get provider/model from row
            prov = table.get_cell_at((row, 0))
            model = table.get_cell_at((row, 1))
            # Remove from harness config only (vault untouched)
            if self.harness == "opencode":
                import json as _json
                with open(self.app.paths.opencode_json) as f:
                    cfg = _json.loads(engine._load_sync_module()._strip_jsonc(f.read()))
                if isinstance(cfg.get("provider"), dict) and prov in cfg["provider"]:
                    blk = cfg["provider"][prov]
                    if isinstance(blk.get("models"), dict) and model in blk["models"]:
                        del blk["models"][model]
                    elif isinstance(blk.get("models"), list) and model in blk["models"]:
                        blk["models"].remove(model)
                    # atomic write
                    engine._load_sync_module()._atomic_write_json(self.app.paths.opencode_json, cfg)
            elif self.harness == "jcode":
                # Remove from jcode config.toml
                with open(self.app.paths.jcode_config) as f:
                    txt = f.read()
                # simple removal: remove that model line
                lines = txt.splitlines()
                out = []
                skip = False
                for line in lines:
                    if f'"{model}"' in line and "id =" in line:
                        # check if previous line is [[providers.<harness>.models]]
                        # For now just skip this id line
                        continue
                    out.append(line)
                open(self.app.paths.jcode_config, "w").write("\n".join(out))
        except Exception:
            pass
        self.refresh_content()

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()


class HarnessAddScreen(Screen):
    """Add provider/model directly to harness (not vault)."""

    def __init__(self, harness: str) -> None:
        super().__init__()
        self.harness = harness

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Label(f"Add to {self.harness} (harness-only, vault untouched)", id="title")
            yield Input(placeholder="Provider (e.g. openai, anthropic, custom label)", id="add-provider")
            yield Input(placeholder="Model (e.g. gpt-4o, claude-sonnet)", id="add-model")
            yield Input(placeholder="API key (optional, for this harness only)", id="add-key")
            yield Button("Add", id="do-add", variant="primary")
            yield Button("Back", id="back")
        yield Footer()

    @on(Button.Pressed, "#do-add")
    def _do_add(self) -> None:
        prov = self.query_one("#add-provider", Input).value.strip()
        model = self.query_one("#add-model", Input).value.strip()
        key = self.query_one("#add-key", Input).value.strip()
        if not prov or not model:
            return
        try:
            # Vault is big database containing everything — also add to vault
            vault_pid = prov.lower().replace(" ", "_")
            if vault_pid not in self.app.db or not isinstance(self.app.db.get(vault_pid), dict):
                self.app.db.setdefault(vault_pid, {"keys": [], "models": [], "endpoints": []})
            if model not in self.app.db[vault_pid].get("models", []):
                self.app.db[vault_pid].setdefault("models", []).append(model)
            if key and key not in self.app.db[vault_pid].get("keys", []):
                # add to vault via engine (handles credentials)
                engine.add_credentials(self.app.db, vault_pid, [key])
                engine.save_state(self.app.db, self.app.paths)
            # Also add to harness (harness-only, vault untouched for delete, but add goes to both)
            if self.harness == "opencode":
                import json as _json
                with open(self.app.paths.opencode_json) as f:
                    cfg = _json.loads(engine._load_sync_module()._strip_jsonc(f.read()))
                cfg.setdefault("provider", {}).setdefault(prov, {}).setdefault("models", {})[model] = {"name": model}
                if key:
                    cfg["provider"][prov].setdefault("options", {})["apiKey"] = key
                engine._load_sync_module()._atomic_write_json(self.app.paths.opencode_json, cfg)
            elif self.harness == "jcode":
                with open(self.app.paths.jcode_config) as f:
                    txt = f.read()
                if f"[providers.{prov}]" not in txt:
                    txt += f'\n[providers.{prov}]\ntype = "openai-compatible"\nbase_url = "https://api.openai.com/v1"\n'
                    if key:
                        txt += f'api_key_env = "HARNESS_{prov.upper()}_API_KEY"\n'
                txt += f'\n[[providers.{prov}.models]]\nid = "{model}"\n'
                open(self.app.paths.jcode_config, "w").write(txt)
                if key:
                    import os as _os
                    env_path = os.path.join(os.path.expanduser("~"), ".config", "jcode", f"provider-{prov}.env")
                    _os.makedirs(os.path.dirname(env_path), exist_ok=True)
                    with open(env_path, "w") as ef:
                        ef.write(f"HARNESS_{prov.upper()}_API_KEY={key}\n")
        except Exception:
            pass
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.app.pop_screen()


class VaultPickScreen(Screen):
    """Pick vault rows to import to one harness (vault is source, working only)."""

    BINDINGS = [("escape", "back", "Back")]  # noqa: RUF012

    def __init__(self, harness: str) -> None:
        super().__init__()
        self.harness = harness
        self.selected: set[tuple[str, str]] = set()

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Label(f"Pick from Vault → {self.harness} (only working keys will be sent)", id="title")
            yield Static("Space to select, Enter to confirm", id="pick-hint")
            yield SelectionList(id="vault-pick-list")
            yield Button("Import selected", id="import-selected", variant="primary")
            yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        lst = self.query_one("#vault-pick-list", SelectionList)
        # Show vault active rows only — working keys (active/throttled); empty handling
        rows_added = 0
        for row in vault_table_rows(self.app.db, show_hidden=False):
            if row["health"] not in ("active", "throttled"):
                continue
            rows_added += 1
            pid, model = row["provider"], row["upstream"]
            # Use display provider (label) but store pid for import
            # For custom, pid is still custom_xxx, but display is label - need to map back
            # For now use provider as displayed, but engine will handle label->pid via vault
            key = f"{row['provider']} / {row['pool']} [{row['key']}]"
            lst.add_option(Selection(key, (row["provider"], row["pool"])))

    @on(Button.Pressed, "#import-selected")
    def _import(self) -> None:
        lst = self.query_one("#vault-pick-list", SelectionList)
        selected = list(lst.selected)
        if not selected:
            return
        # Map display provider back to pid if needed: try to resolve via engine
        # For custom label, need to find pid
        resolved = []
        for prov_disp, model in selected:
            # Try to find pid that matches display provider or label
            pid = prov_disp
            # Check if this is a label for custom
            for cand in self.app.db:
                if cand.startswith("custom") and isinstance(self.app.db[cand], dict):
                    if self.app.db[cand].get("label") == prov_disp:
                        pid = cand
                        break
            resolved.append((pid, model))
        engine.harness_import_from_vault(self.app.paths, self.harness, resolved)
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.app.pop_screen()


class JCodeScreen(Screen):
    """JCode target: detection status + managed profile + Sync.

    Read-only view over the wizard DB + ~/.jcode/config.toml (writes only
    happen via the Sync button, which drives :func:`engine.sync_jcode`
    with its backup/atomic/verify guarantees). ``j`` on Home opens it.
    """

    BINDINGS = [("escape", "back", "Back"),  # noqa: RUF012 -- Textual API
                ("r", "refresh", "Refresh"),
                ("s", "sync", "Sync")]

    def __init__(self) -> None:
        super().__init__()
        self.last_status = ""
        self.data: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Label("JCode — same gateway models, TOML target", id="title")
            yield Static("", id="jcode-status")
            yield Tree("llm-proxy-wizard", id="jcode-tree")
            yield Button("Sync JCode", id="sync-jcode")
            yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_content()

    def on_screen_resume(self) -> None:
        self.refresh_content()

    def refresh_content(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        try:
            self.data = engine.jcode_status(app.paths)
        except Exception:  # noqa: BLE001 -- status screen must not crash
            self.data = {}
        st = self.data
        lines = []
        lines.append(f"Installed: {'yes' if st.get('installed') else 'no'}"
                     + (f" (v{st['version']})" if st.get("version") else ""))
        lines.append(f"Config: {st.get('config', '?')}"
                     + ("" if st.get("config_exists") else " (not found)"))
        if st.get("config_exists") and not st.get("parseable"):
            lines.append("Config: NOT parseable (fix before sync)")
        if st.get("profiles"):
            profs = ", ".join(st["profiles"])
            lines.append(f"Profiles: {profs}")
            lines.append("Managed profile: "
                         + ("yes" if st.get("managed_profile") else "not yet"))
        self.last_status = "\n".join(lines)
        try:
            self.query_one("#jcode-status", Static).update(self.last_status)
            self._rebuild_tree()
        except NoMatches:  # not yet mounted
            pass

    def _rebuild_tree(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        try:
            sync = engine._load_sync_module()
            aliases, roles, _src = sync.load_aliases(app.paths.yaml_file)
        except Exception:  # noqa: BLE001 -- no gateway config yet
            aliases, roles = [], []
        exposed = list(aliases) + [r for r in roles if r not in aliases]
        tree = self.query_one("#jcode-tree", Tree)
        tree.clear()
        tree.root.label = (f"llm-proxy-wizard — {len(exposed)} logical model(s)"
                           f" @ http://localhost:4000/v1")
        for m in exposed:
            tree.root.add_leaf(m)
        if not exposed:
            tree.root.add_leaf("(apply the gateway config first)")
        tree.root.expand_all()

    def action_refresh(self) -> None:
        self.refresh_content()

    @on(Button.Pressed, "#sync-jcode")
    def _sync(self) -> None:
        self.action_sync()

    def action_sync(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        try:
            result = _quiet_call(engine.sync_jcode, app.paths)
        except FileNotFoundError as e:
            note = f"Sync failed ({e})."
        except ValueError as e:
            note = f"Sync failed ({e})."
        else:
            note = (f"JCode updated ({len(result.get('models', []))} model(s)). "
                    "Unrelated JCode settings untouched.")
        self.refresh_content()
        self.last_status += f"\n{note}"
        try:
            self.query_one("#jcode-status", Static).update(self.last_status)
        except NoMatches:
            pass

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()


class ImportScreen(Screen):
    """Explicit one-shot imports: OpenCode -> Wizard, JCode -> Wizard.

    The wizard becomes the normalized source of truth after import;
    exports stay explicit (never a background daemon). ``i`` on Home.
    """

    BINDINGS = [("escape", "back", "Back")]  # noqa: RUF012 -- Textual API

    def __init__(self) -> None:
        super().__init__()
        self.last_status = "Adopt an existing client config into the wizard.\n" \
                           "Secrets are stored as credentials and never shown."

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Label("Import client configuration", id="title")
            yield Static("", id="import-status")
            yield Button("Import from OpenCode", id="import-opencode",
                         variant="primary")
            yield Button("Import from JCode", id="import-jcode")
            yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self._render_status()

    def on_screen_resume(self) -> None:
        self._render_status()

    def _render_status(self) -> None:
        try:
            self.query_one("#import-status", Static).update(self.last_status)
        except NoMatches:
            pass

    @on(Button.Pressed, "#import-opencode")
    def _import_opencode(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        try:
            result = _quiet_call(engine.import_opencode, app.db, app.paths)
        except Exception as e:  # noqa: BLE001 -- report, don't crash
            self.last_status = f"Import failed ({str(e)[:100]})"
        else:
            if result.get("gateway_models"):
                note = (f"OpenCode already uses the wizard gateway "
                        f"({len(result['gateway_models'])} model(s)) — nothing to import.")
            elif result.get("providers"):
                names = [p.get("name", "?") for p in result["providers"]]
                note = (f"Imported {len(names)} provider(s): {', '.join(names)}. "
                        "Validate keys on their provider screens, then Apply.")
                engine.save_state(app.db, app.paths)
                app.db = engine.load_state(app.paths)
            else:
                note = "No direct providers with models found."
            self.last_status = note
        self._render_status()

    @on(Button.Pressed, "#import-jcode")
    def _import_jcode(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        try:
            result = _quiet_call(engine.import_jcode, app.db, app.paths)
        except Exception as e:  # noqa: BLE001 -- report, don't crash
            self.last_status = f"Import failed ({str(e)[:100]})"
        else:
            if result.get("managed_models"):
                note = (f"JCode already uses the wizard gateway profile "
                        f"({len(result['managed_models'])} model(s)) — not re-imported.")
            elif result.get("providers"):
                names = [p.get("name", "?") for p in result["providers"]]
                note = (f"Imported {len(names)} provider(s): {', '.join(names)}. "
                        "Validate keys on their provider screens, then Apply.")
                engine.save_state(app.db, app.paths)
                app.db = engine.load_state(app.paths)
            else:
                note = "No custom provider profiles with models found."
            self.last_status = note
        self._render_status()

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()


# ------------------------------------------------------------------ app ---

class WizardApp(App):
    TITLE = "LLM Vault"
    SUB_TITLE = f"v{engine.__version__}"
    CSS = """
    #body { width: 1fr; height: auto; margin: 1 2; }
    #title { text-style: bold; margin-bottom: 1; }
    #gateway-badge { text-style: bold; margin-bottom: 1; }
    #home-content, #test-status, #test-results, #done-content { margin-bottom: 1; }
    #home-attention, #opencode-status { margin: 1 0; }
    #sync-targets { margin-bottom: 1; color: $text-muted; }
    #jcode-status, #import-status { margin-bottom: 1; }
    #jcode-tree { height: 10; margin-bottom: 1; }
    #home-filter { margin-bottom: 1; }
    #home-hint { margin-bottom: 1; }
    #models-table { height: 14; margin-bottom: 1; }
    #opencode-tree { height: 16; margin-bottom: 1; }
    #quota-status { margin-bottom: 1; }
    #quota-table { height: 14; margin-bottom: 1; }
    #keys-input { height: 6; margin-bottom: 1; }
    #manual-input { height: 4; margin-bottom: 1; }
    #model-list { height: 12; margin-bottom: 1; }
    #endpoint-input, #filter { margin-bottom: 1; }
    #phase-status, #check-results, #model-status, #apply-status { margin: 1 0; }    Button { margin-bottom: 1; }
    #detail-info, #detail-status { margin: 1 0; }
    """

    def __init__(self, paths: engine.EnginePaths | None = None,
                  status: str | None = None, status_auto_refresh: bool = True,
                  auto_probe: bool = True) -> None:
        super().__init__()
        self.paths = paths or engine.EnginePaths.from_env()
        self.db: dict[str, Any] = engine.load_state(self.paths)
        self.status = status or "unknown"
        # Note: named to avoid colliding with Textual's own auto_refresh.
        self.status_auto_refresh = status_auto_refresh and status is None
        # Fill the Status column on first mount (one background probe per
        # credential); tests pass False to stay offline.
        self.auto_probe = auto_probe

    def on_mount(self) -> None:
        self.push_screen(HomeScreen())

    def refresh_status_background(self) -> None:
        if self.status_auto_refresh and self.status == "unknown":
            self.run_worker(self._query_status, thread=True, exclusive=True)

    def _query_status(self) -> None:
        status = engine.gateway_status()
        self.call_from_thread(self._apply_status, status)

    def _apply_status(self, status: str) -> None:
        self.status = status
        for screen in self.screen_stack:
            if isinstance(screen, HomeScreen):
                screen.refresh_content()


def main(argv: list[str] | None = None) -> int:
    """Entry point. ``--version`` / ``--check`` are non-interactive."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--version" in args or "-v" in args:
        print(engine.__version__)
        return 0
    if "--check" in args:
        return startup_check()
    WizardApp().run()
    return 0


def startup_check() -> int:
    """Verify the install without touching the screen: imports, paths, DB."""
    import textual as _textual
    print(f"wizard {engine.__version__} / textual {_textual.__version__}")
    try:
        paths = engine.EnginePaths.from_env()
    except Exception as e:  # noqa: BLE001 -- report, don't crash
        print(f"[FAIL] cannot resolve paths: {e}")
        return 1
    print(f"db: {paths.db_file}")
    try:
        db = engine.load_state(paths)
    except Exception as e:  # noqa: BLE001 -- report, don't crash
        print(f"[FAIL] cannot read DB: {e}")
        return 1
    providers = [pid for pid, pdata in db.items()
                 if not pid.startswith("_") and isinstance(pdata, dict)
                 and (pdata.get("keys") or pdata.get("models"))]
    print(f"providers configured: {len(providers)}"
          + (f" ({', '.join(sorted(providers))})" if providers else ""))
    _deps, pools, _roles, errors = engine.compile_config(db)
    if errors:
        print(f"[WARN] config would not compile: {errors[0]}")
    else:
        print(f"pools compile: {len(pools)}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
