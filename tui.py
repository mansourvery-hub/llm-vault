#!/usr/bin/env python3
"""Textual TUI for llm-proxy-wizard.

Thin presentation layer only. All product logic (provider validation,
quota math, config compilation, secret handling, OpenCode sync) lives in
``engine.py`` / ``wizard.py`` — this file must never implement any of it.

Views: Home (deployment table) -> Configure -> Provider (-> ModelScreen)
-> Review(Done), plus Home -> Test and Home -> OpenCode view. Quota/speed/
routing/alias/service internals are NOT standalone screens; the grouping
question and retired models surface inside the configure flow only.

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
    TextArea,
    Tree,
)
from textual.widgets._selection_list import Selection

import engine

STATUS_MARK = {"running": "[green]●[/]", "stopped": "[red]●[/]",
               "unknown": "[yellow]●[/]"}

# Table columns for the Home dashboard (FCM-style dense table, one row
# per deployment). Keys are sort/filter ids; labels get a ▲/▼ marker
# for the active sort column at render time.
TABLE_COLUMNS = (
    ("pool", "Pool"),
    ("provider", "Provider"),
    ("upstream", "Model"),
    ("tier", "Tier"),
    ("rpm", "RPM"),
    ("tpm", "TPM"),
    ("quota", "Quota"),
    ("ctx", "Ctx"),
    ("health", "Health"),
    ("key", "Key"),
)
SORT_KEYS = ("pool", "provider", "tier", "rpm", "health")

HEALTH_DISPLAY = {
    "healthy": ("✓", "healthy"),
    "partially-throttled": ("~", "part-throttled"),
    "throttled": ("~", "throttled"),
    "invalid": ("✗", "invalid"),
    "unknown": ("?", "unknown"),
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
            provider = str(d.get("provider") or "")
            upstream = str(d.get("upstream_model") or "")
            caps = d.get("capabilities") or {}
            tier = str(caps.get("tier") or "unknown")
            ctx = str(caps.get("context_window") or "unknown")
            rpm, tpm = d.get("rpm"), d.get("tpm")
            qd = str(d.get("quota_domain") or "")
            n = counts.get((qd, upstream), 1)
            if isinstance(rpm, (int, float)):
                rpm_txt = f"{int(rpm)}÷{n}" if n > 1 else str(int(rpm))
            else:
                rpm_txt = "—"
            tpm_txt = str(int(tpm)) if isinstance(tpm, (int, float)) else "—"
            health = str(d.get("health") or "unknown")
            mark, word = HEALTH_DISPLAY.get(health, ("?", health or "unknown"))
            secret = d.get("secret") or ""
            suffix = engine.mask_secret(secret) if secret else (
                "local" if not d.get("credential_id") else "…" + str(d.get("credential_id"))[-4:])
            try:
                confidence = str((quota_conf.get(qd) or {}).get("confidence") or "")
            except AttributeError:
                confidence = ""
            rows.append({
                "pool": pool, "provider": provider, "upstream": upstream,
                "tier": tier, "rpm": rpm, "tpm": tpm,
                "rpm_txt": rpm_txt, "tpm_txt": tpm_txt,
                "quota_domain": qd, "quota": _short_quota(qd),
                "confidence": confidence, "shared": n,
                "ctx": "—" if ctx in ("unknown", "None", "") else ctx,
                "health": health, "health_txt": f"{mark} {word}",
                "health_rank": HEALTH_RANK.get(health, 9),
                "key": suffix,
                "endpoint": str(d.get("endpoint") or "—"),
            })
    return rows


def filter_table_rows(rows: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Case-insensitive substring filter over pool/provider/model/quota."""
    q = (query or "").strip().lower()
    if not q:
        return list(rows)
    return [r for r in rows
            if q in r["pool"].lower() or q in r["provider"].lower()
            or q in r["upstream"].lower() or q in r["quota_domain"].lower()]


def sort_table_rows(rows: list[dict[str, Any]], sort_key: str,
                    reverse: bool = False) -> list[dict[str, Any]]:
    """Sort display rows (pure; never touches the DB)."""
    key = sort_key if sort_key in SORT_KEYS else "pool"
    def _k(r: dict[str, Any]):
        if key == "rpm":
            v = r.get("rpm")
            return (-1 if v is None else int(v), r["pool"], r["provider"])
        if key == "health":
            return (r.get("health_rank", 9), r["pool"], r["provider"])
        return (str(r.get(key) or "").lower(), r["pool"], r["provider"])
    return sorted(rows, key=_k, reverse=reverse)


def gateway_badge(overview: dict[str, Any], n_deps: int) -> str:
    """One-line header badge: gateway state + pool/deployment counts."""
    mark = STATUS_MARK.get(overview.get("gateway", "unknown"),
                           STATUS_MARK["unknown"])
    n_pools = len(overview.get("pools", []))
    return (f"Gateway {mark} {status_word(str(overview.get('gateway', 'unknown')))}"
            f"  •  {n_pools} pool(s)  •  {n_deps} deployment(s)")


def row_detail_text(row: dict[str, Any] | None) -> str:
    """FCM-style detail card for the highlighted row (full, untruncated)."""
    if row is None:
        return "↑↓ move • Enter opens details • / filter • s sort • o OpenCode view"
    shared = ""
    if row.get("shared", 1) > 1:
        shared = (f"  (shared domain: {row['shared']} deployments "
                  f"split this quota — never {row['shared']}×)")
    conf = f" [{row['confidence']}]" if row.get("confidence") else ""
    return (f"{row['pool']}  via {row['provider']} / {row['upstream']}\n"
            f"tier {row['tier']} • ctx {row['ctx']} • "
            f"rpm {row['rpm_txt']} • tpm {row['tpm_txt']} • "
            f"quota {row['quota_domain'] or '—'}{conf}{shared}\n"
            f"health {row['health_txt']} • key {row['key']} • "
            f"endpoint {row['endpoint']}")


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
    """Dashboard table: one row per deployment (FCM-style, keyboard-first).

    Keys: ``c`` configure, ``t`` test, ``v`` review, ``o`` OpenCode view,
    ``/`` filter, ``s`` cycle sort, ``S`` reverse direction, ``x`` clear
    filter, ``h`` hide invalid, ``q`` quit. Arrows/Enter navigate; the
    detail card below always describes the highlighted row.
    """

    BINDINGS = [  # noqa: RUF012 -- Textual API
        ("c", "configure", "Configure"), ("t", "test", "Test"),
        ("v", "review", "Review"), ("o", "opencode", "OpenCode"),
        ("slash", "focus_filter", "Filter"),
        ("s", "cycle_sort", "Sort"), ("S", "reverse_sort", "Reverse"),
        ("x", "clear_filter", "Clear"), ("h", "toggle_hide", "Hide bad"),
        ("q", "quit_app", "Quit")]

    SORT_CYCLE = ("pool", "provider", "tier", "rpm", "health")

    def __init__(self) -> None:
        super().__init__()
        self.last_content = ""
        self.rows: list[dict[str, Any]] = []
        self.view_rows: list[dict[str, Any]] = []
        self.sort_key = "pool"
        self.sort_reverse = False
        self.hide_invalid = False
        self._built_columns = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Label("LLM Proxy Wizard", id="title")
            yield Static("", id="gateway-badge")
            yield Input(placeholder="Filter pools/providers/models ( / to focus, x to clear )",
                        id="home-filter")
            yield DataTable(id="models-table", cursor_type="row")
            yield Static("", id="row-detail")
            yield Static("", id="home-attention")
            yield Button("Configure", id="go-configure", variant="primary")
            yield Button("OpenCode view", id="go-opencode")
            yield Button("Test", id="go-test")
            yield Button("Review", id="go-review")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_content()
        app = self.app
        assert isinstance(app, WizardApp)
        app.refresh_status_background()
        # Keyboard-first like FCM: the table owns focus so single-key
        # actions (c/t/v/o/s/...) fire immediately; / moves to the filter.
        try:
            self.query_one("#models-table", DataTable).focus()
        except NoMatches:
            pass

    def on_screen_resume(self) -> None:
        self.refresh_content()
        try:
            self.query_one("#models-table", DataTable).focus()
        except NoMatches:
            pass

    # -- data --

    def refresh_content(self) -> None:
        app = self.app
        assert isinstance(app, WizardApp)
        overview = engine.gateway_overview(app.db, app.paths, status=app.status)
        self.rows = deployment_table_rows(app.db)
        badge = gateway_badge(overview, len(self.rows))
        self._apply_view()
        detail_row = self._highlighted_row() or (self.view_rows[0] if self.view_rows else None)
        detail = row_detail_text(detail_row)
        attention = overview.get("attention") or []
        if not self.rows and not attention:
            attention = ["No models configured yet — choose Configure to add keys."]
        attn_txt = ("Needs attention\n" + "\n".join(f"  ! {a}" for a in attention)) if attention else ""
        # Plain-text summary kept for tests / narrow terminals.
        self.last_content = "\n".join(
            [badge, "", *[r["pool"] for r in self.rows][:20],
             *([f"! {a}" for a in attention] if attention else [])])
        try:
            self.query_one("#gateway-badge", Static).update(badge)
            self.query_one("#row-detail", Static).update(detail)
            self.query_one("#home-attention", Static).update(attn_txt)
        except NoMatches:  # not yet mounted
            pass

    def _apply_view(self) -> None:
        try:
            filt = self.query_one("#home-filter", Input).value
        except NoMatches:
            filt = ""
        rows = filter_table_rows(self.rows, filt)
        if self.hide_invalid:
            rows = [r for r in rows if r["health"] != "invalid"]
        self.view_rows = sort_table_rows(rows, self.sort_key, self.sort_reverse)
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        try:
            table = self.query_one("#models-table", DataTable)
        except NoMatches:
            return
        table.clear(columns=True)
        for key, label in TABLE_COLUMNS:
            if key == self.sort_key:
                label = f"{label} {'▲' if not self.sort_reverse else '▼'}"
            table.add_column(label, key=key)
        self._built_columns = True
        if not self.view_rows:
            return
        for i, r in enumerate(self.view_rows):
            table.add_row(r["pool"], r["provider"], r["upstream"], r["tier"],
                          r["rpm_txt"], r["tpm_txt"], r["quota"], r["ctx"],
                          r["health_txt"], r["key"], key=f"row-{i}")
        try:
            filt = self.query_one("#home-filter", Input).value.strip()
        except NoMatches:
            filt = ""
        extra = [f"filter '{filt}'"] if filt else []
        if self.hide_invalid:
            extra.append("hiding invalid")
        table.border_title = (f"{len(self.view_rows)}/{len(self.rows)} "
                              f"sorted by {self.sort_key}"
                              + (f" ({', '.join(extra)})" if extra else ""))

    def _highlighted_row(self) -> dict[str, Any] | None:
        try:
            table = self.query_one("#models-table", DataTable)
        except NoMatches:
            return None
        try:
            idx = table.cursor_row
        except Exception:  # noqa: BLE001 -- no cursor yet
            return None
        if idx is None or not (0 <= idx < len(self.view_rows)):
            return None
        return self.view_rows[idx]

    def _update_detail(self) -> None:
        try:
            self.query_one("#row-detail", Static).update(
                row_detail_text(self._highlighted_row()))
        except NoMatches:
            pass

    # -- events --

    @on(Input.Changed, "#home-filter")
    def _filter_changed(self, _event: Input.Changed) -> None:
        self._apply_view()
        self._update_detail()

    @on(Input.Submitted, "#home-filter")
    def _filter_submitted(self, _event: Input.Submitted) -> None:
        try:
            self.query_one("#models-table", DataTable).focus()
        except NoMatches:
            pass

    @on(DataTable.RowHighlighted)
    def _row_highlighted(self, _event: DataTable.RowHighlighted) -> None:
        self._update_detail()

    # -- actions --

    def action_configure(self) -> None:
        self.app.push_screen(ConfigureScreen())

    def action_test(self) -> None:
        self.app.push_screen(TestScreen())

    def action_review(self) -> None:
        self.app.push_screen(DoneScreen())

    def action_opencode(self) -> None:
        self.app.push_screen(OpenCodeScreen())

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

    def action_clear_filter(self) -> None:
        try:
            inp = self.query_one("#home-filter", Input)
            inp.value = ""
            self.query_one("#models-table", DataTable).focus()
        except NoMatches:
            pass
        self._apply_view()
        self._update_detail()

    def action_toggle_hide(self) -> None:
        self.hide_invalid = not self.hide_invalid
        self._apply_view()

    def action_quit_app(self) -> None:
        self.app.exit()

    @on(Button.Pressed, "#go-configure")
    def _go_configure(self) -> None:
        self.action_configure()

    @on(Button.Pressed, "#go-opencode")
    def _go_opencode(self) -> None:
        self.action_opencode()

    @on(Button.Pressed, "#go-test")
    def _go_test(self) -> None:
        self.action_test()

    @on(Button.Pressed, "#go-review")
    def _go_review(self) -> None:
        self.action_review()


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
        self._show_state()

    # -- state --

    def _show_only(self, *ids: str) -> None:
        wanted = set(ids)
        for bid in self.BUTTONS:
            self.query_one(f"#{bid}", Button).display = bid in wanted

    def _needs_endpoint(self) -> bool:
        app = self.app
        assert isinstance(app, WizardApp)
        prov = engine.get_provider(self.pid, app.db) or {}
        if prov.get("type") == "custom_api" and not prov.get("base_url"):
            return True
        entry = app.db.get(self.pid)
        return bool(isinstance(entry, dict) and prov.get("type") == "custom_api"
                    and not entry.get("base_url") and not entry.get("endpoints"))

    def _show_state(self) -> None:
        try:
            keys_box = self.query_one("#keys-input", TextArea)
            ep_box = self.query_one("#endpoint-input", Input)
            status = self.query_one("#phase-status", Static)
            results = self.query_one("#check-results", Static)
        except NoMatches:  # not yet mounted
            return
        keys_box.display = self.phase in ("keys", "checking")
        ep_box.display = self.phase == "keys" and self._needs_endpoint()
        if self.phase == "keys":
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
        if self._needs_endpoint():
            self.endpoint = self.query_one("#endpoint-input", Input).value.strip() or None
            if not self.endpoint:
                self.last_result = "Enter the base URL first."
                self.phase = "keys"
                self._show_state()
                return
        else:
            self.endpoint = None
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
                                   endpoint=self.endpoint)
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
        self.app.push_screen(DoneScreen())

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
        self._worker = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="body"):
            yield Label("Review", id="title")
            yield Static("", id="done-content")
            yield Static("", id="apply-status")
            yield Button("Apply changes", id="apply", variant="primary")
            yield Button("Group suggested models", id="group-suggested")
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
        self.last_content = done_lines(app.db)
        if self.suggestions:
            lines = ["", "Same model on several providers? Group only if",
                     "they are truly interchangeable:"]
            for stem, members in sorted(self.suggestions.items()):
                provs = ", ".join(sorted({m["provider"] for m in members}))
                lines.append(f"  ? {stem}  ({provs})")
            self.last_content += "\n".join(lines)
        self.query_one("#done-content", Static).update(self.last_content)
        self.query_one("#apply-status", Static).update(self.last_status)
        for bid in ("apply", "group-suggested", "sync", "cancel", "back"):
            self.query_one(f"#{bid}", Button).display = bid in self._visible_ids()

    def _visible_ids(self) -> list[str]:
        ids = ["group-suggested"] if self.suggestions else []
        if self.applied_ok:
            return [*ids, "sync", "back"]
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
        app = self.app
        assert isinstance(app, WizardApp)
        try:
            result = _quiet_call(engine.sync_opencode, app.paths)
        except FileNotFoundError:
            self._set_status(self.last_status + "\nOpenCode config not found — "
                             "skipped. The gateway itself is working.")
            return
        except ValueError as e:
            # sync failed AFTER a working gateway: report separately
            self._set_status(self.last_status + f"\nOpenCode sync failed "
                             f"separately ({e}) — the gateway itself is working.")
            return
        exposed = result.get("exposed", [])
        self._set_status(self.last_status + f"\nOpenCode updated "
                         f"({len(exposed)} model(s)). Restart the OpenCode "
                         "TUI, then /models -> litellm/<name>.")

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#back")
    def _back(self) -> None:
        self.action_back()


# ------------------------------------------------------------------ app ---

class WizardApp(App):
    TITLE = "LLM Proxy Wizard"
    SUB_TITLE = f"v{engine.__version__}"
    CSS = """
    #body { width: 1fr; height: auto; margin: 1 2; }
    #title { text-style: bold; margin-bottom: 1; }
    #gateway-badge { text-style: bold; margin-bottom: 1; }
    #home-content, #test-status, #test-results, #done-content { margin-bottom: 1; }
    #home-attention, #opencode-status { margin: 1 0; }
    #home-filter { margin-bottom: 1; }
    #models-table { height: 14; margin-bottom: 1; }
    #opencode-tree { height: 16; margin-bottom: 1; }
    #row-detail { margin: 1 0; border: solid #444444; padding: 0 1; }
    #keys-input { height: 6; margin-bottom: 1; }
    #manual-input { height: 4; margin-bottom: 1; }
    #model-list { height: 12; margin-bottom: 1; }
    #endpoint-input, #filter { margin-bottom: 1; }
    #phase-status, #check-results, #model-status, #apply-status { margin: 1 0; }    Button { margin-bottom: 1; }
    """

    def __init__(self, paths: engine.EnginePaths | None = None,
                 status: str | None = None, status_auto_refresh: bool = True) -> None:
        super().__init__()
        self.paths = paths or engine.EnginePaths.from_env()
        self.db: dict[str, Any] = engine.load_state(self.paths)
        self.status = status or "unknown"
        # Note: named to avoid colliding with Textual's own auto_refresh.
        self.status_auto_refresh = status_auto_refresh and status is None

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
