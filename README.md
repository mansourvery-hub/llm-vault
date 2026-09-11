# LLM Vault — proxy-agnostic API vault + harness sync

> **Hard fork of [`llm-proxy-wizard`](https://github.com/mansourvery-hub/llm-proxy-wizard) → [`llm-vault`](https://github.com/mansourvery-hub/llm-vault).**
> Vault is source of truth. Proxies and harnesses import from vault, not from each other.

## Product direction

**Vault** — accumulates API keys + models over years. Hundreds of keys, tagged `active` / `rate-limited` (kept) / `expired` / `invalid` / `unknown`. Hidden by default: show only `active` (+ rate-limited). Old dead keys stay in vault, never deleted unless you purge. No complicated `opencode → jcode` import — both import from vault.

**Proxy** — routing config, separate from vault. `LiteLLM` is current default (`http://localhost:4000`), `Biofrost` and others are future backends. Proxies declare `type` and `routing` (strategy, retries, cooldowns); vault supplies credentials/models. One proxy → one `config.yaml` (LiteLLM) or equivalent. Proxies are pluggable, not hardcoded.

**Harnesses** — detected dynamically (`opencode` → `~/.config/opencode/opencode.json`, `jcode` → `~/.jcode/config.toml`, future `cline`/`cursor` etc.). Each harness tab shows its **installed config** on first open, with two deletes:
- `Full delete` — wipe harness config.
- `Keep free` — delete everything except free/bundled providers (ex: leaves `OpenCode Zen` in opencode; for jcode both deletes wipe everything — no free built-ins).

Import from vault (per harness):
- `Import all` — every **working** key (`active` + rate-limited) from vault.
- `Pick` — select vault entries to send to that harness (only working keys are sent; dead keys never leave vault).

## Quick start

```bash
gh repo clone mansourvery-hub/llm-vault && cd llm-vault
python3 -m venv ~/.config/litellm/venv && ~/.config/litellm/venv/bin/pip install -r requirements.txt
cp wizard.py engine.py tui.py sync-*.py ~/.config/litellm/ && chmod +x ~/.config/litellm/*.py
llm-vault  # TUI: Vault is main screen
```

**TUI (minimal):**
- `Vault` (main): table `provider / model / key… / status` — `active` only by default, `a` toggle hidden, `/` filter, `P` probe.
- `Harness` tabs: `o` Opencode, `j` JCode (auto-detected, grey if not installed). Inside: `View` (current harness models), `Delete` / `Keep free`, `Import all` / `Pick`.
- `Proxy` tab: choose `litellm` (default) / `biofrost` (planned) → `Apply` writes `config.yaml` and restarts only if changed.
- `q` quit, `Esc` back.

**CLI (same vault):**
```bash
llm-vault --check
wizard.py --import-opencode-to-jcode --dry-run  # now just vault → harness, no direct harness→harness
```

## Files

| File | Role |
|---|---|
| `wizard.py` | vault + proxy compiler (no harness logic) |
| `engine.py` | vault API (harness/proxy call it) |
| `tui.py` | Vault main + harness/proxy tabs only |
| `sync-*.py` | one harness, one job, stdlib only |
| `providers_db.json` | vault store (600, never committed) |

Proxy/harness separation is enforced: `engine` never writes harness config without explicit vault import; `wizard` never hardcodes `litellm` except as `default_proxy`.

## Why this fork

`llm-proxy-wizard` mixed vault + litellm routing + harness sync. Over time that hid dead keys, forced `opencode→jcode` workarounds, and blocked new proxies. `llm-vault` keeps vault long-lived, proxies replaceable, harnesses thin.

Future: `biofrost` proxy, `cursor`/`windsurf` harnesses — same vault, `detect_harnesses()` adds a tab.
