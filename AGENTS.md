# Agent Instructions for `llm-vault` (hard fork of llm-proxy-wizard)

Source: `mansourvery-hub/llm-proxy-wizard` → hard fork `mansourvery-hub/llm-vault`. Vault is long-lived, proxy/harness are pluggable.

## 1. Architecture — vault / proxy / harness separation

```
Vault (providers_db.json)  ──working keys only──>  Harness (opencode.json, ~/.jcode/config.toml, …)
      │                                              ↑
      └─all keys (active/hidden)                     │ detected dynamically
                                                     │
Proxy (config.yaml / biofrost.json)  ← routing only ─┘
```

- **Vault** (`providers_db.json`, schema v2, 0600): `provider → quota_domain → credential → model`. Accumulates hundreds of keys over years, tagged `active` / `throttled` (kept) / `invalid` / `expired` / `unknown`. By default TUI shows `active`+`throttled` only (`a` to show hidden). Dead keys are never auto-deleted. `key != quota`, `deployment != model`.
- **Proxy** (`config.yaml` for `litellm` on `localhost:4000` now; `biofrost` planned): compiled from vault + `proxy.type` + `routing` (`usage-based-routing-v2` …). `wizard.py` compiles, proxy routes. Hardcoding `litellm` as default is allowed, but `engine.Proxy` is an enum and `tui.py` must read `proxy.type` rather than assuming litellm.
- **Harness** (`sync-*.py` one file per harness, stdlib only): `opencode` (`~/.config/opencode/opencode.json`, detects `zen` free), `jcode` (`~/.jcode/config.toml`, no free). Detected via `detect_harnesses()` (`opencode.json` exists, `~/.jcode` exists). Each harness tab offers `Delete` (wipe file/block) and `Keep free` (wipe except `zen` → keep `zen` in opencode, wipe all in jcode). Import: `Import all` (vault active → harness, only `ok`/`throttled`) or `Pick` (select vault rows → harness, one harness at a time). No harness→harness code.

**Files**
- `wizard.py` (stdlib+PyYAML): vault add/probe/tag, proxy compile (`compile_config` → `generate_yaml`), harness-agnostic. No harness mutation.
- `engine.py` (facade): `vault.*`, `proxy.*`, `harness.detect()`, `harness.import_from_vault(harness, filter)`. No logic.
- `tui.py` (Textual, presentation only): `Vault` main table + `Proxy` tab + `Harness` tabs (dynamic). Never holds secrets, quota math, or harness mutation.
- `sync-opencode.py` / `sync-jcode.py` / future `sync-biofrost.py`: one harness, JSONC/TOML tolerant, backup+atomic, `--dry-run`, never secrets.

**DB invariants** (fail → leave `config.yaml` untouched): no dup deployment, no empty endpoint, no alias/role collision, vault dead keys hidden not deleted.

## 2. Environment & Checks

- Wizard/tests: `~/.config/litellm/venv/bin/python` (PyYAML+litellm). Syncs: system `python3`.
- Safe test: `LITELLM_DB_FILE=/tmp/db.json LITELLM_YAML_FILE=/tmp/cfg.yaml OPENCODE_JSON=/tmp/oc.json JCODE_CONFIG=/tmp/jc.toml ~/.config/litellm/venv/bin/python wizard.py`
- Before finish: `python3 -m py_compile wizard.py sync-*.py engine.py tui.py && ~/.config/litellm/venv/bin/python -m unittest discover -s tests && ruff check wizard.py engine.py tui.py`
- After `wizard.py`/`engine.py`/`sync-*.py`/`tui.py` changes: `cp wizard.py engine.py sync-*.py tui.py ~/.config/litellm/ && chmod +x ~/.config/litellm/*.py`

## 3. Conventions

- `__version__` in `wizard.py`. Never commit `providers_db.json`/`config.yaml`/`.master_key`/`*.bak*`.
- Secrets: `snippet()` suffix only, 0600, atomic, no header/body logs.
- UI: `Vault` is main screen; `Harness` tabs are dynamic; no extra non-functional features.
- Future: `biofrost` proxy, `cline`/`cursor` harnesses — same vault, `detect_harnesses()` adds a tab.

## 4. Git — hard fork

```bash
gh repo clone mansourvery-hub/llm-vault   # not llm-proxy-wizard
gh repo create mansourvery-hub/llm-vault --public --source=. --remote=hardfork --push
```
Commit: `Vault vX.Y: …` . Push to `hardfork` (llm-vault), not `origin`.
