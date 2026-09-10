# Agent Instructions for `llm-proxy-wizard`

This document provides context, conventions, and operational workflows for AI coding agents working in this repository.
Hard fork of `litellm-wizard`, renamed because other proxy backends besides LiteLLM are planned. LiteLLM on `localhost:4000` is the current (and only) backend.

---

## 1. Project Architecture & Structure

`llm-proxy-wizard` is a quota-aware, health-aware deployment manager for a local LLM gateway (LiteLLM on `localhost:4000` today; other proxy backends planned).
Mental model: **wizard = control plane / config compiler, LiteLLM = runtime router, OpenCode + JCode = output targets (never the source of truth).**

```
                ┌──→ LiteLLM config (config.yaml)
                │
Wizard DB/Model ├──→ OpenCode config (opencode.json, managed "litellm" block)
                │
                └──→ JCode config (~/.jcode/config.toml, managed provider profile)
```

- **`wizard.py`** (v2.x, Python 3.10+, PyYAML): interactive CLI. Direct key validation, live catalogs, minimal model probes, quota domains, capability pools, role aliases, `config.yaml` compilation, gateway/pool tests, readiness-aware restart, OpenCode/JCode sync offers, explicit OpenCode/JCode imports (`import`, `jcode import`, `--import-opencode`, `--import-jcode`, `--sync-jcode`, `--sync-opencode`).
- **`engine.py`** (stdlib + PyYAML via `wizard`): clean callable facade over `wizard.py`/`sync-opencode.py`/`sync-jcode.py`. No reimplemented logic. The TUI (and future callers) drive the product through it; network/systemd adapters are mockable, paths overridable (`EnginePaths.jcode_config` honours `JCODE_CONFIG`).
- **`tui.py`** (Textual): thin presentation layer — table-first Home dashboard (one row per deployment: pool/provider/model/tier/quota/status, click-header sort, tier filter, auto-probe on mount + `P` re-probe, Enter/click row-detail modal with per-credential probe + gateway test; ctx shown in modal only when the litellm map knows it; one-line sync-targets summary) + OpenCode view (`o`) + JCode view (`j`: detection + managed profile + Sync via `engine.sync_jcode`) + Import screen (`i`: OpenCode/JCode → Wizard via `engine.import_*`) + Quota dashboard (`u`) + Configure/Provider/Test/Review screens only. Must never contain provider logic, quota math, compilation, secret handling, or OpenCode/JCode mutation.
- **`sync-opencode.py`** (stdlib-only): syncs user-facing pools (+roles) into `opencode.json` as a `litellm` block. JSONC-tolerant, backup + atomic write, idempotent, `--dry-run`, never carries secrets.
- **`sync-jcode.py`** (stdlib-only, incl. a minimal TOML-subset parser/writer): syncs the same pools (+roles) into JCode v0.84+'s `~/.jcode/config.toml` as a managed `[providers.llm-proxy-wizard]` profile (`type="openai-compatible"`, `base_url=http://localhost:4000/v1`, `api_key_env=LITELLM_MASTER_KEY`, `[[...models]]` arrays, `[provider]` defaults). Only the managed span + the two default fields are touched — all other JCode sections survive byte-for-byte. Backup + atomic write + re-parse verification with restore; `managed_hash` stamp detects external edits (blocked by default, `--overwrite-external` forces). Idempotent, `--dry-run`, `--print`, `--jcode <path>`, `--no-roles`, `--keep-default`. Never carries secrets.
- **`tests/`**: stdlib `unittest` suite (fake keys only, mocked HTTP; JCode/OpenCode paths always temp-dir overrides). Run with the venv python (system python lacks PyYAML).
- **`litellm.service`**: systemd user service on port 4000.
- **`README.md`**: user-facing guide + jargon buster.

### Imports (explicit one-shots, never a daemon)

`engine.import_opencode` (Case A: an existing `provider.litellm` block at localhost:4000 is detected and NOT re-imported, only reported; Case B: direct providers imported as `custom_<slug>` slots) and `engine.import_jcode` (skips the managed profile; imports other named profiles). Both are idempotent — stable identity by base_url first, then normalized name; models union, never replace; secrets go through the credential mechanism and are never printed.

### Internal hierarchy

```text
Provider -> Quota domain -> Credential -> Deployment -> Logical pool -> Role -> LiteLLM YAML
```

Key distinctions: `key != quota`, `deployment != model`, `identity != capability`,
`validation != health != quota`, `alias(pool) != role`.

### Database (`providers_db.json`, schema v2, mode 0600)

```json
{
  "_schema_version": 2,
  "_settings": {"routing_preference": "balanced", "quota_split_mode": "per-deployment-split",
                "validation_mode": "FAST", "sample_size": 2},
  "_aliases": {"pool": [{"provider": "pid", "model": "mid"}]},
  "_roles": {"fast": {"pools": [...], "fallback": [...], "requires": {}}},
  "_quota_domains": {"google-project-a": {"rpm": 10, "tpm": null, "rpd": null,
                      "confidence": "manual|provider_default|conservative|unknown",
                      "per_model": {"gemini-3.7-flash": {"rpm": 5, "tpm": null}}}},
  "_health": {"pid:cred-id": {"status": "ok|throttled|invalid|unknown", ...}},
  "_performance": {"pid:model": {"samples": [1.2, ...], "updated_at": "..."}},
  "<pid>": {"keys": [...], "credentials": [{"id": "cred-<hash>", "secret": "...",
             "label": "", "quota_domain": "...", "project_id": "", "enabled": true,
             "validation": {"status": "...", ...}}],
             "models": [...], "endpoints": [], "base_url": "...", "disabled": false}
}
```

- `keys[]` is kept in sync for backward compat; `credentials[]` is the source of truth. Credential IDs are `cred-<sha256(secret)[:12]>` — stable across reorders, never the raw key.
- `project_id` (additive, Google projects): keys of one project share one speed limit. `effective_quota_domain()` groups by it (`project:<pid>:<id>`) even over a stale per-credential default; `quota_domains_list()` exposes domains for the TUI dashboard.
- `_quota_domains[qd].per_model` (Google free tiers are per project+model): limit precedence is deployment override > per-model domain > domain-wide > provider default. Shared-domain RPM splits per (domain, model).
- `_performance` (probe latency memory): `record_probe_latency()` keeps a rolling window (last 5) per (provider, model); `probe_latency()` returns the mean. Recorded from `test_models` probes, shown in the TUI row detail; ids only, never secrets.
- Free-first roles (`google-free-fast` = flash/lite tier, `google-free-smart` = pro/reasoning): `suggest_free_first_roles()` proposes when gemini pools exist and the role is missing; `apply_free_first_roles()` writes only `google-free-*` names (user roles untouched). Surfaces as Review button + role manager `[F]`.
- `migrate_db()` is automatic + idempotent: legacy `_unified` -> `_aliases` (deduped), missing sections defaulted, one-credential-per-domain defaults. Never silently deletes valid config; `normalize_aliases()` cleans stale members visibly.
- `context_window` comes from the installed litellm model map via `lookup_context_window()` (exact id strings only, lazy import) — else `unknown`, never guessed.
- Quota semantics: Google = project-scoped (ask bulk grouping); others default to credential/account/unknown per `PROVIDER_META` (only verified facts; else `unknown`).

### Compiler (`compile_config` -> `generate_yaml`)

Stages: migrate/normalize -> credentials -> quota -> model metadata -> deployments
-> pools -> roles -> routing/fallbacks -> YAML. Deterministic order
(pool -> trust tier -> provider -> domain -> credential).

- One deployment = one provider + credential + endpoint + model. Shared-domain RPM is split per (domain, model) — never N x quota.
- custom_api base URL is single-sourced (`effective_base_url`: explicit arg > stored `endpoints[0]` > stored `base_url` > builtin) and honored identically by validate/catalog/probe/yaml — never hardcode a provider base at a call site.
- Roles compile to `model_group_alias` (role -> first pool) + `fallbacks` (verified shapes for installed LiteLLM 1.100.0).
- `router_settings` (verified vs installed LiteLLM): `usage-based-routing-v2`, `num_retries: 1`, `cooldown_time: 60`, `allowed_fails: 1`, `allowed_fails_policy` (`RateLimit: 0`, `Timeout/InternalServer/ServiceUnavailable/BadGateway: 1`), `enable_pre_call_checks: true`, retry policy with ONLY supported keys (`Authentication/BadRequest/ContentPolicyViolation/RateLimit: 0`, `Timeout/InternalServer: 1`). Per-deployment `litellm_params.cooldown_time`: 60s for 429-prone providers (gemini), 30s otherwise — a 429 cools immediately and the next domain's deployment takes over.
- `general_settings.master_key` = `os.environ/LITELLM_MASTER_KEY` (env-backed; resolved via `get_master_key()`: env -> `~/.config/litellm/.master_key` (0600) -> generated). No hard-coded secrets.
- Invariants enforced, failure leaves `config.yaml` untouched: no dup deployments, no stale members, no empty aliases/roles, no missing credentials, no empty endpoints, no pool/role name collisions.

---

## 2. Environment & Testing Guidelines

### Python Environment
- Wizard/tests: `~/.config/litellm/venv/bin/python` (has PyYAML + litellm).
- `sync-opencode.py` / `sync-jcode.py`: system `python3` is fine (stdlib only).

### Testing Code Changes Safely
Never modify real user configs. Always use env overrides:

```bash
LITELLM_DB_FILE=/tmp/test_db.json \
LITELLM_YAML_FILE=/tmp/test_config.yaml \
OPENCODE_JSON=/tmp/test_opencode.json \
JCODE_CONFIG=/tmp/test_jcode.toml \
LITELLM_SECRET_FILE=/tmp/test_master.key \
~/.config/litellm/venv/bin/python wizard.py
```

### Checks (run all before finishing)
```bash
python3 -m py_compile wizard.py sync-opencode.py sync-jcode.py engine.py tui.py
~/.config/litellm/venv/bin/python -m unittest discover -s tests
~/.config/litellm/venv/bin/python -m pytest tests/   # same suite, pytest runner
ruff check wizard.py sync-opencode.py engine.py tui.py tests/
```

Only fake keys in tests; mock HTTP (`_get`/`_post`/`test_single_model`); never hit real providers from automated tests.

### Syncing the Installed Copy
Live `litellm-add` runs from `~/.config/litellm/wizard.py`. Since v2.6.0, wizard.py's `import`/`jcode`/`opencode` commands import `engine.py`, `sync-opencode.py`, and `sync-jcode.py` from the same directory — and the `llm-proxy-wizard` shell alias runs the repo's `tui.py`, which loads the same siblings from the repo. **After any of `wizard.py`, `engine.py`, `sync-opencode.py`, `sync-jcode.py`, `tui.py` changes:**

```bash
cp wizard.py engine.py sync-opencode.py sync-jcode.py tui.py ~/.config/litellm/
chmod +x ~/.config/litellm/*.py
```

(If the installed copy ever falls behind, its `import`/`jcode` commands fail with FileNotFoundError — run the above to fix.)

### Future work (explicitly deferred, do NOT implement unprompted)
- Proxy-to-proxy import: transferring keys/models directly from another running proxy's config (e.g. a foreign LiteLLM/New-API instance) into the wizard DB, without going through OpenCode/JCode config files.

---

## 3. Versioning & Conventions

- `__version__` in `wizard.py` (currently v2.x). Bump on user-facing features/major fixes.
- Never commit `providers_db.json`, `config.yaml`, `.master_key`, `*.bak*` (see `.gitignore`).
- `sync-opencode.py` stays stdlib-only. `wizard.py` stays dependency-light (stdlib + PyYAML).
- Secrets: mask with `snippet()`/`_mask_secret()` (suffix only), never log headers/bodies wholesale, atomic writes (`_atomic_write_json/_text` + 0600) for DB/YAML/opencode.json, backup before replacing user-owned files.
- Preserve fast UX: `add google` -> paste -> DONE -> pick -> DONE -> Q. Single-key flows ask no quota questions; bulk flows ask once.
- Preserve: fuzzy `add <name>`, custom endpoints, free-first picker + `MORE`, keep-valid-keys flow, alias manager, dry-run sync, JSONC parsing, Zen-stays-native rule (never route OpenCode Zen session models via LiteLLM).

---

## 4. Git & GitHub (`gh`) Usage Workflows

### Authentication & Cloning
```bash
gh auth login
gh repo clone mansourvery-hub/llm-proxy-wizard
```

### Commit Conventions
- Format: `Wizard vX.Y.Z: short description` or `sync-opencode: short description`.

### Inspection & Shipping Checklist
1. `git status` (only intended files; tests/ allowed; no secrets/backups).
2. `git diff` (exact modifications).
3. `git log -n 5 --oneline` (message style).
4. Stage + commit + push (`git add wizard.py sync-opencode.py README.md AGENTS.md tests` as appropriate).
