# ARCHITECTURE — llm-vault

Vault is source of truth. Proxy and harness import from vault, never from each other.

## Overview

```
                Vault  providers_db.json (0600)
                   │  provider → quota_domain → credential → model
                   │  hundreds of keys, tag: active/throttled/expired/invalid
          ┌────────┴─────────┐
          ↓                  ↓ (only active/throttled)
   Proxy config          Harness configs (detected)
   config.yaml           opencode.json  ~/.jcode/config.toml  …
   (litellm/biofrost)    sync-opencode  sync-jcode
```

`wizard = vault + proxy compiler` (no harness code); `proxy = routing only`; `harness = one file, stdlib`.

## Vault

- Schema v2, `migrate_db()` idempotent. `credentials[]` is truth, `keys[]` compat. `cred-<sha256[:12]>` stable.
- Health per credential (`_health`), performance per `(provider,model)` (`_performance` 5-sample mean).
- Hidden by default: `active`+`throttled` shown, `invalid/expired/unknown` hidden (`a` toggle).

## Proxy (pluggable)

- `litellm` is default (`localhost:4000`), `biofrost` planned.
- `wizard.py: compile_config → generate_yaml` produces proxy file; `proxy.type` enum, `tui` reads it.
- `effective_base_url(db,pid)` single-source; custom_api uses it. Invariants: no dup deployment, no empty endpoint, no alias/role collision — fail leaves proxy file untouched.

## Harness (pluggable, stdlib only)

- `detect_harnesses()` → tab per harness present (`opencode.json` exists, `~/.jcode` exists). No harness→harness import.
- Each `sync-*.py`: JSONC/TOML tolerant, backup+atomic, `--dry-run`, never secrets, one `--help` screen.
- `opencode` keeps `zen` (free) on `Keep free`; `jcode` has no free → both deletes wipe all.
- Import: `Import all` (vault active → harness) or `Pick` (select vault rows → harness, one at a time).

## Layers

`wizard.py` (vault/proxy) → `engine.py` (facade: `vault.*`, `proxy.*`, `harness.*`) → `tui.py` (Vault table + Proxy + Harness tabs) → `sync-*.py`.

No layer holds another’s secrets or quota math.
