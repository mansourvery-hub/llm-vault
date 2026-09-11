# PRODUCT — llm-vault

Hard fork of llm-proxy-wizard. Vault is long-lived, proxy/harness are pluggable.

## Problem

Keys accumulate for years (hundreds). Tools multiply (opencode, jcode, …). Routing is a different job than storage. Mixing them hides dead keys, forces `opencode→jcode` hacks, and locks you to one proxy.

## Solution

**Vault** is source of truth (active/throttled vs hidden dead). **Proxy** compiles vault → `config.yaml` (litellm) or future `biofrost.json`. **Harness** imports *only working* keys from vault — pick or bulk.

## Users

One developer, local machine, free-tier keys, many harnesses over time.

## Journeys

1. **Vault**: add keys → vault probes → tag `ok`/`throttled` (kept) vs `invalid`/`expired`; pick models; hidden by default.
2. **Proxy**: choose `litellm` (default) → `Apply` writes config, restarts if changed.
3. **Harness**: open tab → see harness’s current file (install state) → `Delete` / `Keep free` → `Import all` (vault active) or `Pick`.

No harness↔harness import; vault never auto-deletes dead keys.

## Requirements

- Vault CRUD + probe + tag + filter.
- Proxy compiler per `proxy.type` (litellm now, biofrost next).
- Harness detection + isolated `sync-*.py`.
- Import only `ok`/`throttled` to harness.
- Minimal UI: Vault + Proxy + Harness tabs.

## Non-goals

Extra dashboards that don’t serve vault/proxy/harness. Keep it clean.
