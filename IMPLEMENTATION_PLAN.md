# IMPLEMENTATION_PLAN — llm-vault (hard fork)

Legacy v2.0→v2.6 delivered in llm-proxy-wizard. This fork pivots to vault-centric.

## Done (inherited)

T1–T17: provider/quota, compiler/YAML, restart/smoke, pools/roles, probe/latency, per-model quota, engine, TUI Home/OpenCode/JCode/Import, JCode TOML.

## Vault pivot (this fork)

- **V1**: hard fork, `engine.Proxy` enum, `harness.detect()`, vault-only main table, proxy/harness tabs, `sync-*.py` one per harness, hidden dead keys.
- **V2**: Biofrost proxy, cursor/cline harnesses.

## Backlog

| ID | Task | Status |
|---|---|---|
| B1 | Proxy-to-proxy import | DEFERRED |
| B2 | Biofrost adapter | PLANNED (V2) |

Same DoD: tests → verify → docs → sync installed copy → commit `Vault vX.Y: …` → push to `hardfork`.
