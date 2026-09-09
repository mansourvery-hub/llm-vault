# LLM Proxy Wizard (local LLM gateway + setup wizard)

Run many AI providers (Google Gemini, OpenRouter, Z.AI, TokenRouter, Ollama…)
through **one address on your own machine**: `http://localhost:4000`.
Any tool that speaks the OpenAI format (OpenCode, Cline, plain scripts) can use it —
you configure keys **once**, and every tool shares them.

> Hard fork of [`litellm-wizard`](https://github.com/mansourvery-hub/litellm-wizard),
> renamed because other proxy backends besides LiteLLM are planned. LiteLLM is the
> current (and only) backend — everything below still runs against it.

> **Jargon buster (read this first, it makes everything below click)**
> - **Terminal / shell** — the black window where you type commands. On KDE, open it with `Ctrl+Alt+T`. Your shell is called `zsh`.
> - **LiteLLM** — a free program that pretends to be OpenAI, but secretly forwards your request to whichever real provider you configured. A *gateway* and the **runtime router**.
> - **Wizard (`litellm-add`)** — this repo's CLI. It is the **control plane / configuration compiler**: it owns your settings and *generates* LiteLLM's `config.yaml`. LiteLLM itself does the live request routing.
> - **Provider** — a company/service that sells or gives away AI access (Google, OpenRouter…).
> - **Credential** — one API key/token belonging to a provider.
> - **Quota domain** — the upstream bucket that limits your usage. Several credentials may share one domain (e.g. several Google keys from the **same Google project** share that project's quota). **13 keys ≠ 13 quota pools** unless they live in 13 independent domains.
> - **Deployment** — one concrete route: provider + credential + endpoint + model. The wizard compiles these; you never address them directly.
> - **Model pool (alias)** — one gateway name (e.g. `deepseek-v4-flash`) fanning out to many deployments. This is what you put in requests.
> - **Role** — an app-level nickname pointing at pools in order (e.g. `fast` → `gemini-3.7-flash`, then `glm-5.3-flash`). Optional convenience.
> - **Health** — whether a deployment currently works (`healthy / throttled / invalid / unknown`). Different from *validation* (does the key authenticate?) and from *quota* (is the bucket saturated?).
> - **`~`** — shortcut for your home folder (`/home/yourname`). `~/.config/litellm` = a settings folder inside it.
> - **`venv`** — an isolated box holding the Python programs for this project, so they don't fight with system programs.
> - **`systemd` service** — a background task Linux starts automatically (here: on login) and restarts if it crashes.

---

# Part A — First-time setup (do once, top to bottom)

## 0. Get the wizard onto the new machine

This repo is **private**, so GitHub needs to know it's you. One-time login:

```bash
gh auth login
# Answer: GitHub.com → HTTPS → Yes → Login with a web browser,
# then open the shown code link on a machine where you're logged into GitHub.
```

Then fetch everything (copy-paste beats retyping 1000 lines into nano):

```bash
cd ~
gh repo clone mansourvery-hub/llm-proxy-wizard
ls llm-proxy-wizard   # wizard.py  sync-opencode.py  litellm.service  tests/  README.md
```

No `gh`? Alternatives, worst first:

- **Copy-paste via nano** (`nano wizard.py`, paste, save) — works but one missed line breaks the script; only for emergencies.
- **Browser download** — open the repo page → file → download, then move the files.
- **`git clone https://github.com/mansourvery-hub/llm-proxy-wizard.git`** — same as `gh repo clone`, Git will ask for your username + a
  personal access token as password (your normal password won't work).

All steps below assume the files sit in `~/llm-proxy-wizard/`.

## 1. What you need

- Linux (these steps were tested on Arch Linux; any distro works — only the install command in step 2 changes).
- Python 3.10+ (`python3 --version` to check).
- At least **one** provider API key, e.g.:
  - Google AI Studio key → https://aistudio.google.com/apikey (free tier)
  - OpenRouter key → https://openrouter.ai/keys (has free models)
  - You can add more providers later — the wizard loops, nothing is final.

## 2. Install the pieces

```bash
# 1) Python tools (Arch; on Ubuntu use: sudo apt install python3 python3-venv curl git github-cli)
sudo pacman -S --needed python python-virtualenv curl git github-cli

# 2) Folders
mkdir -p ~/.config/litellm ~/.config/systemd/user

# 3) Put the repo files into place (from section 0's clone):
cp ~/llm-proxy-wizard/wizard.py ~/.config/litellm/wizard.py
cp ~/llm-proxy-wizard/litellm.service ~/.config/systemd/user/litellm.service
# Make sure the wizard is runnable directly (cp doesn't always keep the exec bit):
chmod +x ~/.config/litellm/wizard.py

# 4) Isolated Python box + install inside it (takes a few minutes)
python3 -m venv ~/.config/litellm/venv
~/.config/litellm/venv/bin/pip install -U pip litellm pyyaml requests textual
# Equivalent, from a clone of this repo (pins the verified versions):
# ~/.config/litellm/venv/bin/pip install -r ~/llm-proxy-wizard/requirements.txt

# 5) Shortcuts so you can launch things by typing one word
echo "alias litellm-add='~/.config/litellm/venv/bin/python ~/.config/litellm/wizard.py'" >> ~/.zshrc
echo "alias llm-proxy-wizard='~/.config/litellm/venv/bin/python ~/llm-proxy-wizard/tui.py'" >> ~/.zshrc
source ~/.zshrc
```

## 3. Gateway password (mostly automatic now)

Everything on your machine talks to the gateway using a password called the **master key**.
The wizard resolves it as: `LITELLM_MASTER_KEY` environment variable → local secret
file `~/.config/litellm/.master_key` (mode `0600`) → auto-generated on first run.
The generated `config.yaml` references it as `os.environ/LITELLM_MASTER_KEY`, so the
secret never sits in the YAML itself.

```bash
# Recommended: set it once in your shell so every terminal program can use it:
echo 'export LITELLM_MASTER_KEY=put-a-long-random-value-here' >> ~/.zshrc
source ~/.zshrc   # or close + reopen the terminal
```

If you skip this, the wizard generates and stores one for you — but then only
processes that read the secret file (via the env var you export afterwards) can
authenticate. Either way: **never edit `wizard.py` to set a password** (v1 required
that; v2 does not).

## 4. Start the gateway automatically

```bash
systemctl --user daemon-reload
systemctl --user enable --now litellm.service
# Keep it running even when you log out (laptops can skip this):
loginctl enable-linger "$USER"
# Check it's alive:
systemctl --user status litellm.service --no-pager | head -n 8
```

`enable` = start on every login. If it ever crashes, systemd restarts it after 3 seconds.

## 5. Add your providers (the easy part)

The primary interface is the terminal app:

```bash
llm-proxy-wizard
```

Home is a table: one row per deployment with pool, provider, model, tier,
quota (shared quota reads like `10÷3 RPM`, plus the domain), status, and
key suffix. The header always shows how many models are working
(`3/21 models working · 97 untested`) — status fills itself in on launch
via one background probe per credential, and `P` re-probes on demand.
Context windows (from the installed LiteLLM model map, exact ids only)
and full quota/last-check details live in the row view. Providers and
tiers are color-coded. Anything needing attention is listed underneath.
**Configure** walks you through one provider at a
time: paste keys → they are checked in the background → answer at most one
question (do your keys share one usage limit?) → pick models from the live
list (free first, type to filter) → each pick is probe-tested → **Review**
shows what Apply will do. For OpenAI-compatible endpoints the base URL is
pre-filled (stored override, else the builtin default) — edit it only if
yours differs. Same-model pools across providers are grouped
automatically; uncertain ones are offered for one-confirm grouping.
**Test** checks every model through the gateway and helps park wrong-key
connections. **OpenCode view** shows exactly what OpenCode sees: every
`litellm/<alias>` grouped with its backing deployments, plus sync state.
**Apply** writes safely, restarts only if anything changed,
and offers the OpenCode sync.

Keyboard on Home: arrows navigate, `Enter`/click opens a row's detail and
actions (probe that credential, test via gateway), clicking a header sorts,
`/` filter, `s` cycle sort,
`S` reverse, `T` tier filter, `P` probe all (again to cancel), `x` clear filter,
`h` hide invalid, `c`/`t`/`v`/`o` jump to
Configure/Test/Review/OpenCode view, `q` quits, `Esc` goes back.
No network call ever blocks the UI.
Sanity check without the UI: `llm-proxy-wizard --check`.

The classic CLI (`litellm-add`) still works unchanged and shares the same
database — use either. What happens in the CLI, step by step:

1. You see only what you've already configured (empty at first). Type `add google`
   (names are fuzzy: `google`, `openrouter`, `claude`, `gpt`, `zen`, `zai`, `glm`…).
   `all` shows every provider; a bare number (`5`) works too.
2. **Paste your API keys**, then an empty line / `DONE`.
   The wizard **tests every key directly against the real provider right then**.
   You only move on when all keys pass (`R` retry, `K` keep valid ones, `S` save anyway, `A` abort).
3. **Quota grouping (Google-aware).** If you pasted several keys for a provider whose
   quota is project-scoped (Google), the wizard asks once whether they share one
   project — it never interrogates you per key, and a single key needs no questions
   at all. Details can always be refined later with the `quota` command.
4. **Pick models from the live catalog** — no guessing IDs, no Google searches.
   Free models are shown first; type `MORE` for the paid rest, `/text` to filter,
   numbers or names to select, `DONE` when happy.
5. The wizard **test-calls each model** (a minimal 1-word ping with `max_tokens: 1`).
   Broken/retired models are blocked before they can pollute your config.
   Default mode is **FAST** (one credential per model, cheapest). `mode` switches to
   **STRICT** (every credential × every model) or **SAMPLE** (a few credentials per
   model, one per quota domain first) when per-key access differences matter.
6. Repeat for more providers. **Q** compiles everything, restarts the gateway only if
   the config actually changed, quits. Use `plan` first anytime to preview the
   deployment/pool/role diff without writing anything.

Your secrets land in `~/.config/litellm/providers_db.json`, `config.yaml`, and
`.master_key` (all `chmod 600`, readable only by you).
**Never upload these files anywhere.** OpenCode sync copies alias *names* only —
never keys.

## 6. Prove it works (cheap checks)

```bash
# List everything the gateway currently serves:
curl -s http://localhost:4000/v1/models \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" | python3 -m json.tool | grep '"id"'

# One minimal ping through the whole chain:
curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gemini-3.5-flash-lite","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'
# HTTP 200 with a "choices" block = working. HTTP 429 = rate-limited, wait out the cooldown.
```

## 7. Use it in OpenCode CLI

**Automatic (recommended):** the repo's `sync-opencode.py` writes every gateway pool
(plus role aliases like `fast`) into `opencode.json` for you — no hand-editing,
no comma mistakes. Two ways to run it, pick one (they do exactly the same thing):

```bash
cd ~/llm-proxy-wizard && ./sync-opencode.py            # needs the exec bit (chmod +x)
python3 ~/llm-proxy-wizard/sync-opencode.py            # always works — the guide uses this form
```

(Why two spellings? `./file` relies on the file's executable permission, which can get
lost when copying. `python3 file` states the interpreter explicitly, so it works no
matter what. The script needs only system python — no venv, no packages.)

Want to see what it *would* do first? Add `--dry-run` — it only prints, changes nothing.
`--no-roles` syncs pools only, skipping role aliases:

```bash
python3 ~/llm-proxy-wizard/sync-opencode.py --dry-run
```

So: run with `--dry-run` when you're nervous, run it plain to actually apply.

It backs up `opencode.json` first (and restores the backup if it ever wrote invalid
JSON), only touches the managed `litellm` block, is safe to re-run after every wizard
change, and validates the result as strict JSON. Then restart the OpenCode TUI and open
`/models` — every pool appears as `litellm/<alias>`.

**Manual alternative:** in `~/.config/opencode/opencode.json`, inside the existing
`"provider"` section, add (watch the comma after the previous block):

```json
"litellm": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "Local LiteLLM",
  "options": {
    "baseURL": "http://localhost:4000/v1",
    "apiKey": "{env:LITELLM_MASTER_KEY}"
  },
  "models": {
    "gemini-3.5-flash-lite": { "name": "Gemini 3.5 Flash Lite (local)" },
    "minimax-m3:free": { "name": "MiniMax M3 free (local)" }
  }
}
```

List only the aliases you actually use — each becomes `litellm/<alias>` in OpenCode's
`/models` picker. Notes:

- The **model that answers is chosen by you** (`litellm/<alias>`). The gateway's
  *routing rule* (`usage-based-routing-v2`, pre-call checks, 60s cooldown after
  failures, retries only for transient server errors — never for auth/bad-request/
  rate-limit) picks *which deployment* serves it. That's already configured.
- Keep provider-native models where they belong: OpenCode Zen free models
  (`opencode/...`) work **only** inside OpenCode (they need its session protocol),
  so don't route those through LiteLLM.

## 8. Use it with anything else

Any OpenAI-compatible tool just needs two values:

```bash
export OPENAI_BASE_URL=http://localhost:4000/v1
export OPENAI_API_KEY="$LITELLM_MASTER_KEY"
```

---

# Part B — Future tweaks (come back here, skip Part A)

Setup is done — everything below reuses it. The rhythm is always:
**wizard (`litellm-add`) → Q to apply → re-sync OpenCode if pools changed.**

## Wizard commands (cheat sheet)

| Command | What |
|---|---|
| `add <name>` | add/configure a provider (fuzzy names, `add custom` for new endpoints) |
| `alias` | manage model pools: list / add / remove / apply suggestions |
| `role` | manage role aliases (`fast`, `smart`, …) → pools with ordered fallback |
| `quota` | inspect/assign/create/rename quota domains and limits |
| `mode` | validation mode: FAST / STRICT / SAMPLE |
| `plan` | dry-run: show deployment/pool/role diff, change nothing |
| `diagnose` | actionable checks: schema, stale aliases, quota, YAML, service, secrets |
| `pools` / `health` | compiled pool overview / deployment health test |
| `T` | **gateway smoke test** — one request per alias through LiteLLM |
| `P` | **pool test** — each underlying deployment directly (asks first when many) |
| `F` | **full sweep** — DB consistency + pool test + gateway test |
| `disable <provider>` / `enable <provider>` | exclude/include without deleting config |
| `quarantine <provider> <suffix>` | park a failing credential instead of deleting it |
| `remove <provider>` | delete a provider entry entirely |
| `Q` | compile → write (only if valid) → restart if changed → quit |
| `help` | plain-language glossary + full command list |

## Adding API keys later

```bash
litellm-add
# pick the provider (number or: add openrouter) → paste the new keys → DONE
# keys are tested immediately; models step unlocks only if all pass → Q
```

Adding keys never breaks OpenCode: the pools stay the same, so `opencode.json`
needs no update. (Re-running `sync-opencode.py` afterwards is harmless but unnecessary.)

## Removing API keys later

Same flow — the wizard shows your current keys numbered:

```bash
litellm-add
# pick the provider → type REMOVE → enter numbers (e.g. 1,3) → DONE → Q
```

Removing the *last* key of a provider aborts safely instead of saving a keyless
provider. If models changed as a side effect, re-run `sync-opencode.py`.

## Quota domains: why your 13 keys might be 1 pool

Upstream providers limit **quota buckets**, not key counts. All Google keys from one
project share that project's RPM/TPM. The wizard models this explicitly:

- `quota` shows every domain, its credentials, and its configured limits.
- Capacity is estimated **per unique domain** — shared keys never multiply it.
- Generated per-deployment RPM is split conservatively across deployments sharing
  one domain+model, so LiteLLM never believes `4 × 10 RPM` when reality is `10 RPM`.
- Adding another **provider or project** helps more than adding more keys to the
  same bucket. The status screen says so when everything depends on one provider.

## Adding a provider that's not in the list

Any API shaped like OpenAI works (DeepSeek direct, Groq, Mistral, xAI, Together…):
it needs `GET {base}/models` + `POST {base}/chat/completions` with a Bearer key.
Anthropic-native or Gemini-native APIs are not supported — LiteLLM speaks to them
through different protocols (the Zen lesson).

```bash
litellm-add
# pick 10 (or: add custom) → short name (e.g. DeepSeek direct) →
# base URL (e.g. https://api.deepseek.com/v1) → keys → pick models → Q
# shortcut: add <name> (e.g. add tokenharbor) — unknown names offer
# to create a custom endpoint with that name on the spot
```

The endpoint is stored alongside the keys, so keys, catalog, and tests all use it.
Your custom shows as `[C] Name (custom)` and is found by `add <name>`.
Afterwards: `sync-opencode.py` + restart the TUI, same as any model change.

## Adding / removing models later

Pick the provider in the wizard → keys validate → choose from the live catalog
(free first, `MORE` for paid, `DONE` to finish) → each model is ping-tested
(minimal 1-word probe) → Q. Then **always**:

```bash
python3 ~/llm-proxy-wizard/sync-opencode.py   # refresh opencode.json ...
```

…**and restart the OpenCode TUI** (it only reads config at startup), then `/models`.

Rule of thumb: **models changed → sync + restart TUI. Keys only → just Q.**
Quitting the wizard offers the sync itself when `opencode.json` is stale
(Enter = yes, anything else leaves it for a manual `sync-opencode.py` run).

If a previously configured model vanishes from the provider catalog, the wizard
asks whether to keep it (marked stale, excluded from healthy claims) or drop it —
it is never silently deleted.

## Roles: `fast`, `smart`, …

Roles are optional nicknames over pools with ordered fallback:

```bash
litellm-add
# role → Add → name: fast → primaries: gemini-3.7-flash glm-5.3-flash → fallback: gemini-3.5-flash-lite
```

Compiled to LiteLLM's `model_group_alias` + `fallbacks` (verified against the
installed LiteLLM), so `fast` tries its primary pools in order, then fallbacks.
Strict roles (`coder`, `vision`, `reasoning`) only accept deployments with verified
capabilities — `unknown` doesn't qualify unless you allow it.

## Updating the wizard itself

`litellm-add` doesn't run this repo's file — it runs a copy you installed at
`~/.config/litellm/wizard.py`. So after pulling new versions (or editing code
here), copy it over and keep the exec bit, otherwise `litellm-add` keeps using
the old code:

```bash
cp ~/llm-proxy-wizard/wizard.py ~/.config/litellm/wizard.py
chmod +x ~/.config/litellm/wizard.py
```

(Only `wizard.py` lives in both places. `sync-opencode.py` is meant to run
from the repo directly.)

Upgrading from v1 is automatic: the first v2 run migrates `providers_db.json`
in place (keys, models, aliases incl. legacy `_unified` are preserved; each key
becomes a credential in its own quota domain — regroup Google keys by project
with `quota` afterwards).

## 10. Daily use & troubleshooting

| Situation | Command |
|---|---|
| Add/remove keys or models | `llm-proxy-wizard` (Configure → Review → Apply; re-sync OpenCode if pools changed) |
| Same, classic CLI | `litellm-add` (Q applies + restarts; re-sync OpenCode if pools changed) |
| Quick self-check, no UI | `llm-proxy-wizard --check` |
| Is it running? | `systemctl --user status litellm.service --no-pager` |
| What broke? | `journalctl --user -u litellm.service -n 50 --no-pager` |
| Quick self-check | `litellm-add` → `diagnose` |
| Preview changes | `litellm-add` → `plan` |
| Apply config by hand | `systemctl --user restart litellm.service` |
| HTTP 429 | Free-tier throttle — waits out the 60s cooldown, router fails over |
| HTTP 401/403 | Wrong/expired provider key — re-run wizard for that provider (or `quarantine`) |
| HTTP 500 + `Connection error` | Provider unreachable or bad base URL — check logs |

## 11. Files in this repo

| File | What | Secrets? |
|---|---|---|
| `wizard.py` | Control plane: validation, catalogs, probes, quota/compiler, YAML gen | No |
| `engine.py` | Clean callable API over the wizard (used by the TUI; no logic of its own) | No |
| `tui.py` | Terminal app: Home/Configure/Test/Review screens (needs `textual`) | No |
| `requirements.txt` | Pinned, verified dependency set (`litellm`, `textual`, `pyyaml`, `requests`) | No |
| `sync-opencode.py` | Writes gateway pools (+roles) into `opencode.json` (backup + `--dry-run`) | No |
| `tests/` | Automated suite (`python -m unittest discover -s tests`, venv python) | No (fake keys only) |
| `litellm.service` | systemd unit that runs the gateway on port 4000 | No |
| `README.md` | This guide | No |

`providers_db.json`, `config.yaml`, and `.master_key` are **deliberately absent** — they hold your real keys.

## 12. Future work

- Support other LLM proxy backends (e.g. New-API, Bifrost) as alternatives to LiteLLM.
