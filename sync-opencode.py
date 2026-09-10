#!/usr/bin/env python3
"""Sync LiteLLM gateway aliases into OpenCode CLI config.

Reads the user-facing aliases your LiteLLM gateway serves (model pools from
model_list plus role aliases from model_group_alias — never raw deployment
details) and writes them as a "litellm" provider block into opencode.json —
so every gateway pool shows up in OpenCode's /models picker as
litellm/<alias>.

- Stdlib only (works with system python3, no venv needed).
- Backs up opencode.json before touching it.
- Tolerates JSONC (comments + trailing commas) in opencode.json.
- Idempotent: re-running just refreshes the litellm block.
- Contains NO secrets: auth uses the {env:LITELLM_MASTER_KEY} placeholder.

Usage:
    python3 sync-opencode.py [--dry-run] [--print]
    python3 sync-opencode.py --opencode ~/.config/opencode/opencode.json \\
        --litellm-config ~/.config/litellm/config.yaml
"""
import argparse
import datetime
import json
import os
import re
import shutil
import sys

LITELLM_DIR = os.path.join(os.path.expanduser("~"), ".config", "litellm")
OPENCODE_JSON = os.environ.get("OPENCODE_JSON",
                               os.path.join(os.path.expanduser("~"), ".config", "opencode", "opencode.json"))
GATEWAY_URL = "http://localhost:4000/v1"


def _strip_jsonc(text):
    """Remove // and /* */ comments (outside strings) + trailing commas."""
    out, i, n, in_str, esc = [], 0, len(text), False, False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            esc = (c == "\\" and not esc)
            if c == '"' and not esc:
                in_str = False
            elif c != "\\":
                esc = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    cleaned = "".join(out)
    return re.sub(r",(\s*[}\]])", r"\1", cleaned)


def _atomic_write_json(path, data):
    """Atomic write (temp + fsync + rename). Never half-writes user config."""
    import tempfile
    d = os.path.dirname(os.path.abspath(path)) or "."
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
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_aliases(litellm_config):
    """Unique user-facing gateway aliases, in config order.

    Returns (aliases, roles, source) where roles are model_group_alias
    entries (app-level names like fast/smart). Only logical pools and
    roles are exposed — never raw provider/credential deployment details.
    """
    roles = []
    if litellm_config and os.path.exists(litellm_config):
        # Fast path: regex over model_name lines (no PyYAML needed).
        seen, aliases = set(), []
        with open(litellm_config) as f:
            text = f.read()
        in_alias = False
        for line in text.splitlines():
            if re.match(r"^\s*model_group_alias:\s*$", line):
                in_alias = True
                continue
            if in_alias:
                m = re.match(r"^\s{2,}(\S+):\s*(\S.*)?$", line)
                if m and not line.strip().startswith("-"):
                    name = m.group(1).rstrip(":")
                    if name and name not in seen and name not in ("model", "hidden"):
                        seen.add(name)
                        roles.append(name)
                    continue
                if re.match(r"^[a-z_]+:\s*$", line):
                    in_alias = False
            m = re.match(r"^\s*(?:-\s*)?model_name:\s*(\S+)\s*$", line)
            if m and m.group(1) not in seen:
                seen.add(m.group(1))
                aliases.append(m.group(1))
        if aliases or roles:
            return aliases, roles, f"config ({litellm_config})"
        try:
            import yaml  # type: ignore
            with open(litellm_config) as f:
                cfg = yaml.safe_load(f) or {}
            seen, aliases, roles = set(), [], []
            for item in (cfg.get("model_list") or []):
                a = item.get("model_name")
                if a and a not in seen:
                    seen.add(a)
                    aliases.append(a)
            for r in (cfg.get("model_group_alias") or {}):
                if r not in seen:
                    seen.add(r)
                    roles.append(r)
            if aliases or roles:
                return aliases, roles, f"config ({litellm_config})"
        except ImportError:
            pass  # fall through to providers_db.json
    db_path = os.environ.get("LITELLM_DB_FILE",
                             os.path.join(LITELLM_DIR, "providers_db.json"))
    with open(db_path) as f:
        db = json.load(f)
    # Unified aliases: single gateway name -> many providers (evade rate limits)
    aliases_map = db.get("_aliases") or db.get("_unified") or {}
    seen, aliases = set(), []
    # Unified first (keep declared order)
    for canon in aliases_map:
        if canon and canon not in seen:
            seen.add(canon)
            aliases.append(canon)
    # Track covered (pid,model) so bare aliases don't double-show
    covered = set()
    for members in aliases_map.values():
        if not isinstance(members, list):
            continue
        for mem in members:
            if isinstance(mem, dict) and mem.get("provider") and mem.get("model"):
                covered.add((mem["provider"], mem["model"]))
    pids = ["gemini", "openrouter", "ollama_cloud", "zai", "tokenrouter",
            "anthropic", "openai", "opencode_zen"]
    pids += sorted(k for k in db if k.startswith("custom_") and k not in pids)
    for pid in pids:
        for m in db.get(pid, {}).get("models", []):
            if (pid, m) in covered:
                continue
            # LiteLLM alias convention: last segment (matches model_name in config.yaml)
            a = m.split("/")[-1] if "/" in m else m
            if a not in seen:
                seen.add(a)
                aliases.append(a)
    roles = [r for r in (db.get("_roles") or {}) if r not in seen]
    return aliases, roles, f"provider DB ({db_path})"


def main():
    ap = argparse.ArgumentParser(description="Sync LiteLLM aliases into opencode.json")
    ap.add_argument("--opencode", default=OPENCODE_JSON)
    ap.add_argument("--litellm-config", default=os.path.join(LITELLM_DIR, "config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="print block, change nothing")
    ap.add_argument("--print", action="store_true", help="print resulting litellm block")
    ap.add_argument("--no-roles", action="store_true",
                    help="sync model pools only, skip role aliases")
    ap.add_argument("--only-pools", action="store_true",
                    help="alias for --no-roles: expose pools, not roles")
    args = ap.parse_args()

    aliases, roles, source = load_aliases(args.litellm_config)
    if not aliases and not roles:
        print("[!] No gateway aliases found. Run the wizard (litellm-add) first.")
        sys.exit(1)
    exposed = list(aliases)
    if roles and not (args.no_roles or args.only_pools):
        exposed += [r for r in roles if r not in exposed]
    print(f"[*] {len(aliases)} pools" + (f" + {len(roles)} roles" if roles else "") +
          f" from {source}" + ("" if roles and not (args.no_roles or args.only_pools)
                               else (" (roles excluded)" if roles else "")))

    block = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Local LiteLLM",
        "options": {
            "baseURL": GATEWAY_URL,
            "apiKey": "{env:LITELLM_MASTER_KEY}"
        },
        "models": {a: {"name": a} for a in exposed},
    }

    if args.dry_run:
        print(json.dumps({"litellm": block}, indent=2))
        return

    if not os.path.exists(args.opencode):
        print(f"[!] Not found: {args.opencode}")
        sys.exit(1)
    with open(args.opencode) as f:
        raw = f.read()
    try:
        cfg = json.loads(_strip_jsonc(raw))
    except json.JSONDecodeError as e:
        print(f"[!] Could not parse {args.opencode}: {e}")
        print("[!] Doing nothing — fix the JSON and re-run.")
        sys.exit(1)

    # Managed block only: unrelated providers and keys stay untouched.
    cfg.setdefault("provider", {})["litellm"] = block
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = f"{args.opencode}.bak-{stamp}"
    shutil.copy2(args.opencode, backup)
    _atomic_write_json(args.opencode, cfg)
    # strict-JSON sanity check on what we just wrote; restore on failure
    try:
        with open(args.opencode) as f:
            json.load(f)
    except ValueError:
        shutil.copy2(backup, args.opencode)
        print(f"[!] Wrote invalid JSON — restored backup {backup}.")
        sys.exit(1)
    print(f"[+] Backup: {backup}")
    print(f"[+] Wrote litellm block with {len(exposed)} models to {args.opencode}")
    if args.print:
        print(json.dumps({"litellm": block}, indent=2))
    print("[*] Restart the OpenCode TUI, then /models -> litellm/<alias>.")


if __name__ == "__main__":
    main()
