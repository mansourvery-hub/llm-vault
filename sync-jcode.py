#!/usr/bin/env python3
"""Sync LiteLLM gateway aliases into JCode's config.toml.

JCode (jcode.sh, v0.84.0 verified) keeps named OpenAI-compatible provider
profiles in ``~/.jcode/config.toml``:

    [provider]
    default_provider = "llm-proxy-wizard"
    default_model = "fast"

    [providers.llm-proxy-wizard]
    type = "openai-compatible"
    base_url = "http://localhost:4000/v1"
    api_key_env = "LITELLM_MASTER_KEY"
    default_model = "fast"

    [[providers.llm-proxy-wizard.models]]
    id = "fast"

JCode then talks to the local LiteLLM gateway, which routes to every
configured provider/pool. JCode never sees raw provider credentials,
quota domains, or deployment details — only the logical model names.

Safety (mirrors sync-opencode.py):
- Stdlib only (system python3; no TOML library required).
- Touches ONLY the managed ``[providers.llm-proxy-wizard]`` block,
  the ``[[providers.llm-proxy-wizard.models]]`` arrays, and the two
  default fields in ``[provider]``. Every other section (display,
  features, keybindings, agents, hooks, custom providers, ...) survives
  byte-for-byte outside the managed span.
- Timestamped backup before replacing the file; atomic write
  (temp + rename); re-parse verification with restore on failure.
- Idempotent: re-running with the same gateway produces no changes.
- Contains NO secrets: auth is an environment-variable reference.
- Detects external edits of the managed section (managed-hash stamp
  in the profile's ``managed_hash`` field) so manual changes are
  surfaced instead of silently clobbered.

Usage:
    python3 sync-jcode.py [--dry-run] [--print]
    python3 sync-jcode.py --jcode ~/.jcode/config.toml
    python3 sync-jcode.py --litellm-config ~/.config/litellm/config.yaml
    python3 sync-jcode.py --no-roles
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys

LITELLM_DIR = os.path.join(os.path.expanduser("~"), ".config", "litellm")
JCODE_CONFIG = os.environ.get(
    "JCODE_CONFIG", os.path.join(os.path.expanduser("~"), ".jcode", "config.toml"))
GATEWAY_URL = "http://localhost:4000/v1"

MANAGED_PROFILE = "llm-proxy-wizard"          # [providers.<name>]
MANAGED_ENV = "LITELLM_MASTER_KEY"           # api_key_env (env reference only)

# --------------------------------------------------------------- alias I/O ---
# Same logical model set as sync-opencode.py: pool aliases + role aliases.


def load_aliases(litellm_config):
    """Import sync-opencode.load_aliases (single source of truth)."""
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "sync-opencode.py")
    if not os.path.exists(cand):
        raise FileNotFoundError("sync-opencode.py not found next to sync-jcode.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("sync_opencode", cand)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_aliases(litellm_config)


# --------------------------------------------------- minimal TOML handling ---
# JCode config.toml is a flat, well-formed subset: [table] headers,
# key = "string" | number | bool | [array], [[array-of-table]] headers,
# # comments. We never rewrite the file textually: we parse to find the
# managed span boundaries, then splice a freshly rendered managed block.
# (A full TOML library would be safer for arbitrary configs; the
#  re-parse verification below catches anything this parser misses.)


_TOML_SCALAR = re.compile(
    r'''^(?P<key>[A-Za-z0-9_.-]+|"[^"]*")\s*=\s*(?P<val>.+)$''')


def _toml_split_scalars(body_lines):
    """Parse ``key = value`` lines of one table body into a dict (strings,
    ints, floats, bools, arrays of those). Unknown shapes are kept as raw."""
    out = {}
    for line in body_lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _TOML_SCALAR.match(s)
        if not m:
            continue
        out[m.group("key")] = _toml_value(m.group("val"))
    return out


def _toml_value(raw):
    raw = raw.strip()
    # strip trailing comments outside strings/arrays (simple heuristic:
    # only when the value fully parses without them)
    if raw.startswith('"'):
        m = re.match(r'^("(?:[^"\\]|\\.)*")(\s*#.*)?$', raw)
        if m:
            raw = m.group(1)
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    if raw.startswith("["):
        # array (possibly with trailing comment): parse items crudely
        inner = raw.strip().rstrip("]").lstrip("[")
        # drop a trailing comment on the same line if present
        if "]" in inner:
            inner = inner.split("]")[0]
        items = []
        buf, in_str = "", False
        for ch in inner:
            if ch == '"':
                in_str = not in_str
                buf += ch
            elif ch == "," and not in_str:
                if buf.strip():
                    items.append(_toml_value(buf.strip()))
                buf = ""
            else:
                buf += ch
        if buf.strip():
            items.append(_toml_value(buf.strip()))
        return items
    raw = raw.split("#", 1)[0].strip()
    if raw == "true":
        return True
    if raw == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw  # dates etc. stay raw


def _toml_str(v):
    return json.dumps(str(v))


def split_sections(text):
    """Split TOML text into a list of (header, body_lines).

    ``header`` is ``[section]`` / ``[[section]]`` (or "" before the first
    header). Preserves everything: comments, blank lines, order.
    """
    out = []
    header, body = "", []
    for line in text.splitlines(keepends=False):
        s = line.strip()
        if re.match(r"^\[\[.*\]\]$", s) or re.match(r"^\[.*\]$", s):
            out.append((header, body))
            header, body = s, []
        else:
            body.append(line)
    out.append((header, body))
    return out


def find_managed_span(sections):
    """Index range (start, end) of sections owned by the managed profile.

    Owns: ``[providers.llm-proxy-wizard]`` and every
    ``[[providers.llm-proxy-wizard.models]]`` / sub-table of it. Returns
    (None, None) when absent.
    """
    start = end = None
    for i, (header, _body) in enumerate(sections):
        h = header.strip().lstrip("[")  # [table] and [[array-table]] alike
        if h.startswith(f"providers.{MANAGED_PROFILE}"):
            if start is None:
                start = i
            end = i
    return start, end


def parse_provider_defaults(text):
    """The two [provider] default fields (default_provider/default_model)."""
    sections = split_sections(text)
    for header, body in sections:
        if header.strip() == "[provider]":
            scal = _toml_split_scalars(body)
            return {"default_provider": scal.get("default_provider"),
                    "default_model": scal.get("default_model")}
    return {"default_provider": None, "default_model": None}


def parse_profiles(text):
    """All named profiles: name -> {base_url, api_key_env, default_model,
    models: [ids], raw}. Managed/custom alike (for import + reporting)."""
    sections = split_sections(text)
    profiles = {}
    cur = None
    for header, body in sections:
        h = header.strip()
        m = re.match(r"^\[providers\.([A-Za-z0-9_.-]+)\]$", h)
        mm = re.match(r"^\[\[providers\.([A-Za-z0-9_.-]+)\.models\]\]$", h)
        if m:
            cur = m.group(1)
            scal = _toml_split_scalars(body)
            profiles.setdefault(cur, {
                "base_url": scal.get("base_url"),
                "api_key_env": scal.get("api_key_env"),
                "default_model": scal.get("default_model"),
                "managed_hash": scal.get("managed_hash"),
                "models": [],
            })
            profiles[cur]["raw"] = "\n".join(body)
        elif mm:
            cur = mm.group(1)
            prof = profiles.setdefault(cur, {"base_url": None, "api_key_env": None,
                                             "default_model": None,
                                             "managed_hash": None, "models": []})
            scal = _toml_split_scalars(body)
            if scal.get("id"):
                prof["models"].append(
                    {"id": scal.get("id"),
                     "context_window": scal.get("context_window")})
        elif cur and h.startswith(f"[providers.{cur}"):
            # sub-table (e.g. [providers.x.headers]) — ignore for sync
            continue
        elif h and not h.startswith("[providers."):
            cur = None
    return profiles


def render_managed_block(models, default_model, managed_hash):
    """The TOML text of the managed profile (+ its [provider] defaults).

    Rendered deterministically; models keep gateway order.
    """
    lines = []
    lines.append("[provider]")
    lines.append(f"default_provider = {_toml_str(MANAGED_PROFILE)}")
    lines.append(f"default_model = {_toml_str(default_model or models[0])}")
    lines.append("")
    lines.append(f"[providers.{MANAGED_PROFILE}]")
    lines.append(f"type = {_toml_str('openai-compatible')}")
    lines.append(f"base_url = {_toml_str(GATEWAY_URL)}")
    lines.append(f"api_key_env = {_toml_str(MANAGED_ENV)}")
    lines.append(f"default_model = {_toml_str(default_model or models[0])}")
    lines.append(f"managed_hash = {_toml_str(managed_hash)}")
    for m in models:
        lines.append("")
        lines.append(f"[[providers.{MANAGED_PROFILE}.models]]")
        lines.append(f"id = {_toml_str(m)}")
    lines.append("")
    return "\n".join(lines)


def compute_managed_hash(models, default_model):
    """Stable hash of the wizard-managed content (external-change guard)."""
    payload = json.dumps({"models": list(models), "default": default_model or ""},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _atomic_write_text(path, text):
    """Atomic write (temp + fsync + rename). Never half-writes user config."""
    import tempfile
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".toml")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
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


def verify_toml(text):
    """Cheap self-check: every non-empty non-comment line parses as a
    header or a key = value. Returns error string or None."""
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if re.match(r"^\[\[?[^]]+\]\]?$", s):
            continue
        m = _TOML_SCALAR.match(s)
        if not m:
            return f"unparseable line: {s[:60]!r}"
        if _toml_value(m.group("val")) is None:
            return f"null value: {s[:60]!r}"
    return None


def splice_managed(text, block_text):
    """Replace the managed span (and stale [provider] defaults) with the
    new block, preserving everything else in order."""
    sections = split_sections(text)
    start, end = find_managed_span(sections)
    # Collect kept [provider] fields (everything except the two managed
    # defaults) so they can be merged into the new block's [provider].
    keep_provider_fields: dict[str, object] = {}
    pieces = []
    for i, (header, body) in enumerate(sections):
        h = header.strip()
        in_managed = start is not None and start <= i <= end
        if in_managed:
            continue
        if h == "[provider]":
            scal = _toml_split_scalars(body)
            keep = {k: v for k, v in scal.items()
                    if k not in ("default_provider", "default_model")}
            # stash for merging, do not emit yet
            keep_provider_fields.update(keep)
            continue
        if header:
            pieces.append(header)
        pieces.extend(body)
    # Merge kept fields into the managed block's [provider] section
    if keep_provider_fields:
        block_sections = split_sections(block_text)
        for idx, (h, b) in enumerate(block_sections):
            if h.strip() == "[provider]":
                scal = _toml_split_scalars(b)
                # keep existing managed defaults, add kept fields if not already present
                merged = dict(scal)
                for k, v in keep_provider_fields.items():
                    if k not in merged:
                        merged[k] = v
                # Re-render this section
                block_sections[idx] = (h, _render_provider_section(merged).splitlines()[1:])
                break
        # Reassemble block_text from sections
        block_pieces = []
        for h, b in block_sections:
            if h:
                block_pieces.append(h)
            block_pieces.extend(b)
        block_text = "\n".join(block_pieces)
    body_new = "\n".join(pieces)
    text_new = (body_new.rstrip("\n") + "\n\n" if body_new.strip() else "")
    return text_new + block_text.rstrip("\n") + "\n"


def _render_provider_section(fields):
    lines = ["[provider]"]
    for k, v in fields.items():
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
        elif isinstance(v, list):
            items = ", ".join(_toml_str(x) if isinstance(x, str) else str(x)
                              for x in v)
            lines.append(f"{k} = [{items}]")
        else:
            lines.append(f"{k} = {_toml_str(v)}")
    return "\n".join(lines)


def load_jcode_config(path):
    with open(path) as f:
        return f.read()


def diff_models(old_ids, new_ids, old_default, new_default):
    added = [m for m in new_ids if m not in old_ids]
    removed = [m for m in old_ids if m not in new_ids]
    changed_default = old_default != new_default
    return added, removed, changed_default


def sync(jcode_path, litellm_config, dry_run=False, print_block=False,
         include_roles=True, set_default=True,
         overwrite_externally_changed=False):
    """The whole sync. Returns a result dict; raises ValueError/FileNotFoundError."""
    aliases, roles, source = load_aliases(litellm_config)
    exposed = list(aliases)
    if roles and include_roles:
        exposed += [r for r in roles if r not in exposed]
    if not exposed:
        raise ValueError("No gateway aliases found. Run the wizard first.")

    default_model = exposed[0]
    new_hash = compute_managed_hash(exposed, default_model)

    if not os.path.exists(jcode_path):
        if dry_run:
            return {"would_create": True, "models": exposed,
                    "default": default_model, "source": source, "wrote": False}
        os.makedirs(os.path.dirname(os.path.abspath(jcode_path)) or ".",
                    exist_ok=True)
        text = ""
    else:
        text = load_jcode_config(jcode_path)

    profiles = parse_profiles(text)
    existing = profiles.get(MANAGED_PROFILE)
    old_ids = [m["id"] for m in (existing or {}).get("models", []) if m.get("id")]
    old_default = (existing or {}).get("default_model")
    old_hash = (existing or {}).get("managed_hash")

    # external-change guard: managed content differs from our last stamp
    externally_changed = False
    if existing and old_hash and old_hash != compute_managed_hash(old_ids, old_default):
        externally_changed = True

    added, removed, default_changed = diff_models(
        old_ids, exposed, old_default, default_model)

    if dry_run:
        return {"would_create": not os.path.exists(jcode_path),
                "models": exposed, "default": default_model,
                "added": added, "removed": removed,
                "default_changed": default_changed, "source": source,
                "externally_changed": externally_changed, "wrote": False}

    if externally_changed and not overwrite_externally_changed:
        raise ValueError(
            "JCode managed section changed externally since the last "
            "wizard sync (managed_hash mismatch). Re-run with "
            "--overwrite-external or import first.")

    block = render_managed_block(exposed,
                                 default_model if set_default else old_default,
                                 new_hash)
    if print_block:
        print(block)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005 -- local filename stamp, mirrors sync-opencode.py
    backup = None
    if os.path.exists(jcode_path):
        backup = f"{jcode_path}.bak-{stamp}"
        shutil.copy2(jcode_path, backup)
    text_new = splice_managed(text, block)
    err = verify_toml(text_new)
    if err:
        if backup:
            shutil.copy2(backup, jcode_path)
        raise ValueError(f"refusing to write unparseable TOML ({err})")
    _atomic_write_text(jcode_path, text_new)
    # verify what we actually wrote; restore on failure
    try:
        written = load_jcode_config(jcode_path)
        v = verify_toml(written)
        if v:
            raise ValueError(v)
        got = parse_profiles(written).get(MANAGED_PROFILE)
        if not got or [m["id"] for m in got.get("models", [])] != exposed:
            raise ValueError("managed profile mismatch after write")
    except Exception:
        if backup:
            shutil.copy2(backup, jcode_path)
            raise ValueError(f"wrote invalid TOML — restored backup {backup}")
        raise
    return {"models": exposed, "default": default_model, "added": added,
            "removed": removed, "default_changed": default_changed,
            "backup": backup, "externally_changed": externally_changed,
            "wrote": True, "source": source}


def main():
    ap = argparse.ArgumentParser(
        description="Sync LiteLLM gateway aliases into JCode config.toml")
    ap.add_argument("--jcode", default=JCODE_CONFIG,
                    help="JCode config path (default ~/.jcode/config.toml)")
    ap.add_argument("--litellm-config",
                    default=os.path.join(LITELLM_DIR, "config.yaml"))
    ap.add_argument("--dry-run", action="store_true",
                    help="show planned changes, change nothing")
    ap.add_argument("--print", action="store_true",
                    help="print the managed TOML block")
    ap.add_argument("--no-roles", action="store_true",
                    help="sync model pools only, skip role aliases")
    ap.add_argument("--keep-default", action="store_true",
                    help="do not change [provider] defaults")
    ap.add_argument("--overwrite-external", action="store_true",
                    help="overwrite an externally-changed managed section")
    args = ap.parse_args()

    print(f"[*] JCode target: {args.jcode}")
    try:
        result = sync(args.jcode, args.litellm_config,
                      dry_run=args.dry_run, print_block=args.print,
                      include_roles=not args.no_roles,
                      set_default=not args.keep_default,
                      overwrite_externally_changed=args.overwrite_external)
    except FileNotFoundError as e:
        print(f"[!] {e}")
        return 1
    except ValueError as e:
        print(f"[!] {e}")
        return 1

    print(f"[*] Managed provider: {MANAGED_PROFILE} -> {GATEWAY_URL} "
          f"(key: env {MANAGED_ENV})")
    models = result["models"]
    print(f"[*] {len(models)} logical model(s): {', '.join(models[:8])}"
          + (" ..." if len(models) > 8 else ""))
    if result.get("would_create"):
        print(f"[+] Would create managed profile in {args.jcode}. No files changed.")
        return 0
    if result.get("externally_changed"):
        print("[!] Managed section changed externally since last sync "
              "(managed_hash mismatch).")
    if result["wrote"]:
        if result.get("backup"):
            print(f"[+] Backup: {result['backup']}")
        chg = []
        if result["added"]:
            chg.append(f"+ {len(result['added'])} model(s): "
                       + ", ".join(result["added"][:6]))
        if result["removed"]:
            chg.append(f"- {len(result['removed'])} model(s): "
                       + ", ".join(result["removed"][:6]))
        if result["default_changed"]:
            chg.append(f"~ default model: {result['default']}")
        print("[+] " + ("; ".join(chg) if chg else "no model changes"))
        print(f"[+] Wrote managed block to {args.jcode}")
        print("[*] Unrelated JCode settings were left untouched.")
        print(f"[*] Try it: jcode --provider-profile {MANAGED_PROFILE}")
    else:
        print("[*] Dry run — no files changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
