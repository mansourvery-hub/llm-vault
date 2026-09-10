"""Shared test helpers: import wizard with isolated temp paths."""
import importlib
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_wizard():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import wizard as w
    importlib.reload(w)
    tmp = tempfile.mkdtemp(prefix="wiztest-")
    w.DB_FILE = os.path.join(tmp, "providers_db.json")
    w.YAML_FILE = os.path.join(tmp, "config.yaml")
    w.SECRET_FILE = os.path.join(tmp, ".master_key")
    return w


def load_sync():
    import importlib.util
    path = os.path.join(REPO, "sync-opencode.py")
    spec = importlib.util.spec_from_file_location("sync_opencode", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_jcode_sync():
    import importlib.util
    path = os.path.join(REPO, "sync-jcode.py")
    spec = importlib.util.spec_from_file_location("sync_jcode", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_gateway_config(path, pools=("gemini-3.7-flash",), roles=()):
    """A minimal gateway config.yaml with model_list + role aliases."""
    lines = ["model_list:"]
    for p in pools:
        lines += [f"  - model_name: {p}",
                  "    litellm_params:",
                  f"      model: gemini/{p}"]
    if roles:
        lines.append("model_group_alias:")
        for r in roles:
            first = pools[0] if pools else "x"
            lines.append(f"  {r}: {first}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


JCODE_USER_TOML = """\
# my jcode config — do not touch my settings
[server]
wake_mode = "internal"

[display]
theme = "my-theme"
emoji = true

[features]
memory = true

[providers.my-gateway]
type = "openai-compatible"
base_url = "https://api.example.com/v1"
api_key_env = "MY_GATEWAY_KEY"
default_model = "their-model"

[[providers.my-gateway.models]]
id = "their-model"
context_window = 128000

[ambient]
enabled = false
"""


def fake_db_two_keys_shared():
    return {
        "gemini": {"keys": ["K1-SECRET-AAA", "K2-SECRET-BBB"],
                   "models": ["gemini-3.7-flash"], "endpoints": []},
    }


def read_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


LEGACY_DB = {
    "gemini": {"keys": ["OLDKEY1", "OLDKEY2"],
               "models": ["gemini-3.7-flash", "gemini-3.1-pro-preview"],
               "endpoints": []},
    "openrouter": {"keys": ["ORKEY"], "models": ["minimax/minimax-m3:free"],
                   "endpoints": []},
    "_unified": {
        "deepseek-v4-flash": [
            {"provider": "custom_a", "model": "free/deepseek-v4-flash-0731"},
            {"provider": "custom_a", "model": "free/deepseek-v4-flash-0731"},
            "garbage-entry",
            {"provider": "", "model": ""},
        ]
    },
}
