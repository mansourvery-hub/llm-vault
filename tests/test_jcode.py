"""JCode integration: export, import, round trips, safety, idempotency.

Never touches the real ~/.jcode/config.toml or opencode.json — all paths
are temp-dir overrides (JCODE_CONFIG / EnginePaths.temp).
"""
import os
import unittest

from helpers import JCODE_USER_TOML, load_jcode_sync, write_gateway_config

import engine

sj = load_jcode_sync()


def read(path):
    with open(path) as f:
        return f.read()


def temp_setup(tmpdir, toml_text=JCODE_USER_TOML, with_config=True):
    """Temp paths + a seeded jcode config + gateway yaml."""
    paths = engine.EnginePaths.temp(tmpdir)
    if with_config:
        with open(paths.jcode_config, "w") as f:
            f.write(toml_text)
    write_gateway_config(paths.yaml_file,
                         pools=("gemini-3.7-flash", "fast"),
                         roles=("smart",))
    return paths


class JCodeSyncTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="jcode-")

    def test_creates_managed_profile(self):
        paths = temp_setup(self.tmp)
        r = sj.sync(paths.jcode_config, paths.yaml_file)
        self.assertTrue(r["wrote"])
        text = read(paths.jcode_config)
        self.assertIn("[providers.llm-proxy-wizard]", text)
        self.assertIn('type = "openai-compatible"', text)
        self.assertIn('base_url = "http://localhost:4000/v1"', text)
        self.assertIn('api_key_env = "LITELLM_MASTER_KEY"', text)
        self.assertIn("[[providers.llm-proxy-wizard.models]]", text)
        # pools + roles both exposed, in order
        self.assertEqual(r["models"],
                         ["gemini-3.7-flash", "fast", "smart"])
        # [provider] defaults point at the managed profile
        self.assertIn('default_provider = "llm-proxy-wizard"', text)
        self.assertIn('default_model = "gemini-3.7-flash"', text)

    def test_no_secrets_in_output(self):
        paths = temp_setup(self.tmp)
        sj.sync(paths.jcode_config, paths.yaml_file)
        text = read(paths.jcode_config)
        # env reference only — never a raw key
        self.assertNotIn("sk-", text)
        self.assertNotIn("SECRET", text)
        self.assertIn("LITELLM_MASTER_KEY", text)

    def test_preserves_unrelated_config_and_custom_providers(self):
        paths = temp_setup(self.tmp)
        sj.sync(paths.jcode_config, paths.yaml_file)
        text = read(paths.jcode_config)
        for section in ("[server]", "[display]", "[features]", "[ambient]",
                        "[providers.my-gateway]"):
            self.assertIn(section, text)
        # user's custom provider survived byte-for-byte (fields)
        self.assertIn('theme = "my-theme"', text)
        self.assertIn("wake_mode = \"internal\"", text)
        self.assertIn('base_url = "https://api.example.com/v1"', text)
        self.assertIn('id = "their-model"', text)
        self.assertIn('context_window = 128000', text)
        # comment survived
        self.assertIn("# my jcode config — do not touch my settings", text)

    def test_dry_run_changes_nothing(self):
        paths = temp_setup(self.tmp)
        before = read(paths.jcode_config)
        r = sj.sync(paths.jcode_config, paths.yaml_file, dry_run=True)
        self.assertFalse(r["wrote"])
        self.assertEqual(before, read(paths.jcode_config))

    def test_idempotent_second_sync_no_changes(self):
        paths = temp_setup(self.tmp)
        sj.sync(paths.jcode_config, paths.yaml_file)
        first = read(paths.jcode_config)
        r2 = sj.sync(paths.jcode_config, paths.yaml_file)
        self.assertTrue(r2["wrote"])
        self.assertEqual(first, read(paths.jcode_config))
        self.assertEqual(r2["added"], [])
        self.assertEqual(r2["removed"], [])

    def test_backup_created(self):
        paths = temp_setup(self.tmp)
        sj.sync(paths.jcode_config, paths.yaml_file)
        baks = [f for f in os.listdir(self.tmp) if f.startswith("jcode_config.toml.bak-")]
        self.assertEqual(len(baks), 1)

    def test_no_roles_option(self):
        paths = temp_setup(self.tmp)
        r = sj.sync(paths.jcode_config, paths.yaml_file, include_roles=False)
        self.assertEqual(r["models"], ["gemini-3.7-flash", "fast"])

    def test_externally_changed_detected_and_blocked(self):
        paths = temp_setup(self.tmp)
        sj.sync(paths.jcode_config, paths.yaml_file)
        # user edits the managed block behind our back
        text = read(paths.jcode_config)
        text = text.replace('id = "fast"', 'id = "fast-edited"')
        with open(paths.jcode_config, "w") as f:
            f.write(text)
        with self.assertRaises(ValueError):
            sj.sync(paths.jcode_config, paths.yaml_file)
        # overwrite flag proceeds
        r = sj.sync(paths.jcode_config, paths.yaml_file,
                    overwrite_externally_changed=True)
        self.assertTrue(r["wrote"])
        self.assertIn("fast", r["models"])
        self.assertNotIn("fast-edited", r["models"])

    def test_user_edit_outside_managed_block_is_kept(self):
        # edits OUTSIDE the managed span (user settings) never trigger the
        # external-change guard and survive sync
        paths = temp_setup(self.tmp)
        sj.sync(paths.jcode_config, paths.yaml_file)
        text = read(paths.jcode_config)
        text = text.replace('theme = "my-theme"', 'theme = "new-theme"')
        with open(paths.jcode_config, "w") as f:
            f.write(text)
        r = sj.sync(paths.jcode_config, paths.yaml_file)
        self.assertTrue(r["wrote"])
        self.assertFalse(r["externally_changed"])
        self.assertIn('theme = "new-theme"', read(paths.jcode_config))

    def test_creates_config_when_missing(self):
        paths = temp_setup(self.tmp, with_config=False)
        r = sj.sync(paths.jcode_config, paths.yaml_file)
        self.assertTrue(r["wrote"])
        text = read(paths.jcode_config)
        self.assertIn("[providers.llm-proxy-wizard]", text)

    def test_no_gateway_aliases_refuses(self):
        paths = temp_setup(self.tmp)
        write_gateway_config(paths.yaml_file, pools=())
        # isolate: no DB fallback either (honours LITELLM_DB_FILE)
        old = os.environ.get("LITELLM_DB_FILE")
        os.environ["LITELLM_DB_FILE"] = os.path.join(self.tmp, "db.json")
        try:
            with self.assertRaises((ValueError, FileNotFoundError)):
                sj.sync(paths.jcode_config, paths.yaml_file)
        finally:
            if old is None:
                del os.environ["LITELLM_DB_FILE"]
            else:
                os.environ["LITELLM_DB_FILE"] = old

    def test_cli_dry_run_main(self):
        # end-to-end through main() with --dry-run: nothing written
        import io
        import sys
        from contextlib import redirect_stdout
        paths = temp_setup(self.tmp)
        before = read(paths.jcode_config)
        argv = sys.argv
        sys.argv = ["sync-jcode.py", "--jcode", paths.jcode_config,
                    "--litellm-config", paths.yaml_file, "--dry-run"]
        try:
            with redirect_stdout(io.StringIO()):
                rc = sj.main()
        finally:
            sys.argv = argv
        self.assertEqual(rc, 0)
        self.assertEqual(before, read(paths.jcode_config))


class JCodeTOMLTest(unittest.TestCase):
    """The minimal TOML layer behind the sync."""

    def test_parse_profiles(self):
        profiles = sj.parse_profiles(JCODE_USER_TOML)
        self.assertEqual(sorted(profiles), ["my-gateway"])
        gw = profiles["my-gateway"]
        self.assertEqual(gw["base_url"], "https://api.example.com/v1")
        self.assertEqual(gw["api_key_env"], "MY_GATEWAY_KEY")
        self.assertEqual(gw["models"], [{"id": "their-model",
                                         "context_window": 128000}])

    def test_verify_toml_flags_garbage(self):
        self.assertIsNotNone(sj.verify_toml("this is not toml = = ="))
        self.assertIsNone(sj.verify_toml(JCODE_USER_TOML))

    def test_scalars(self):
        scal = sj._toml_split_scalars([
            's = "x"', 'i = 5', 'f = 1.5', 'b = true', 'a = ["x", 2]',
            'c = "v" # comment', 'q = "with \\"esc\\""',
        ])
        self.assertEqual(scal["s"], "x")
        self.assertEqual(scal["i"], 5)
        self.assertEqual(scal["f"], 1.5)
        self.assertIs(scal["b"], True)
        self.assertEqual(scal["a"], ["x", 2])
        self.assertEqual(scal["c"], "v")
        self.assertEqual(scal["q"], 'with "esc"')

    def test_managed_hash_stable(self):
        self.assertEqual(sj.compute_managed_hash(["a"], "a"),
                         sj.compute_managed_hash(["a"], "a"))
        self.assertNotEqual(sj.compute_managed_hash(["a"], "a"),
                            sj.compute_managed_hash(["b"], "a"))


class JCodeImportTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="jcodeimp-")

    def test_import_custom_profile(self):
        paths = temp_setup(self.tmp)
        db = engine.load_state(paths)
        r = engine.import_jcode(db, paths)
        self.assertTrue(r["ok"])
        self.assertEqual(len(r["providers"]), 1)
        p = r["providers"][0]
        self.assertEqual(p["name"], "my-gateway")
        self.assertTrue(p["created"])
        self.assertEqual(p["pid"], "custom_my_gateway")
        self.assertEqual(p["new_models"], ["their-model"])
        # stored shape
        entry = db["custom_my_gateway"]
        self.assertEqual(entry["base_url"], "https://api.example.com/v1")
        self.assertEqual(entry["label"], "my-gateway")
        self.assertEqual(entry["models"], ["their-model"])

    def test_import_idempotent_no_duplicates(self):
        paths = temp_setup(self.tmp)
        db = engine.load_state(paths)
        engine.import_jcode(db, paths)
        db2 = engine.load_state(paths)
        db2.update(db)  # simulate the same db persisted + re-imported
        r2 = engine.import_jcode(db2, paths)
        self.assertFalse(r2["providers"][0]["created"])
        custom = [k for k in db2 if k.startswith("custom_")]
        self.assertEqual(custom, ["custom_my_gateway"])
        self.assertEqual(db2["custom_my_gateway"]["models"], ["their-model"])

    def test_import_skips_managed_profile(self):
        # a jcode config whose ONLY profile is the wizard's managed one
        paths = temp_setup(self.tmp, toml_text="")
        sj.sync(paths.jcode_config, paths.yaml_file)
        db = engine.load_state(paths)
        r = engine.import_jcode(db, paths)
        self.assertEqual(r["managed_models"],
                         ["gemini-3.7-flash", "fast", "smart"])
        self.assertEqual(r["providers"], [])
        self.assertNotIn("custom_llm_proxy_wizard", db)

    def test_missing_config(self):
        import tempfile
        paths = engine.EnginePaths.temp(tempfile.mkdtemp())
        db = engine.load_state(paths)
        r = engine.import_jcode(db, paths)
        self.assertFalse(r["ok"])
        self.assertIn("not found", r["note"])

    def test_malformed_config(self):
        paths = temp_setup(self.tmp, toml_text="[broken\nnot toml @@@ =")
        db = engine.load_state(paths)
        r = engine.import_jcode(db, paths)
        # malformed -> ok=False reported, never raised, DB untouched
        self.assertFalse(r["ok"])

    def test_never_prints_or_stores_secrets(self):
        # api_key_env references are env names — no secret material exists
        # in a jcode config; the import must not invent any either
        paths = temp_setup(self.tmp)
        db = engine.load_state(paths)
        engine.import_jcode(db, paths)
        import json
        raw = json.dumps(db.get("custom_my_gateway", {}))
        self.assertNotIn("MY_GATEWAY_KEY", raw)  # env NAME not needed either


class EngineJCodeFacadeTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="jcodeeng-")
        self.paths = temp_setup(self.tmp)

    def test_sync_jcode_facade(self):
        r = engine.sync_jcode(self.paths)
        self.assertTrue(r["wrote"])
        self.assertEqual(len(r["models"]), 3)

    def test_sync_jcode_dry_run(self):
        before = read(self.paths.jcode_config)
        r = engine.sync_jcode(self.paths, dry_run=True)
        self.assertFalse(r["wrote"])
        self.assertEqual(before, read(self.paths.jcode_config))

    def test_jcode_differs(self):
        self.assertTrue(engine.jcode_differs(self.paths))  # no profile yet
        engine.sync_jcode(self.paths)
        self.assertFalse(engine.jcode_differs(self.paths))
        # change the gateway set -> stale again
        write_gateway_config(self.paths.yaml_file,
                             pools=("gemini-3.7-flash", "fast", "extra-pool"))
        self.assertTrue(engine.jcode_differs(self.paths))

    def test_jcode_status(self):
        st = engine.jcode_status(self.paths)
        self.assertTrue(st["config_exists"])
        self.assertTrue(st["parseable"])
        self.assertIn("my-gateway", st["profiles"])
        self.assertFalse(st["managed_profile"])
        engine.sync_jcode(self.paths)
        st = engine.jcode_status(self.paths)
        self.assertTrue(st["managed_profile"])

    def test_paths_jcode_env_override(self):
        os.environ["JCODE_CONFIG"] = "/tmp/definitely-not-here.toml"
        try:
            p = engine.EnginePaths.from_env()
            self.assertEqual(p.jcode_config, "/tmp/definitely-not-here.toml")
        finally:
            del os.environ["JCODE_CONFIG"]


class RoundTripTest(unittest.TestCase):
    """JCode -> Wizard -> JCode and OpenCode -> Wizard -> JCode."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="jcodert-")
        self.paths = engine.EnginePaths.temp(self.tmp)

    def test_jcode_roundtrip(self):
        # seed a jcode config with a custom gateway
        with open(self.paths.jcode_config, "w") as f:
            f.write(JCODE_USER_TOML)
        db = engine.load_state(self.paths)
        r1 = engine.import_jcode(db, self.paths)
        self.assertEqual(r1["providers"][0]["name"], "my-gateway")
        engine.save_state(db, self.paths)
        # compile the wizard DB -> gateway config (needs a usable pool)
        db = engine.load_state(self.paths)
        engine.set_models(db, "custom_my_gateway", ["their-model"])
        engine.save_state(db, self.paths)
        engine.compile_config(db)
        # import must not have destroyed anything; sync back to jcode
        r2 = engine.sync_jcode(self.paths)
        self.assertTrue(r2["wrote"])
        # the user's own profile survived the sync back
        text = read(self.paths.jcode_config)
        self.assertIn('base_url = "https://api.example.com/v1"', text)
        self.assertIn('id = "their-model"', text)
        # and the managed profile exists alongside it
        self.assertIn("[providers.llm-proxy-wizard]", text)

    def test_opencode_to_jcode_roundtrip(self):
        # OpenCode with a DIRECT provider (case B)
        oc = {
            "provider": {
                "direct-api": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {"baseURL": "https://api.deepseek.com/v1",
                                "apiKey": "sk-SECRET-123"},
                    "models": {"deepseek-chat": {"name": "deepseek-chat"}},
                },
            },
        }
        import json
        with open(self.paths.opencode_json, "w") as f:
            json.dump(oc, f)
        db = engine.load_state(self.paths)
        r = engine.import_opencode(db, self.paths)
        self.assertEqual(len(r["providers"]), 1)
        p = r["providers"][0]
        self.assertEqual(p["name"], "direct-api")
        self.assertEqual(p["pid"], "custom_direct_api")
        # secret stored as a credential, NOT in a log/report field
        self.assertNotIn("sk-SECRET-123", json.dumps(r))
        entry = db["custom_direct_api"]
        self.assertEqual(entry["models"], ["deepseek-chat"])
        # wizard keeps the secret in the credential store only
        self.assertIn("sk-SECRET-123", entry["keys"])
        engine.save_state(db, self.paths)
        # export to jcode: NO raw key in the TOML, env ref only
        write_gateway_config(self.paths.yaml_file, pools=("deepseek-chat",))
        engine.sync_jcode(self.paths)
        text = read(self.paths.jcode_config)
        self.assertNotIn("sk-SECRET-123", text)
        self.assertNotIn("deepseek.com", text)
        self.assertIn('api_key_env = "LITELLM_MASTER_KEY"', text)

    def test_opencode_litellm_block_not_reimported(self):
        # Case A: opencode already points at OUR gateway
        import json
        oc = {
            "provider": {
                "litellm": {
                    "options": {"baseURL": "http://localhost:4000/v1",
                                "apiKey": "{env:LITELLM_MASTER_KEY}"},
                    "models": {"fast": {"name": "fast"}},
                },
            },
        }
        with open(self.paths.opencode_json, "w") as f:
            json.dump(oc, f)
        db = engine.load_state(self.paths)
        r = engine.import_opencode(db, self.paths)
        self.assertEqual(r["providers"], [])
        self.assertEqual(r["gateway_models"], ["fast"])
        # no custom_* fork of the gateway itself
        self.assertNotIn("custom_litellm", db)

    def test_opencode_import_idempotent(self):
        import json
        oc = {
            "provider": {
                "direct-api": {
                    "options": {"baseURL": "https://api.deepseek.com/v1"},
                    "models": {"deepseek-chat": {}, "deepseek-reasoner": {}},
                },
            },
        }
        with open(self.paths.opencode_json, "w") as f:
            json.dump(oc, f)
        db = engine.load_state(self.paths)
        engine.import_opencode(db, self.paths)
        engine.save_state(db, self.paths)
        db2 = engine.load_state(self.paths)
        r2 = engine.import_opencode(db2, self.paths)
        self.assertFalse(r2["providers"][0]["created"])
        self.assertEqual(r2["providers"][0]["new_models"], [])
        customs = [k for k in db2 if k.startswith("custom_")]
        self.assertEqual(customs, ["custom_direct_api"])

    def test_opencode_missing_and_malformed(self):
        db = engine.load_state(self.paths)
        r = engine.import_opencode(db, self.paths)
        self.assertFalse(r["ok"])
        # malformed JSONC-free garbage
        with open(self.paths.opencode_json, "w") as f:
            f.write("{broken json,,,")
        r2 = engine.import_opencode(db, self.paths)
        self.assertFalse(r2["ok"])
        self.assertIn("malformed", r2["note"])

    def test_opencode_jsonc_with_comments(self):
        # JSONC tolerance via the shared parser
        with open(self.paths.opencode_json, "w") as f:
            f.write("""{
  // my provider
  "provider": {
    "direct-api": { /* inline */
      "options": {"baseURL": "https://x.example/v1",},
      "models": {"m1": {}},
    },
  },
}""")
        db = engine.load_state(self.paths)
        r = engine.import_opencode(db, self.paths)
        self.assertEqual(len(r["providers"]), 1)
        self.assertEqual(db["custom_direct_api"]["models"], ["m1"])


if __name__ == "__main__":
    unittest.main()
