"""Engine facade tests (Milestone 1). Temp dirs + fake keys only."""
import json
import os
import stat
import tempfile
import unittest

import engine


def tmp_paths():
    tmp = tempfile.mkdtemp(prefix="engtest-")
    return engine.EnginePaths.temp(tmp), tmp


class PathsTest(unittest.TestCase):
    def test_env_overrides(self):
        tmp = tempfile.mkdtemp(prefix="engtest-")
        os.environ["LITELLM_DB_FILE"] = os.path.join(tmp, "db.json")
        try:
            p = engine.EnginePaths.from_env()
            self.assertEqual(p.db_file, os.path.join(tmp, "db.json"))
        finally:
            del os.environ["LITELLM_DB_FILE"]


class StateTest(unittest.TestCase):
    def test_roundtrip_preserves_unknown_fields(self):
        p, _ = tmp_paths()
        db = engine.load_state(p)
        db["_mystery_plugin"] = {"keep": 1}
        db["gemini"] = {"keys": [], "models": [], "endpoints": []}
        engine.save_state(db, p)
        back = engine.load_state(p)
        self.assertEqual(back["_mystery_plugin"], {"keep": 1})
        mode = stat.S_IMODE(os.stat(p.db_file).st_mode)
        self.assertEqual(oct(mode), "0o600")

    def test_credential_id_stable(self):
        self.assertEqual(engine.credential_id("abc"), engine.credential_id("abc"))
        self.assertTrue(engine.credential_id("abc").startswith("cred-"))
        self.assertNotIn("abc", engine.credential_id("a-very-long-secret-abc"))

    def test_mask_never_leaks(self):
        self.assertNotIn("sk-abcdef", engine.mask_secret("sk-abcdef123456"))


class CredentialsTest(unittest.TestCase):
    def test_add_and_list_and_remove(self):
        db = engine.load_state(engine.EnginePaths.temp(tempfile.mkdtemp()))
        ids = engine.add_credentials(db, "gemini", ["K1-FAKE", "K2-FAKE"])
        self.assertEqual(len(ids), 2)
        creds = engine.list_credentials(db, "gemini")
        self.assertEqual(len(creds), 2)
        for c in creds:  # secret-free summaries
            self.assertIn("suffix", c)
            self.assertNotIn("K1-FAKE", json.dumps(c))
            self.assertNotIn("K2-FAKE", json.dumps(c))
        n = engine.remove_credentials(db, "gemini", [creds[0]["id"]])
        self.assertEqual(n, 1)
        self.assertEqual(len(engine.list_credentials(db, "gemini")), 1)

    def test_capacity_counts_unique_domains_not_keys(self):
        db = engine.load_state(engine.EnginePaths.temp(tempfile.mkdtemp()))
        engine.add_credentials(db, "gemini", ["K1-FAKE", "K2-FAKE"])
        engine.set_quota_domains(db, "gemini", "shared")
        cap = engine.calculate_capacity(db, [("gemini", "gemini-3.7-flash")])
        self.assertEqual(cap["domains"], 1)

    def test_grouping_question_single_key_no_ask(self):
        import wizard as w
        db = w.migrate_db({"gemini": {"keys": ["ONLY-FAKE"], "models": [],
                                      "endpoints": []}})
        self.assertFalse(engine.needs_grouping_question(db, "gemini"))

    def test_grouping_multi_key_asks_then_reviewed(self):
        db = engine.load_state(engine.EnginePaths.temp(tempfile.mkdtemp()))
        engine.add_credentials(db, "gemini", ["K1-FAKE", "K2-FAKE"])
        self.assertTrue(engine.needs_grouping_question(db, "gemini"))
        self.assertEqual(engine.set_quota_domains(db, "gemini", "later"), "later")
        self.assertFalse(engine.needs_grouping_question(db, "gemini"))
        self.assertTrue(db["gemini"]["quota_reviewed"])

    def test_models_get_set_and_catalog_stamp(self):
        db = engine.load_state(engine.EnginePaths.temp(tempfile.mkdtemp()))
        self.assertEqual(engine.get_models(db, "gemini"), [])
        self.assertEqual(engine.set_models(db, "gemini", ["b", "a", "b", ""]),
                         ["b", "a"])
        self.assertEqual(engine.get_models(db, "gemini"), ["b", "a"])
        engine.mark_catalog_checked(db, "gemini")
        self.assertEqual(db["gemini"]["catalog_source"], "live provider catalog")

    def test_free_first_ordering_and_modes(self):
        catalog = [("pro-model-paid", "Paid"), ("free-thing:free", "Free"),
                   ("alpha-paid", "Paid")]
        ordered = engine.order_catalog_free_first(catalog)
        self.assertEqual(ordered[0][0], "free-thing:free")
        self.assertTrue(engine.is_free_model("x:free"))
        self.assertFalse(engine.is_free_model("pro-model-paid"))
        db = engine.load_state(engine.EnginePaths.temp(tempfile.mkdtemp()))
        self.assertEqual(engine.get_validation_mode(db), "FAST")
        self.assertEqual(engine.get_sample_size(db), 2)
        db["_settings"]["validation_mode"] = "BOGUS"
        self.assertEqual(engine.get_validation_mode(db), "FAST")


class CompilerTest(unittest.TestCase):
    def test_compile_and_write_and_overview(self):
        p, _ = tmp_paths()
        db = engine.load_state(p)
        engine.add_credentials(db, "gemini", ["GK1-FAKE"])
        db["gemini"]["models"] = ["gemini-3.7-flash"]
        deps, _pools, _roles, errors = engine.compile_config(db)
        self.assertEqual(errors, [])
        self.assertEqual(len(deps), 1)
        n = engine.write_config(db, p)
        self.assertEqual(n, 1)
        mode = stat.S_IMODE(os.stat(p.yaml_file).st_mode)
        self.assertEqual(oct(mode), "0o600")
        ov = engine.gateway_overview(db, p, status="running")
        self.assertIn("gemini-3.7-flash", ov["pools"])

    def test_combine_tier_safety(self):
        db = engine.load_state(engine.EnginePaths.temp(tempfile.mkdtemp()))
        # different stems are MANUAL -> must not silently merge
        with self.assertRaises(ValueError):
            engine.combine_models(db, "mixed-pool",
                                  [("gemini", "gemini-3.7-flash"),
                                   ("ollama_local", "qwen2.5-coder:7b")])
        engine.combine_models(db, "flash-pool",
                              [("gemini", "gemini-3.7-flash"),
                               ("openrouter", "gemini-3.7-flash")])
        self.assertIn("flash-pool", engine.get_combined_models(db))


class ApplyTest(unittest.TestCase):
    def test_apply_no_restart_when_unchanged(self):
        p, _ = tmp_paths()
        db = engine.load_state(p)
        engine.add_credentials(db, "gemini", ["GK1-FAKE"])
        db["gemini"]["models"] = ["gemini-3.7-flash"]
        calls = []

        def runner(*a, **k):
            calls.append(a)
            class R:
                returncode = 0
            return R()

        r1 = engine.apply_config(db, p, runner=runner, wait_fn=lambda t: True)
        self.assertTrue(r1["changed"])
        self.assertTrue(r1["restarted"])
        calls.clear()
        r2 = engine.apply_config(db, p, runner=runner, wait_fn=lambda t: True)
        self.assertFalse(r2["changed"])
        self.assertFalse(r2["restarted"])
        self.assertEqual(calls, [])

    def test_gateway_status_mockable(self):
        class R:
            stdout = "active\n"
            returncode = 0
        self.assertEqual(engine.gateway_status(runner=lambda *a, **k: R()), "running")


class SyncTest(unittest.TestCase):
    def test_sync_dry_run_then_write(self):
        p, tmp = tmp_paths()
        db = engine.load_state(p)
        engine.add_credentials(db, "gemini", ["GK1-FAKE"])
        db["gemini"]["models"] = ["gemini-3.7-flash"]
        engine.write_config(db, p)
        oc = os.path.join(tmp, "opencode.json")
        with open(oc, "w") as f:
            json.dump({"provider": {}}, f)
        p.opencode_json = oc
        r = engine.sync_opencode(p, dry_run=True)
        self.assertFalse(r["wrote"])
        self.assertIn("gemini-3.7-flash", r["exposed"])
        r2 = engine.sync_opencode(p)
        self.assertTrue(r2["wrote"])
        with open(oc) as f:
            cfg = json.load(f)
        self.assertIn("gemini-3.7-flash", cfg["provider"]["litellm"]["models"])
        self.assertEqual(cfg["provider"]["litellm"]["options"]["apiKey"],
                         "{env:LITELLM_MASTER_KEY}")
        with open(oc) as f:
            on_disk = f.read()
        self.assertNotIn("GK1-FAKE", on_disk)
        self.assertTrue(any(n.startswith("opencode.json.bak-") for n in os.listdir(tmp)))


class CombiningTest(unittest.TestCase):
    def test_auto_combine_same_stem_flash(self):
        db = engine.load_state(engine.EnginePaths.temp(tempfile.mkdtemp()))
        engine.add_credentials(db, "gemini", ["GK1-FAKE"])
        engine.add_credentials(db, "openrouter", ["OR1-FAKE"])
        db["gemini"]["models"] = ["gemini-3.7-flash"]
        db["openrouter"]["models"] = ["gemini-3.7-flash"]
        created = engine.auto_combine(db, "openrouter", ["gemini-3.7-flash"])
        self.assertIn("gemini-3.7-flash", created)
        _deps, pools, _, errors = engine.compile_config(db)
        self.assertEqual(errors, [])
        self.assertEqual(len(pools["gemini-3.7-flash"]), 2)

    def test_pending_suggestions_only_uncertain(self):
        db = engine.load_state(engine.EnginePaths.temp(tempfile.mkdtemp()))
        engine.add_credentials(db, "openrouter", ["OR1-FAKE"])
        engine.add_credentials(db, "zai", ["Z1-FAKE"])
        db["openrouter"]["models"] = ["mimo-v2.5-free"]
        db["zai"]["models"] = ["mimo-v2.5-free"]
        # unknown tiers, same stem -> SUGGESTED, never silent
        self.assertEqual(engine.auto_combine(db, "zai", ["mimo-v2.5-free"]), [])
        pending = engine.pending_suggestions(db)
        self.assertIn("mimo-v2.5", pending)
        engine.combine_models(db, "mimo-v2.5",
                              [(m["provider"], m["model"])
                               for m in pending["mimo-v2.5"]])
        self.assertNotIn("mimo-v2.5", engine.pending_suggestions(db))


class GatewayProbeTest(unittest.TestCase):
    def test_aliases_empty_without_config(self):
        p, _ = tmp_paths()
        self.assertEqual(engine.gateway_aliases(p), [])

    def test_aliases_from_written_config(self):
        p, _ = tmp_paths()
        db = engine.load_state(p)
        engine.add_credentials(db, "gemini", ["GK1-FAKE"])
        db["gemini"]["models"] = ["gemini-3.7-flash"]
        engine.write_config(db, p)
        self.assertEqual(engine.gateway_aliases(p), ["gemini-3.7-flash"])

    def test_probe_gateway_alias_mockable(self):
        import wizard as w
        p, _ = tmp_paths()
        real = w._gateway_request
        w._gateway_request = lambda *a, **k: ("OK", "choices OK")
        try:
            self.assertEqual(engine.probe_gateway_alias("m", p), ("OK", "choices OK"))
        finally:
            w._gateway_request = real

    def test_quarantine_roundtrip(self):
        db = engine.load_state(engine.EnginePaths.temp(tempfile.mkdtemp()))
        ids = engine.add_credentials(db, "gemini", ["GK1-FAKE"])
        engine.set_models(db, "gemini", ["gemini-3.7-flash"])
        self.assertTrue(engine.set_credential_quarantined(db, "gemini", ids[0]))
        deps, _, _, errors = engine.compile_config(db)
        self.assertEqual(errors, [])
        self.assertEqual(deps, [])  # parked: saved but excluded
        self.assertTrue(engine.set_credential_quarantined(db, "gemini", ids[0], False))
        deps, _, _, _ = engine.compile_config(db)
        self.assertEqual(len(deps), 1)
        self.assertFalse(engine.set_credential_quarantined(db, "gemini", "cred-nope"))


class HealthRefreshTest(unittest.TestCase):
    def test_probe_all_records_and_counts(self):
        import contextlib
        import io

        import wizard as w
        p, _ = tmp_paths()
        db = engine.load_state(p)
        engine.add_credentials(db, "gemini", ["K1-FAKE", "K2-FAKE"])
        engine.set_models(db, "gemini", ["gemini-3.7-flash"])
        engine.add_credentials(db, "openrouter", ["OR-FAKE"])
        engine.set_models(db, "openrouter", ["paid-b"])
        seen, progress = [], []
        real = w.probe_model_classified

        def fake(pid, model, key, endpoint=None):
            seen.append((pid, model))
            return ("OK", "fine") if pid == "gemini" else ("AUTH_ERROR", "bad")

        w.probe_model_classified = fake
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                counts = engine.refresh_credential_health(
                    db, sleep_s=0,
                    progress=lambda *a: progress.append(a))
        finally:
            w.probe_model_classified = real
        self.assertEqual(counts, {"checked": 3, "ok": 2, "throttled": 0,
                                  "invalid": 1, "unknown": 0})
        self.assertEqual(len(seen), 3)  # one probe per credential, not per deployment
        self.assertEqual(len(progress), 3)
        for c in db["gemini"]["credentials"]:
            self.assertEqual((c.get("validation") or {}).get("status"), "ok")
        orch = db["openrouter"]["credentials"][0]
        self.assertEqual((orch.get("validation") or {}).get("status"), "invalid")

    def test_stop_and_skips(self):
        import contextlib
        import io
        import threading

        import wizard as w
        p, _ = tmp_paths()
        db = engine.load_state(p)
        engine.add_credentials(db, "gemini", ["K1-FAKE"])
        engine.set_models(db, "gemini", ["m"])
        stop = threading.Event()
        stop.set()
        with contextlib.redirect_stdout(io.StringIO()):
            counts = engine.refresh_credential_health(db, sleep_s=0, stop=stop)
        self.assertEqual(counts["checked"], 0)
        # quarantined credentials are skipped, not probed
        engine.set_credential_quarantined(db, "gemini",
                                          db["gemini"]["credentials"][0]["id"], True)
        real = w.probe_model_classified
        w.probe_model_classified = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not probe"))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                counts = engine.refresh_credential_health(db, sleep_s=0)
        finally:
            w.probe_model_classified = real
        self.assertEqual(counts["checked"], 0)


if __name__ == "__main__":
    unittest.main()
