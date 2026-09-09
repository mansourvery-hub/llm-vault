"""M3/M3b: performance memory + free-first role suggestions."""
import unittest

from helpers import load_wizard

w = load_wizard()


def gemini_db(*, models, keys=("K1",)):
    return w.migrate_db({"gemini": {"keys": list(keys), "models": list(models),
                                    "endpoints": []}})


class PerformanceMemoryTest(unittest.TestCase):
    def test_record_and_mean(self):
        db = gemini_db(models=["m1"])
        for s in (1.0, 2.0, 3.0):
            w.record_probe_latency(db, "gemini", "m1", s)
        self.assertEqual(w.probe_latency(db, "gemini", "m1"), 2.0)

    def test_rolling_window(self):
        db = gemini_db(models=["m1"])
        for s in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
            w.record_probe_latency(db, "gemini", "m1", s)
        # only the last PERF_WINDOW samples count: 3,4,5,6,7 -> mean 5.0
        self.assertEqual(w.PERF_WINDOW, len(db[w.PERF_KEY]["gemini:m1"]["samples"]))
        self.assertEqual(w.probe_latency(db, "gemini", "m1"), 5.0)

    def test_ignores_bad_values(self):
        db = gemini_db(models=["m1"])
        w.record_probe_latency(db, "gemini", "m1", -5)
        w.record_probe_latency(db, "gemini", "m1", None)
        w.record_probe_latency(db, "gemini", "", 1.0)
        self.assertIsNone(w.probe_latency(db, "gemini", "m1"))
        self.assertNotIn(w.PERF_KEY, db)

    def test_unknown_model_returns_none(self):
        db = gemini_db(models=["m1"])
        self.assertIsNone(w.probe_latency(db, "gemini", "nope"))

    def test_never_secret_keys(self):
        db = gemini_db(models=["m1"])
        w.record_probe_latency(db, "gemini", "m1", 1.5)
        import json
        raw = json.dumps(db[w.PERF_KEY])
        for k in ("K1",):
            self.assertNotIn(k, raw)


class FreeFirstRolesTest(unittest.TestCase):
    def test_suggests_both_when_tiers_exist(self):
        db = gemini_db(models=["gemini-3.7-flash", "gemini-3.1-pro-preview"])
        s = w.suggest_free_first_roles(db)
        self.assertEqual(s["google-free-fast"]["pools"], ["gemini-3.7-flash"])
        self.assertEqual(s["google-free-smart"]["pools"], ["gemini-3.1-pro-preview"])

    def test_no_suggestion_without_gemini(self):
        db = w.migrate_db({"openrouter": {"keys": ["K1"], "models": ["m:free"],
                                          "endpoints": []}})
        self.assertEqual(w.suggest_free_first_roles(db), {})

    def test_no_duplicate_when_role_exists(self):
        db = gemini_db(models=["gemini-3.7-flash"])
        db[w.ROLES_KEY]["google-free-fast"] = {"pools": ["gemini-3.7-flash"],
                                                "fallback": [], "requires": {}}
        s = w.suggest_free_first_roles(db)
        self.assertNotIn("google-free-fast", s)

    def test_apply_creates_roles_and_compiles(self):
        db = gemini_db(models=["gemini-3.7-flash", "gemini-3.0-thinking"])
        applied, skipped = w.apply_free_first_roles(db)
        self.assertEqual(applied, ["google-free-fast", "google-free-smart"])
        self.assertEqual(skipped, [])
        roles = db[w.ROLES_KEY]
        self.assertEqual(roles["google-free-fast"]["pools"], ["gemini-3.7-flash"])
        self.assertEqual(roles["google-free-smart"]["pools"], ["gemini-3.0-thinking"])
        # roles compile into model_group_alias + fallbacks
        w.generate_yaml(db)
        import yaml
        with open(w.YAML_FILE) as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["model_group_alias"]["google-free-fast"],
                         "gemini-3.7-flash")
        self.assertEqual(cfg["model_group_alias"]["google-free-smart"],
                         "gemini-3.0-thinking")

    def test_apply_never_touches_user_roles(self):
        db = gemini_db(models=["gemini-3.7-flash"])
        db[w.ROLES_KEY]["myrole"] = {"pools": ["gemini-3.7-flash"],
                                     "fallback": [], "requires": {}}
        w.apply_free_first_roles(db)
        self.assertEqual(db[w.ROLES_KEY]["myrole"]["pools"], ["gemini-3.7-flash"])

    def test_apply_skips_pool_name_collision(self):
        db = gemini_db(models=["gemini-3.7-flash"])
        # a pool named exactly like the role -> suggestion skipped, no crash
        db[w.ALIAS_KEY]["google-free-fast"] = [{"provider": "gemini",
                                                "model": "gemini-3.7-flash"}]
        db[w.ROLES_KEY] = {}
        applied, skipped = w.apply_free_first_roles(db)
        self.assertIn("google-free-fast", skipped)
        self.assertNotIn("google-free-fast", applied)


if __name__ == "__main__":
    unittest.main()
