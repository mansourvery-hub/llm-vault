"""Quota domains: capacity is per-domain, never keys x RPM."""
import unittest

from helpers import load_wizard, read_yaml

w = load_wizard()


def db_with(*, keys, domain_of=None, rpm=10, models=None):
    db = w.migrate_db({"gemini": {"keys": list(keys), "models": models or ["gemini-3.7-flash"],
                                  "endpoints": []}})
    creds = db["gemini"]["credentials"]
    if domain_of:
        for c in creds:
            c["quota_domain"] = domain_of(str(c["secret"]))
    for qd in {c["quota_domain"] for c in creds}:
        w.ensure_quota_domain(db, qd, provider="gemini", rpm=rpm,
                              confidence="manual", source="test")
    return db


class QuotaTest(unittest.TestCase):
    def test_shared_domain_counts_once(self):
        db = db_with(keys=["K1", "K2", "K3", "K4"],
                     domain_of=lambda k: "google-project-a", rpm=10)
        deps, pools, _, errors = w.compile_config(db)
        self.assertEqual(errors, [])
        self.assertEqual(len(deps), 4)
        # sum of emitted RPM must not exceed domain quota (no 4x10)
        total = sum(d["rpm"] or 0 for d in deps)
        self.assertLessEqual(total, 10)
        cap = w.estimate_capacity(db, [("gemini", "gemini-3.7-flash")])
        self.assertEqual(cap["rpm_known"], 10)

    def test_separate_domains_sum(self):
        db = db_with(keys=["K1", "K2"],
                     domain_of=lambda k: "proj-a" if k == "K1" else "proj-b",
                     rpm=10)
        deps, _, _, errors = w.compile_config(db)
        self.assertEqual(errors, [])
        total = sum(d["rpm"] or 0 for d in deps)
        self.assertEqual(total, 20)
        cap = w.estimate_capacity(db, [("gemini", "gemini-3.7-flash")] * 2)
        self.assertEqual(cap["rpm_known"], 20)

    def test_unknown_domains_stay_unknown(self):
        db = w.migrate_db({"openrouter": {"keys": ["K1"], "models": ["m1"], "endpoints": []}})
        deps, _, _, errors = w.compile_config(db)
        self.assertEqual(errors, [])
        self.assertIsNone(deps[0]["rpm"])
        w.generate_yaml(db)
        cfg = read_yaml(w.YAML_FILE)
        for item in cfg["model_list"]:
            self.assertNotIn("rpm", item["litellm_params"])

    def test_split_never_overstates(self):
        self.assertEqual(w.split_shared_quota(10, 4), 2)
        self.assertEqual(w.split_shared_quota(10, 1), 10)
        self.assertIsNone(w.split_shared_quota(None, 4))
        self.assertGreaterEqual(w.split_shared_quota(3, 5), 1)

    def test_split_is_per_domain_and_model(self):
        db = db_with(keys=["K1", "K2", "K3", "K4"],
                     domain_of=lambda k: "google-project-a", rpm=10,
                     models=["gemini-3.7-flash", "gemini-3.5-flash"])
        deps, _, _, errors = w.compile_config(db)
        self.assertEqual(errors, [])
        per_model = {}
        for d in deps:
            per_model.setdefault(d["upstream_model"], []).append(d["rpm"])
        for model, rpms in per_model.items():
            self.assertLessEqual(sum(rpms), 10, model)

    def test_limit_precedence(self):
        db = db_with(keys=["K1"], domain_of=lambda k: "d1", rpm=10)
        cred = db["gemini"]["credentials"][0]
        rpm, _ = w.resolve_deployment_limits(db, "gemini", cred, "gemini-3.7-flash")
        self.assertEqual(rpm, 10)
        cred["limits"] = {"rpm": 5}
        rpm, _ = w.resolve_deployment_limits(db, "gemini", cred, "gemini-3.7-flash")
        self.assertEqual(rpm, 5)

    def test_quota_scope_metadata(self):
        self.assertEqual(w.PROVIDER_META["gemini"]["quota_scope"], "project")
        self.assertIn(w.PROVIDER_META["openrouter"]["quota_scope"],
                      ("account", "credential", "unknown"))

    def test_project_id_groups_credentials(self):
        # credentials sharing a project collapse into ONE domain
        db = w.migrate_db({"gemini": {"keys": ["K1", "K2"], "models": ["m"],
                                      "endpoints": []}})
        for c in db["gemini"]["credentials"]:
            c["project_id"] = "proj-a"
        self.assertEqual(w.default_quota_domain_id("gemini",
                         db["gemini"]["credentials"][0]), "project:gemini:proj-a")
        domains = w.quota_domains_list(db)
        self.assertEqual(len(domains), 1)
        self.assertEqual(domains[0]["domain_id"], "project:gemini:proj-a")
        self.assertEqual(domains[0]["key_count"], 2)
        self.assertEqual(domains[0]["project_id"], "proj-a")

    def test_no_project_id_stays_per_credential(self):
        db = w.migrate_db({"gemini": {"keys": ["K1", "K2"], "models": ["m"],
                                      "endpoints": []}})
        domains = w.quota_domains_list(db)
        self.assertEqual(len(domains), 2)
        self.assertTrue(all(d["domain_id"].startswith("credential:")
                           for d in domains))

    def test_project_domain_inherits_rpm_once(self):
        # keys in one project domain share one RPM: never keys x RPM
        db = w.migrate_db({"gemini": {"keys": ["K1", "K2"], "models": ["m"],
                                      "endpoints": []}})
        for c in db["gemini"]["credentials"]:
            c["project_id"] = "proj-a"
        w.ensure_quota_domain(db, "project:gemini:proj-a", provider="gemini",
                              rpm=10, confidence="manual", source="test")
        deps, _, _, errors = w.compile_config(db)
        self.assertEqual(errors, [])
        self.assertLessEqual(sum(d["rpm"] or 0 for d in deps), 10)

    def test_normalize_credentials_defaults_project_id(self):
        pdata = {"keys": ["K1"], "credentials": [{"id": "cred-x", "secret": "K1"}]}
        w.normalize_credentials(pdata)
        self.assertEqual(pdata["credentials"][0].get("project_id"), "")


if __name__ == "__main__":
    unittest.main()
