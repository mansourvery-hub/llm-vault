"""DB migration: legacy schema -> v2, idempotent, lossless."""
import copy
import json
import unittest

from helpers import LEGACY_DB, load_wizard

w = load_wizard()


class MigrationTest(unittest.TestCase):
    def test_legacy_unified_migrated(self):
        db = w.migrate_db(copy.deepcopy(LEGACY_DB))
        self.assertEqual(db.get("_schema_version"), w.SCHEMA_VERSION)
        self.assertNotIn("_unified", db)
        self.assertIn("deepseek-v4-flash", db.get("_aliases", {}))
        members = db["_aliases"]["deepseek-v4-flash"]
        # deduped + malformed dropped
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0]["provider"], "custom_a")

    def test_keys_models_preserved(self):
        db = w.migrate_db(copy.deepcopy(LEGACY_DB))
        self.assertEqual(db["gemini"]["keys"], ["OLDKEY1", "OLDKEY2"])
        self.assertEqual(db["gemini"]["models"],
                         ["gemini-3.7-flash", "gemini-3.1-pro-preview"])
        self.assertEqual(db["openrouter"]["models"], ["minimax/minimax-m3:free"])

    def test_credentials_created_with_stable_ids(self):
        db = w.migrate_db(copy.deepcopy(LEGACY_DB))
        creds = db["gemini"]["credentials"]
        self.assertEqual(len(creds), 2)
        for c in creds:
            self.assertTrue(c["id"].startswith("cred-"))
            self.assertNotIn(c["id"], (c["secret"],))
            self.assertTrue(c["quota_domain"])
        # stable across re-migration and reorder
        db2 = w.migrate_db(copy.deepcopy(db))
        ids1 = sorted(c["id"] for c in db["gemini"]["credentials"])
        ids2 = sorted(c["id"] for c in db2["gemini"]["credentials"])
        self.assertEqual(ids1, ids2)
        self.assertNotIn("OLDKEY1", json.dumps([c["id"] for c in creds]))

    def test_idempotent(self):
        once = w.migrate_db(copy.deepcopy(LEGACY_DB))
        twice = w.migrate_db(copy.deepcopy(once))
        self.assertEqual(once, twice)

    def test_missing_fields_defaulted(self):
        db = w.migrate_db({"gemini": {"keys": ["K"]}})
        self.assertEqual(db["gemini"]["models"], [])
        self.assertEqual(db["gemini"]["endpoints"], [])
        self.assertIn(w.SETTINGS_KEY, db)
        self.assertIn(w.QUOTA_KEY, db)
        self.assertIn(w.ROLES_KEY, db)
        for k, v in w.DEFAULT_SETTINGS.items():
            self.assertEqual(db[w.SETTINGS_KEY][k], v)

    def test_project_id_preserved_and_defaulted(self):
        # additive: existing project_id survives migration, new creds get ""
        db = w.migrate_db({"gemini": {"keys": ["K1", "K2"],
                                      "credentials": [
                                          {"id": "cred-x", "secret": "K1",
                                           "project_id": "proj-a"}]}})
        creds = {c["secret"]: c for c in db["gemini"]["credentials"]}
        self.assertEqual(creds["K1"].get("project_id"), "proj-a")
        self.assertEqual(creds["K2"].get("project_id"), "")

    def test_save_load_roundtrip(self):
        db = w.migrate_db(copy.deepcopy(LEGACY_DB))
        w.save_db(db)
        with open(w.DB_FILE) as f:
            raw = json.load(f)
        self.assertEqual(raw["gemini"]["keys"], ["OLDKEY1", "OLDKEY2"])
        # restrictive perms
        import os
        self.assertEqual(oct(os.stat(w.DB_FILE).st_mode & 0o777), "0o600")

    def test_normalize_aliases_cleans_stale(self):
        db = w.migrate_db(copy.deepcopy(LEGACY_DB))
        db["_aliases"]["stale-pool"] = [
            {"provider": "gemini", "model": "gemini-3.7-flash"},
            {"provider": "gemini", "model": "deleted-model"},
        ]
        cleaned, emptied = w.normalize_aliases(db)
        # 1 stale in stale-pool + 1 stale legacy member (custom_a unknown) emptied
        self.assertEqual(cleaned, 2)
        self.assertIn("deepseek-v4-flash", emptied)
        self.assertEqual(db["_aliases"]["stale-pool"],
                         [{"provider": "gemini", "model": "gemini-3.7-flash"}])
        # empty alias removed
        db["_aliases"]["gone"] = [{"provider": "gemini", "model": "nope"}]
        cleaned, emptied = w.normalize_aliases(db)
        self.assertIn("gone", emptied)
        self.assertNotIn("gone", db["_aliases"])


if __name__ == "__main__":
    unittest.main()
