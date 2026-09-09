"""Alias normalization + capability grouping tests."""
import unittest

from helpers import load_wizard

w = load_wizard()


class StemTest(unittest.TestCase):
    def test_free_suffixes(self):
        self.assertEqual(w._alias_stem("deepseek-v4-flash-0731"), "deepseek-v4-flash")
        self.assertEqual(w._alias_stem("deepseek-v4-flash:free"), "deepseek-v4-flash")
        self.assertEqual(w._alias_stem("deepseek-v4-flash-free"), "deepseek-v4-flash")
        self.assertEqual(w._alias_stem("deepseek/deepseek-v4-flash-free"), "deepseek-v4-flash")
        self.assertEqual(w._alias_stem("free/deepseek-v4-flash-0731"), "deepseek-v4-flash")

    def test_glm(self):
        self.assertEqual(w._alias_stem("glm-5.3-flash"), "glm-5.3-flash")
        self.assertEqual(w._alias_stem("glm-5.3-flash-free"), "glm-5.3-flash")
        self.assertEqual(w._alias_stem("z-ai/glm-5.3-flash-free"), "glm-5.3-flash")

    def test_tiers_not_merged(self):
        flash = w._alias_stem("gemini-3.1-flash-lite")
        pro = w._alias_stem("gemini-3.1-pro-preview")
        self.assertNotEqual(flash, pro)
        self.assertIn("lite", flash)
        self.assertIn("pro", pro)

    def test_version_digits_kept(self):
        # 5.3 must not collapse (dot versions are significant)
        self.assertEqual(w._alias_stem("glm-5.3-flash"), "glm-5.3-flash")
        self.assertNotEqual(w._alias_stem("glm-5.3-flash"), w._alias_stem("glm-53-flash"))


class TierTest(unittest.TestCase):
    def test_flash_vs_pro(self):
        self.assertEqual(w.classify_model_tier("gemini-3.7-flash"), "flash")
        self.assertEqual(w.classify_model_tier("gemini-3.1-pro-preview"), "pro")
        self.assertEqual(w.classify_model_tier("gemini-3.1-flash-lite"), "lite")

    def test_reasoning(self):
        self.assertEqual(w.classify_model_tier("deepseek-reasoner-x"), "reasoning")


class CapabilityTest(unittest.TestCase):
    def test_flash_flash_compatible(self):
        a = w.infer_capabilities("custom_a", "free/deepseek-v4-flash-0731")
        b = w.infer_capabilities("custom_b", "deepseek-v4-flash:free")
        self.assertTrue(w.capability_compatible(a, b))

    def test_flash_pro_incompatible(self):
        a = w.infer_capabilities("gemini", "gemini-3.7-flash")
        b = w.infer_capabilities("gemini", "gemini-3.1-pro-preview")
        self.assertFalse(w.capability_compatible(a, b))

    def test_group_safety_auto_vs_suggested(self):
        mode = w._group_safety("custom_a", "free/deepseek-v4-flash-0731",
                               "custom_b", "deepseek-v4-flash:free")
        self.assertIn(mode, ("AUTO", "SUGGESTED"))
        # flash vs pro same-pool attempt must never be AUTO
        mode2 = w._group_safety("gemini", "gemini-3.7-flash",
                                "gemini", "gemini-3.1-pro-preview")
        self.assertNotEqual(mode2, "AUTO")

    def test_suggest_groups_structure(self):
        db = w.migrate_db({
            "custom_a": {"keys": ["A"], "models": ["free/deepseek-v4-flash-0731"],
                         "endpoints": [], "base_url": "https://a.example/v1"},
            "custom_b": {"keys": ["B"], "models": ["deepseek-v4-flash:free"],
                         "endpoints": [], "base_url": "https://b.example/v1"},
        })
        sugg = w._suggest_alias_groups(db)
        self.assertIn("deepseek-v4-flash", sugg)
        info = sugg["deepseek-v4-flash"]
        self.assertIn(info["mode"], ("AUTO", "SUGGESTED"))
        self.assertEqual(len(info["members"]), 2)

    def test_context_lookup_exact_only(self):
        self.assertIsNone(w.lookup_context_window("custom", "no-such-model-xyz"))
        self.assertIsNone(w.lookup_context_window("gemini", ""))
        self.assertIsNone(w.lookup_context_window("custom", ""))
        caps = w.infer_capabilities("custom", "no-such-model-xyz")
        self.assertEqual(caps["context_window"], "unknown")

    def test_context_lookup_known_model(self):
        try:
            from litellm import model_cost  # noqa: F401 -- availability probe
        except Exception:  # noqa: BLE001 -- any import failure means skip
            self.skipTest("installed litellm model map unavailable")
        self.assertEqual(w.lookup_context_window("openai", "gpt-4o"), 128000)
        caps = w.infer_capabilities("openai", "gpt-4o")
        self.assertEqual(caps["context_window"], 128000)


if __name__ == "__main__":
    unittest.main()
