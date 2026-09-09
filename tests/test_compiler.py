"""Compiler: deployments, invariants, routing/retry policy, roles."""
import unittest

from helpers import load_wizard, read_yaml

w = load_wizard()


def base_db():
    return w.migrate_db({
        "gemini": {"keys": ["GK1"], "models": ["gemini-3.7-flash"], "endpoints": []},
        "openrouter": {"keys": ["OK1"], "models": ["minimax/minimax-m3:free"], "endpoints": []},
        "custom_apinex": {"keys": ["AX1"], "models": ["free/deepseek-v4-flash-0731"],
                          "endpoints": [], "base_url": "https://api.apinex.bond/v1",
                          "label": "apinex"},
        "ollama_local": {"keys": [], "models": ["qwen2.5-coder:7b"],
                         "endpoints": ["http://localhost:11434"]},
        "ollama_cloud": {"keys": ["OC1"], "models": ["gemma4:31b"],
                         "endpoints": ["https://ollama.com"]},
    })


class CompilerTest(unittest.TestCase):
    def test_one_key_one_deployment(self):
        db = w.migrate_db({"gemini": {"keys": ["K"], "models": ["gemini-3.7-flash"],
                                      "endpoints": []}})
        deps, pools, roles, errors = w.compile_config(db)
        self.assertEqual(errors, [])
        self.assertEqual(len(deps), 1)
        self.assertIn("gemini-3.7-flash", pools)

    def test_deterministic_ordering(self):
        d1 = [tuple(sorted(x.items())) for x in w.build_deployments(base_db())]
        d2 = [tuple(sorted(x.items())) for x in w.build_deployments(base_db())]
        self.assertEqual(d1, d2)

    def test_custom_and_ollama_emit(self):
        deps, pools, _, errors = w.compile_config(base_db())
        self.assertEqual(errors, [])
        by_pool = {d["logical_model"] for d in deps}
        self.assertIn("deepseek-v4-flash-0731", by_pool)
        self.assertIn("qwen2.5-coder:7b", by_pool)
        self.assertIn("gemma4:31b", by_pool)

    def test_disabled_excluded_but_kept(self):
        db = base_db()
        db["gemini"]["disabled"] = True
        deps, _, _, errors = w.compile_config(db)
        self.assertEqual(errors, [])
        self.assertNotIn("gemini", {d["provider"] for d in deps})
        self.assertIn("gemini", db)  # stored, not deleted

    def test_no_duplicate_deployments(self):
        db = base_db()
        db["gemini"]["models"] = ["gemini-3.7-flash", "gemini-3.7-flash"]
        deps, _, _, errors = w.compile_config(db)
        keys = [(d["logical_model"], d["provider"], d.get("credential_id"),
                 str(d.get("endpoint")), d["upstream_model"]) for d in deps]
        self.assertEqual(len(keys), len(set(keys)))

    def test_stale_members_excluded(self):
        db = base_db()
        db["_aliases"]["pool-x"] = [
            {"provider": "gemini", "model": "gemini-3.7-flash"},
            {"provider": "gemini", "model": "retired-model"},
        ]
        deps, pools, _, errors = w.compile_config(db)
        self.assertEqual(errors, [])
        self.assertIn("pool-x", pools)
        self.assertEqual({d["upstream_model"] for d in pools["pool-x"]},
                         {"gemini-3.7-flash"})

    def test_empty_alias_removed(self):
        db = base_db()
        db["_aliases"]["pool-gone"] = [{"provider": "gemini", "model": "nope"}]
        deps, pools, _, errors = w.compile_config(db)
        self.assertNotIn("pool-gone", pools)

    def test_role_missing_pool_dropped(self):
        db = base_db()
        db["_roles"]["fast"] = {"pools": ["gemini-3.7-flash", "nope-pool"],
                                "fallback": [], "requires": {}}
        deps, pools, roles, errors = w.compile_config(db)
        self.assertEqual(errors, [])
        self.assertIn("fast", roles)
        self.assertEqual(roles["fast"]["pools"], ["gemini-3.7-flash"])

    def test_routing_policy(self):
        w.generate_yaml(base_db())
        cfg = read_yaml(w.YAML_FILE)
        rs = cfg["router_settings"]
        self.assertEqual(rs["routing_strategy"], "usage-based-routing-v2")
        self.assertTrue(rs["enable_pre_call_checks"])
        self.assertEqual(rs["cooldown_time"], 60)
        self.assertEqual(rs["allowed_fails"], 1)
        self.assertEqual(rs["num_retries"], 1)

    def test_retry_policy_sane(self):
        w.generate_yaml(base_db())
        cfg = read_yaml(w.YAML_FILE)
        rp = cfg["router_settings"]["retry_policy"]
        for k in ("AuthenticationErrorRetries", "BadRequestErrorRetries",
                  "ContentPolicyViolationErrorRetries", "RateLimitErrorRetries"):
            self.assertEqual(rp[k], 0, k)
        self.assertEqual(rp["TimeoutErrorRetries"], 1)
        self.assertEqual(rp["InternalServerErrorRetries"], 1)
        # only keys supported by installed LiteLLM
        from litellm.types.router import RetryPolicy
        self.assertEqual(set(rp), set(RetryPolicy.model_fields))

    def test_allowed_fails_policy_emitted_and_supported(self):
        w.generate_yaml(base_db())
        cfg = read_yaml(w.YAML_FILE)
        afp = cfg["router_settings"]["allowed_fails_policy"]
        # 429 cools immediately; transient errors get one failure first
        self.assertEqual(afp["RateLimitErrorAllowedFails"], 0)
        self.assertEqual(afp["TimeoutErrorAllowedFails"], 1)
        self.assertEqual(afp["InternalServerErrorAllowedFails"], 1)
        # only keys supported by installed LiteLLM's AllowedFailsPolicy
        from litellm.types.router import AllowedFailsPolicy
        self.assertLessEqual(set(afp), set(AllowedFailsPolicy.model_fields))
        # and it must be a valid Router init arg
        from litellm import Router
        self.assertIn("allowed_fails_policy", Router.get_valid_args())

    def test_per_deployment_cooldown_emitted(self):
        w.generate_yaml(base_db())
        cfg = read_yaml(w.YAML_FILE)
        for item in cfg["model_list"]:
            lp = item["litellm_params"]
            # gemini (429-prone) waits the full rate-limit pause
            if lp["model"].startswith("gemini/"):
                self.assertEqual(lp["cooldown_time"], 60.0)
            else:
                self.assertEqual(lp["cooldown_time"], 30.0)

    def test_master_key_env_backed(self):
        w.generate_yaml(base_db())
        cfg = read_yaml(w.YAML_FILE)
        self.assertEqual(cfg["general_settings"]["master_key"],
                         "os.environ/LITELLM_MASTER_KEY")

    def test_role_alias_emitted(self):
        db = base_db()
        db["_roles"]["fast"] = {"pools": ["gemini-3.7-flash", "minimax-m3:free"],
                                "fallback": [], "requires": {}}
        w.generate_yaml(db)
        cfg = read_yaml(w.YAML_FILE)
        self.assertEqual(cfg["model_group_alias"]["fast"], "gemini-3.7-flash")
        self.assertEqual(cfg["fallbacks"], [{"fast": ["minimax-m3:free"]}])

    def test_role_collision_skipped(self):
        db = base_db()
        db["_roles"]["gemini-3.7-flash"] = {"pools": ["gemini-3.7-flash"],
                                            "fallback": [], "requires": {}}
        w.generate_yaml(db)  # must not raise; role skipped
        cfg = read_yaml(w.YAML_FILE)
        self.assertNotIn("model_group_alias", cfg)

    def test_strict_role_excludes_unknown(self):
        db = base_db()
        db["_roles"]["coder"] = {"pools": ["gemini-3.7-flash"],
                                 "fallback": [], "requires": {"tools": True}}
        _, _, roles, _ = w.compile_config(db)
        self.assertNotIn("coder", roles)  # gemini tools=unknown -> excluded

    def test_yaml_mode_600(self):
        import os
        w.generate_yaml(base_db())
        self.assertEqual(oct(os.stat(w.YAML_FILE).st_mode & 0o777), "0o600")

    def test_diff_reporting(self):
        old = w.build_deployments(base_db())
        db = base_db()
        db["gemini"]["models"].append("gemini-3.1-pro-preview")
        new = w.build_deployments(db)
        diff = w.deployment_diff(old, new)
        self.assertEqual(diff["added"], 1)
        self.assertEqual(diff["removed"], 0)


class EndpointResolutionTest(unittest.TestCase):
    def test_precedence(self):
        db = w.migrate_db({
            "zai": {"keys": ["Z"], "models": ["glm-4.7-flash"],
                    "endpoints": ["https://api.z.ai/api/v1"]},
        })
        # stored endpoints[0] wins over everything
        self.assertEqual(w.effective_base_url(db, "zai"),
                         "https://api.z.ai/api/v1")
        self.assertEqual(w.effective_base_url(db, "zai", "https://x.example/v9/"),
                         "https://x.example/v9")
        db["zai"]["endpoints"] = []
        db["zai"]["base_url"] = "https://stored.example/v1"
        self.assertEqual(w.effective_base_url(db, "zai"),
                         "https://stored.example/v1")
        del db["zai"]["base_url"]
        # builtin default, trailing slashes trimmed
        self.assertEqual(w.effective_base_url(db, "zai"),
                         "https://api.z.ai/api/paas/v4")
        # stateless callers still get the builtin
        self.assertEqual(w.effective_base_url(None, "zai"),
                         "https://api.z.ai/api/paas/v4")
        # nothing known -> None (true-custom before URL entry)
        self.assertIsNone(w.effective_base_url({}, "custom"))
        self.assertIsNone(w.effective_base_url(None, "custom"))

    def test_yaml_uses_stored_override(self):
        db = w.migrate_db({
            "zai": {"keys": ["Z"], "models": ["glm-4.7-flash"],
                    "endpoints": ["https://api.z.ai/api/v1"]},
        })
        w.generate_yaml(db)
        cfg = read_yaml(w.YAML_FILE)
        bases = {e["litellm_params"]["api_base"] for e in cfg["model_list"]}
        self.assertEqual(bases, {"https://api.z.ai/api/v1"})

    def test_yaml_falls_back_to_builtin(self):
        db = w.migrate_db({
            "zai": {"keys": ["Z"], "models": ["glm-4.7-flash"], "endpoints": []},
        })
        w.generate_yaml(db)
        cfg = read_yaml(w.YAML_FILE)
        bases = {e["litellm_params"]["api_base"] for e in cfg["model_list"]}
        self.assertEqual(bases, {"https://api.z.ai/api/paas/v4"})


if __name__ == "__main__":
    unittest.main()
