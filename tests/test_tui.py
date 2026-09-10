"""TUI tests (Milestones 2-3). Temp dirs + fake keys only.

Network guard: wizard HTTP adapters and urlopen raise if touched — any
unmocked network access fails loudly. Milestone 3 flow tests mock at the
``wizard.validate_keys`` / ``wizard.fetch_catalog`` / ``wizard.test_models``
level so the engine delegation path is genuinely exercised.
"""
import asyncio
import os
import tempfile
import time
import unittest
from unittest import mock

import engine
import tui
import wizard
from tui import (
    ConfigureScreen,
    DoneScreen,
    HomeScreen,
    ImportScreen,
    JCodeScreen,
    ModelDetailScreen,
    ModelScreen,
    ProviderScreen,
    QuotaDashboardScreen,
    TestScreen,
    WizardApp,
    done_lines,
    home_lines,
    split_keys,
)


def temp_paths():
    tmp = tempfile.mkdtemp(prefix="tuitest-")
    return engine.EnginePaths.temp(tmp)


def seed_db(paths):
    db = engine.load_state(paths)
    engine.add_credentials(db, "gemini", ["GK1-FAKE"])
    db["gemini"]["models"] = ["gemini-3.7-flash"]
    engine.save_state(db, paths)
    return db


class HelpersTest(unittest.TestCase):
    def test_split_keys(self):
        self.assertEqual(split_keys("a b,c\nd a"), ["a", "b", "c", "d"])
        self.assertEqual(split_keys("  "), [])
        self.assertEqual(split_keys("'quoted' \"q2\""), ["quoted", "q2"])

    def test_home_lines_quiet(self):
        ov = {"gateway": "running", "pools": ["gemini-3.7-flash"],
              "members": {"gemini-3.7-flash": ["gemini"]}, "attention": []}
        text = home_lines(ov)
        self.assertIn("Running", text)
        self.assertIn("gemini-3.7-flash", text)
        self.assertNotIn("Needs attention", text)

    def test_home_lines_attention(self):
        ov = {"gateway": "unknown", "pools": [], "members": {},
              "attention": ["No config.yaml yet"]}
        text = home_lines(ov)
        self.assertIn("Unknown", text)
        self.assertIn("No config.yaml yet", text)

    def test_test_and_done_lines(self):
        paths = temp_paths()
        db = seed_db(paths)
        self.assertIn("gemini-3.7-flash", tui.test_lines(db))
        self.assertIn("1 connection(s)", done_lines(db))
        self.assertIn("Run the test", tui.test_lines(db))
        self.assertIn("Apply writes safely", done_lines(db))


class DashboardHelpersTest(unittest.TestCase):
    def test_format_ctx(self):
        self.assertEqual(tui.format_ctx(1048576), "1M")
        self.assertEqual(tui.format_ctx(128000), "128K")
        self.assertEqual(tui.format_ctx(45000), "45K")
        self.assertEqual(tui.format_ctx(1500000), "1.5M")
        self.assertEqual(tui.format_ctx(None), "—")
        self.assertEqual(tui.format_ctx("unknown"), "—")
        self.assertEqual(tui.format_ctx(-5), "—")
        self.assertEqual(tui.format_ctx(True), "—")

    def test_format_limit(self):
        self.assertEqual(tui.format_limit(10, None, 1), "10 RPM")
        self.assertEqual(tui.format_limit(10, None, 3), "10÷3 RPM")
        self.assertIsNone(tui.format_limit(None, None, 1))
        self.assertIsNone(tui.format_limit("x", None, 1))
        self.assertEqual(tui.format_limit(20, 50000, 1), "20 RPM · 50K TPM")

    def test_fit_cell(self):
        self.assertEqual(tui.fit_cell("abc", 5), "abc  ")
        self.assertEqual(tui.fit_cell("abcdef", 5), "abcd…")
        self.assertEqual(len(tui.fit_cell("x", 8)), 8)

    def test_next_sort(self):
        self.assertEqual(tui.next_sort("pool", False, "pool"), ("pool", True))
        self.assertEqual(tui.next_sort("pool", True, "pool"), ("pool", False))
        self.assertEqual(tui.next_sort("pool", True, "provider"), ("provider", False))

    def _rows(self):
        def r(pool, health, tier="flash"):
            return {"pool": pool, "provider": "gemini", "upstream": "m",
                    "tier": tier, "quota_domain": "d", "health": health}
        return [r("b", "unknown"), r("a", "healthy"), r("c", "invalid", "lite")]

    def test_summary_counts(self):
        counts = tui.summary_counts(self._rows())
        self.assertEqual(counts, {"pools": 3, "deployments": 3,
                                  "working": 1, "untested": 1})

    def test_badge_shows_working(self):
        badge = tui.gateway_badge({"gateway": "running"},
                                  {"pools": 3, "deployments": 3,
                                   "working": 1, "untested": 1})
        self.assertIn("Running", badge)
        self.assertIn("1/3", badge)
        self.assertIn("untested", badge)

    def test_tier_filter(self):
        rows = self._rows()
        self.assertEqual(len(tui.filter_table_rows(rows, "", tier="flash")), 2)
        self.assertEqual(len(tui.filter_table_rows(rows, "c", tier="flash")), 0)
        self.assertEqual(len(tui.filter_table_rows(rows, "", tier=None)), 3)

    def test_table_rows_shape(self):
        paths = temp_paths()
        db = seed_db(paths)
        rows = tui.deployment_table_rows(db)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row["quota"].startswith("10 RPM · cred-"),
                        row["quota"])
        self.assertTrue(row["credential_id"].startswith("cred-"))
        # ctx is real data when the litellm map is installed, else honest "—"
        self.assertIn(row["ctx"], ("1M", "—"))
        self.assertIn("untested", row["health_txt"])
        self.assertTrue(tui.row_key_for(row))

    def test_row_keys_stable(self):
        paths = temp_paths()
        db = seed_db(paths)
        rows = tui.deployment_table_rows(db)
        again = tui.deployment_table_rows(db)
        self.assertEqual([tui.row_key_for(r) for r in rows],
                         [tui.row_key_for(r) for r in again])
        # key covers pool + provider + credential + endpoint (unique per route)
        self.assertIn("gemini-3.7-flash", tui.row_key_for(rows[0]))


class TuiPilotTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Fail loudly on any network attempt from any screen.
        self._get, wizard._get = wizard._get, _no_network
        self._post, wizard._post = wizard._post, _no_network
        import urllib.request
        self._urlopen = urllib.request.urlopen
        urllib.request.urlopen = _no_network_urlopen

    def tearDown(self):
        wizard._get = self._get
        wizard._post = self._post
        import urllib.request
        urllib.request.urlopen = self._urlopen

    def make_app(self, seed=True):
        paths = temp_paths()
        if seed:
            seed_db(paths)
        return WizardApp(paths=paths, status="unknown", status_auto_refresh=False,
                          auto_probe=False)

    async def test_launch_shows_home(self):
        app = self.make_app()
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.pause()
            self.assertIsInstance(app.screen, HomeScreen)
            assert isinstance(app.screen, HomeScreen)
            self.assertIn("gemini-3.7-flash", app.screen.last_content)

    async def test_row_enter_opens_modal(self):
        app = self.make_app()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            home = app.screen
            assert isinstance(home, HomeScreen)
            self.assertGreater(home.query_one("#models-table", tui.DataTable).row_count, 0)
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, ModelDetailScreen)
            modal = app.screen
            assert isinstance(modal, ModelDetailScreen)
            self.assertEqual(modal.row["pool"], "gemini-3.7-flash")
            modal.query_one("#detail-info", tui.Static)
            modal.query_one("#probe-cred", tui.Button)
            await pilot.press("escape")
            await pilot.pause()
            self.assertIsInstance(app.screen, HomeScreen)

    async def test_row_click_opens_modal(self):
        paths = temp_paths()
        db = engine.load_state(paths)
        engine.add_credentials(db, "gemini", ["GK1-FAKE"])
        db["gemini"]["models"] = ["gemini-3.5-flash-lite", "gemini-3.7-flash"]
        engine.save_state(db, paths)
        app = WizardApp(paths=paths, status="unknown", status_auto_refresh=False,
                          auto_probe=False)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            home = app.screen
            assert isinstance(home, HomeScreen)
            table = home.query_one("#models-table", tui.DataTable)
            self.assertEqual(table.row_count, 2)
            self.assertEqual(tuple(table.cursor_coordinate), (0, 0))
            # header is y=0, data rows start at y=1: first click moves cursor
            await pilot.click("#models-table", offset=(6, 2))
            await pilot.pause()
            self.assertIsInstance(app.screen, HomeScreen)
            self.assertEqual(tuple(table.cursor_coordinate), (1, 0))
            # second click on the highlighted row acts (same as Enter)
            await pilot.click("#models-table", offset=(6, 2))
            await pilot.pause()
            self.assertIsInstance(app.screen, ModelDetailScreen)
            modal = app.screen
            assert isinstance(modal, ModelDetailScreen)
            self.assertEqual(modal.row["pool"], "gemini-3.7-flash")
            await pilot.press("escape")
            await pilot.pause()
            self.assertIsInstance(app.screen, HomeScreen)

    async def test_sort_and_tier_keys(self):
        app = self.make_app()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            home = app.screen
            assert isinstance(home, HomeScreen)
            await pilot.press("s")
            await pilot.pause()
            self.assertEqual(home.sort_key, "provider")
            self.assertFalse(home.sort_reverse)
            home._sort_by_column("tier")  # every header sorts, tier included
            self.assertEqual(home.sort_key, "tier")
            self.assertIsNone(home.tier_filter)  # header never filters (T does)
            home._sort_by_column("provider")  # new column -> ascending
            self.assertEqual(home.sort_key, "provider")
            self.assertFalse(home.sort_reverse)
            home._sort_by_column("provider")  # same column -> reverse
            self.assertTrue(home.sort_reverse)

    async def test_auto_probe_fills_status(self):
        paths = temp_paths()
        seed_db(paths)
        app = WizardApp(paths=paths, status="unknown",
                        status_auto_refresh=False)  # auto_probe on (default)
        with mock.patch.object(engine, "refresh_credential_health",
                               return_value={"checked": 1, "ok": 1,
                                             "throttled": 0, "invalid": 0,
                                             "unknown": 0}) as probed:
            async with app.run_test(size=(120, 50)) as pilot:
                await pilot.pause()
                home = app.screen
                assert isinstance(home, HomeScreen)
                for _ in range(200):
                    if probed.called and not home.probing:
                        break
                    await asyncio.sleep(0.05)
                self.assertTrue(probed.called)
                self.assertIn("last probe", home.probe_note)

    async def test_column_widths_stable(self):
        app = self.make_app()
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            home = app.screen
            assert isinstance(home, HomeScreen)
            table = home.query_one("#models-table", tui.DataTable)
            widths0 = [c.width for c in table.columns.values()]
            self.assertTrue(all(w > 0 for w in widths0))
            await pilot.press("slash")
            await pilot.pause()
            for ch in "zzz":
                await pilot.press(ch)
                await pilot.pause()
            self.assertEqual(table.row_count, 0)
            self.assertEqual([c.width for c in table.columns.values()], widths0)
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("x")
            await pilot.pause()
            self.assertEqual(table.row_count, 1)
            self.assertEqual([c.width for c in table.columns.values()], widths0)

    async def test_all_views_reachable(self):
        app = self.make_app()
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            self.assertIsInstance(app.screen, ConfigureScreen)
            # choose the first provider (keyboard: list is focused)
            view = app.screen.query_one("#provider-list", tui.ListView)
            self.assertGreater(len(view.children), 0)
            view.index = 0
            await pilot.press("enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, ProviderScreen)
            await pilot.press("escape")
            await pilot.pause()
            self.assertIsInstance(app.screen, ConfigureScreen)
            await pilot.press("escape")
            await pilot.pause()
            self.assertIsInstance(app.screen, HomeScreen)
            await pilot.press("t")
            await pilot.pause()
            self.assertIsInstance(app.screen, TestScreen)
            await pilot.press("escape")
            await pilot.pause()
            self.assertIsInstance(app.screen, HomeScreen)
            await pilot.press("v")
            await pilot.pause()
            self.assertIsInstance(app.screen, DoneScreen)
            await pilot.press("escape")
            await pilot.pause()
            self.assertIsInstance(app.screen, HomeScreen)
            await pilot.press("u")
            await pilot.pause()
            self.assertIsInstance(app.screen, QuotaDashboardScreen)
            await pilot.press("escape")
            await pilot.pause()
            self.assertIsInstance(app.screen, HomeScreen)

    async def test_quota_dashboard_groups_projects(self):
        paths = temp_paths()
        db = engine.load_state(paths)
        engine.add_credentials(db, "gemini", ["GK1-FAKE", "GK2-FAKE"])
        db["gemini"]["models"] = ["gemini-3.7-flash"]
        for c in db["gemini"]["credentials"]:
            c["project_id"] = "proj-a"
        engine.add_credentials(db, "openrouter", ["OR1-FAKE"])
        db["openrouter"]["models"] = ["m1:free"]
        engine.save_state(db, paths)
        app = WizardApp(paths=paths, status="unknown", status_auto_refresh=False,
                        auto_probe=False)
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.pause()
            await pilot.press("u")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, QuotaDashboardScreen)
            table = screen.query_one("#quota-table", tui.DataTable)
            self.assertEqual(table.row_count, 2)  # one google project + one per-cred
            domains = {d["domain_id"] for d in screen.domains}
            self.assertIn("project:gemini:proj-a", domains)
            status = screen.query_one("#quota-status", tui.Static)
            self.assertIn("2 independent domain(s), 3 key(s)",
                          str(status.render()))
            await pilot.press("escape")
            await pilot.pause()
            self.assertIsInstance(app.screen, HomeScreen)

    async def test_jcode_screen_reachable_and_syncs(self):
        import tempfile
        tmp = tempfile.mkdtemp(prefix="jcodetui-")
        paths = engine.EnginePaths.temp(
            tmp, opencode=os.path.join(tmp, "opencode.json"),
            jcode=os.path.join(tmp, "jcode_config.toml"))
        db = engine.load_state(paths)
        engine.add_credentials(db, "gemini", ["GK1-FAKE"])
        db["gemini"]["models"] = ["gemini-3.7-flash"]
        engine.save_state(db, paths)
        # gateway yaml so sync has aliases (patched: writes to temp paths)
        with engine._patched_wizard(paths):
            engine._wiz.generate_yaml(db)
        app = WizardApp(paths=paths, status="unknown", status_auto_refresh=False,
                        auto_probe=False)
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.pause()
            # sync-targets line rendered on Home
            targets = str(app.screen.query_one("#sync-targets", tui.Static).render())
            self.assertIn("JCode", targets)
            await pilot.press("j")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, JCodeScreen)
            await pilot.click("#sync-jcode")
            await pilot.pause()
            self.assertTrue(os.path.exists(paths.jcode_config))

            def _read_config() -> str:
                with open(paths.jcode_config) as f:
                    return f.read()

            text = await asyncio.to_thread(_read_config)
            self.assertIn("[providers.llm-proxy-wizard]", text)
            self.assertIn('id = "gemini-3.7-flash"', text)
            await pilot.press("escape")
            await pilot.pause()
            self.assertIsInstance(app.screen, HomeScreen)

    async def test_import_screen_reachable(self):
        app = self.make_app()
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.pause()
            await pilot.press("i")
            await pilot.pause()
            self.assertIsInstance(app.screen, ImportScreen)
            await pilot.press("escape")
            await pilot.pause()
            self.assertIsInstance(app.screen, HomeScreen)

    async def test_check_and_save_keys(self):
        paths = temp_paths()
        app = WizardApp(paths=paths, status="unknown", status_auto_refresh=False,
                          auto_probe=False)
        validated = ([("OR1-FAKE", True, "fine"), ("OR2-FAKE", True, "fine")], None)
        with mock.patch.object(wizard, "validate_keys", return_value=validated):
            async with app.run_test(size=(100, 50)) as pilot:
                await pilot.pause()
                app.push_screen(ProviderScreen("openrouter"))
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ProviderScreen)
                screen.query_one("#keys-input", tui.TextArea).load_text(
                    "OR1-FAKE, OR2-FAKE")
                await pilot.click("#check")
                await _wait_until(lambda: screen.phase == "results")
                self.assertIn("All 2 connection(s) work", screen.last_result)
                await pilot.click("#save-continue")
                await _wait_until(lambda: isinstance(app.screen, ModelScreen))
                # persisted + readable by the engine/CLI path
                back = engine.load_state(paths)
                self.assertEqual(len(back["openrouter"]["keys"]), 2)

    async def test_background_refresh_applies(self):
        paths = temp_paths()
        calls = []
        real = engine.gateway_status
        engine.gateway_status = lambda *a, **k: calls.append(1) or "running"
        try:
            app = WizardApp(paths=paths, auto_probe=False)  # status unknown, bg refresh on
            async with app.run_test(size=(100, 50)) as pilot:
                await pilot.pause()
                for _ in range(200):
                    if app.status == "running":
                        break
                    await asyncio.sleep(0.05)
                self.assertEqual(app.status, "running")
                self.assertTrue(calls)
        finally:
            engine.gateway_status = real


def _no_network(*a, **k):
    raise AssertionError("network call attempted from TUI shell")


def _no_network_urlopen(*a, **k):
    raise AssertionError("network call attempted from TUI shell")


async def _wait_until(pred, timeout=15):
    start = time.monotonic()
    while not pred():
        if time.monotonic() - start > timeout:
            raise AssertionError("timed out waiting for UI state")
        await asyncio.sleep(0.05)


FREE_CATALOG = [("free-a:free", "Free A"), ("paid-b", "Paid B")]


async def _check_keys(pilot, app, pid, text, validate):
    """Drive ProviderScreen check; returns the screen in results phase."""
    app.push_screen(ProviderScreen(pid))
    await pilot.pause()
    screen = app.screen
    assert isinstance(screen, ProviderScreen)
    screen.query_one("#keys-input", tui.TextArea).load_text(text)
    with mock.patch.object(wizard, "validate_keys", return_value=validate):
        await pilot.click("#check")
        await _wait_until(lambda: screen.phase == "results")
    return screen


class ConfigureFlowTest(unittest.IsolatedAsyncioTestCase):
    """Milestone 3: provider -> check -> grouping -> models -> probe -> apply."""

    def setUp(self):
        self._get, wizard._get = wizard._get, _no_network
        self._post, wizard._post = wizard._post, _no_network
        import urllib.request
        self._urlopen = urllib.request.urlopen
        urllib.request.urlopen = _no_network_urlopen

    def tearDown(self):
        wizard._get = self._get
        wizard._post = self._post
        import urllib.request
        urllib.request.urlopen = self._urlopen

    def make_app(self):
        paths = temp_paths()
        return WizardApp(paths=paths, status="unknown",
                         status_auto_refresh=False, auto_probe=False), paths

    async def test_single_key_skips_grouping(self):
        app, _ = self.make_app()
        validated = ([("K1-FAKE", True, "fine")], None)
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.pause()
            screen = await _check_keys(pilot, app, "gemini", "K1-FAKE", validated)
            await pilot.click("#save-continue")
            await _wait_until(lambda: isinstance(app.screen, ModelScreen))
            self.assertEqual(screen.results, [("K1-FAKE", True, "fine")])

    async def test_two_keys_grouping_question(self):
        app, paths = self.make_app()
        validated = ([("K1-FAKE", True, "fine"), ("K2-FAKE", True, "fine")], None)
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.pause()
            screen = await _check_keys(pilot, app, "gemini",
                                       "K1-FAKE K2-FAKE", validated)
            await pilot.click("#save-continue")
            await _wait_until(lambda: screen.phase == "grouping")
            await pilot.click("#share")
            await _wait_until(lambda: isinstance(app.screen, ModelScreen))
            db = engine.load_state(paths)
            creds = list(db["gemini"]["credentials"])
            self.assertEqual({c["quota_domain"] for c in creds},
                             {"project:gemini-shared"})
            self.assertTrue(db["gemini"]["quota_reviewed"])

    async def test_partial_failure_keeps_valid_only(self):
        app, paths = self.make_app()
        validated = ([("K1-FAKE", True, "fine"),
                      ("K2-FAKE", False, "bad key")], None)
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.pause()
            screen = await _check_keys(pilot, app, "openrouter",
                                       "K1-FAKE K2-FAKE", validated)
            self.assertIn("1 of 2 work", screen.last_result)
            await pilot.click("#save-continue")
            await _wait_until(lambda: isinstance(app.screen, ModelScreen))
            model_screen = app.screen
            assert isinstance(model_screen, ModelScreen)
            self.assertEqual(model_screen.secrets, ["K1-FAKE"])
            back = engine.load_state(paths)
            self.assertEqual(back["openrouter"]["keys"], ["K1-FAKE"])

    async def test_select_probe_save_review(self):
        app, paths = self.make_app()
        validated = ([("K1-FAKE", True, "fine")], None)
        probed = [("paid-b", "OK", "fine")]
        with mock.patch.object(wizard, "fetch_catalog", return_value=FREE_CATALOG), \
                mock.patch.object(wizard, "test_models", return_value=probed):
            async with app.run_test(size=(100, 50)) as pilot:
                await pilot.pause()
                await _check_keys(pilot, app, "openrouter", "K1-FAKE", validated)
                await pilot.click("#save-continue")
                await _wait_until(lambda: isinstance(app.screen, ModelScreen))
                screen = app.screen
                assert isinstance(screen, ModelScreen)
                await _wait_until(lambda: screen.catalog_state == "ready")
                # free-first: only the free model shown until toggled
                mlist = screen.query_one("#model-list", tui.SelectionList)
                self.assertEqual(mlist.option_count, 1)
                await pilot.click("#show-toggle")
                await pilot.pause()
                self.assertEqual(mlist.option_count, 2)
                # filter narrows
                screen.query_one("#filter", tui.Input).value = "paid"
                await pilot.pause()
                self.assertEqual(mlist.option_count, 1)
                screen.query_one("#filter", tui.Input).value = ""
                await pilot.pause()
                mlist.select("paid-b")
                await pilot.click("#probe")
                await _wait_until(lambda: screen.catalog_state == "probed")
                self.assertIn("[OK] paid-b", screen.last_status)
                await pilot.click("#keep-passing")
                await _wait_until(lambda: isinstance(app.screen, DoneScreen))
                back = engine.load_state(paths)
                self.assertEqual(back["openrouter"]["models"], ["paid-b"])
                # terminal flow: Back goes Home, not back through Configure
                await pilot.press("escape")
                await pilot.pause()
                self.assertIsInstance(app.screen, HomeScreen)

    async def test_probe_blocks_failures(self):
        app, paths = self.make_app()
        validated = ([("K1-FAKE", True, "fine")], None)
        probed = [("free-a:free", "OK", "fine"),
                  ("paid-b", "AUTH_ERROR", "bad key")]
        with mock.patch.object(wizard, "fetch_catalog", return_value=FREE_CATALOG), \
                mock.patch.object(wizard, "test_models", return_value=probed):
            async with app.run_test(size=(100, 50)) as pilot:
                await pilot.pause()
                await _check_keys(pilot, app, "openrouter", "K1-FAKE", validated)
                await pilot.click("#save-continue")
                await _wait_until(lambda: isinstance(app.screen, ModelScreen))
                screen = app.screen
                assert isinstance(screen, ModelScreen)
                await _wait_until(lambda: screen.catalog_state == "ready")
                await pilot.click("#show-toggle")
                await pilot.pause()
                mlist = screen.query_one("#model-list", tui.SelectionList)
                mlist.select("free-a:free")
                mlist.select("paid-b")
                await pilot.click("#probe")
                await _wait_until(lambda: screen.catalog_state == "probed")
                self.assertIn("[FAIL] paid-b", screen.last_status)
                await pilot.click("#keep-passing")
                await _wait_until(lambda: isinstance(app.screen, DoneScreen))
                back = engine.load_state(paths)
                self.assertEqual(back["openrouter"]["models"], ["free-a:free"])

    async def test_manual_models_fallback(self):
        app, paths = self.make_app()
        validated = ([("K1-FAKE", True, "fine")], None)
        probed = [("custom-m1", "OK", "fine")]
        with mock.patch.object(wizard, "fetch_catalog", return_value=None), \
                mock.patch.object(wizard, "test_models", return_value=probed):
            async with app.run_test(size=(100, 50)) as pilot:
                await pilot.pause()
                await _check_keys(pilot, app, "openrouter", "K1-FAKE", validated)
                await pilot.click("#save-continue")
                await _wait_until(lambda: isinstance(app.screen, ModelScreen))
                screen = app.screen
                assert isinstance(screen, ModelScreen)
                await _wait_until(lambda: screen.catalog_state == "failed")
                screen.query_one("#manual-input", tui.TextArea).load_text(
                    "custom-m1")
                await pilot.click("#manual-use")
                await _wait_until(lambda: screen.catalog_state == "probed")
                await pilot.click("#keep-passing")
                await _wait_until(lambda: isinstance(app.screen, DoneScreen))
                back = engine.load_state(paths)
                self.assertEqual(back["openrouter"]["models"], ["custom-m1"])

    async def test_retired_models_noticed(self):
        app, _ = self.make_app()
        db = engine.load_state(app.paths)
        engine.add_credentials(db, "openrouter", ["K1-FAKE"])
        engine.set_models(db, "openrouter", ["old-retired"])
        engine.save_state(db, app.paths)
        app.db = engine.load_state(app.paths)
        validated = ([("K1-FAKE", True, "fine")], None)
        with mock.patch.object(wizard, "fetch_catalog", return_value=FREE_CATALOG):
            async with app.run_test(size=(100, 50)) as pilot:
                await pilot.pause()
                await _check_keys(pilot, app, "openrouter", "K1-FAKE", validated)
                await pilot.click("#save-continue")
                await _wait_until(lambda: isinstance(app.screen, ModelScreen))
                screen = app.screen
                assert isinstance(screen, ModelScreen)
                await _wait_until(lambda: screen.catalog_state == "ready")
                self.assertIn("old-retired", screen.last_status)

    async def test_endpoint_prefilled_and_editable(self):
        app, paths = self.make_app()
        db = engine.load_state(paths)
        engine.add_credentials(db, "zai", ["Z1-FAKE"])
        db["zai"]["endpoints"] = ["https://api.z.ai/api/v1"]
        engine.save_state(db, paths)
        app.db = engine.load_state(paths)
        validated = ([("Z1-FAKE", True, "fine")], None)
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.pause()
            app.push_screen(ProviderScreen("zai"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ProviderScreen)
            ep = screen.query_one("#endpoint-input", tui.Input)
            self.assertTrue(ep.display)
            self.assertEqual(ep.value, "https://api.z.ai/api/v1")
            screen.query_one("#keys-input", tui.TextArea).load_text("Z1-FAKE")
            with mock.patch.object(wizard, "validate_keys",
                                   return_value=validated) as vk:
                await pilot.click("#check")
                await _wait_until(lambda: screen.phase == "results")
            # prefill is used for the check but not duplicated into storage
            self.assertEqual(vk.call_args[0][2], ["https://api.z.ai/api/v1"])
            await pilot.click("#save-continue")
            await _wait_until(lambda: isinstance(app.screen, ModelScreen))
            back = engine.load_state(paths)
            self.assertEqual(back["zai"]["endpoints"], ["https://api.z.ai/api/v1"])

    async def test_endpoint_edit_stored(self):
        app, paths = self.make_app()
        validated = ([("K1-FAKE", True, "fine")], None)
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.pause()
            app.push_screen(ProviderScreen("custom"))
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ProviderScreen)
            ep = screen.query_one("#endpoint-input", tui.Input)
            self.assertTrue(ep.display)
            self.assertEqual(ep.value, "")  # nothing known: must type
            screen.query_one("#keys-input", tui.TextArea).load_text("K1-FAKE")
            with mock.patch.object(wizard, "validate_keys", return_value=validated):
                await pilot.click("#check")
                await _wait_until(lambda: screen.phase == "keys")
            self.assertIn("base URL", screen.last_result)
            ep.value = "https://edited.example/v1"
            with mock.patch.object(wizard, "validate_keys",
                                   return_value=validated) as vk:
                # pilot quirk: move the synthetic mouse off the button first
                # or the second click is not delivered
                await pilot.hover("#keys-input")
                await pilot.pause()
                await pilot.click("#check")
                await _wait_until(lambda: screen.phase == "results")
            self.assertEqual(vk.call_args[0][2], ["https://edited.example/v1"])
            await pilot.click("#save-continue")
            await _wait_until(lambda: isinstance(app.screen, ModelScreen))
            back = engine.load_state(paths)
            self.assertIn("https://edited.example/v1",
                          back["custom"]["endpoints"])

    async def test_apply_success_then_sync_offer(self):
        app, paths = self.make_app()
        db = engine.load_state(paths)
        engine.add_credentials(db, "gemini", ["GK1-FAKE"])
        engine.set_models(db, "gemini", ["gemini-3.7-flash"])
        engine.save_state(db, paths)
        app.db = engine.load_state(paths)
        with mock.patch.object(engine, "restart_gateway", return_value=True):
            async with app.run_test(size=(100, 50)) as pilot:
                await pilot.pause()
                app.push_screen(DoneScreen())
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, DoneScreen)
                await pilot.click("#apply")
                await _wait_until(lambda: screen.applied_ok)
                self.assertIn("Gateway working", screen.last_status)
                self.assertIn("out of sync", screen.last_status)
                # OpenCode file absent: sync failure reported separately
                await pilot.click("#sync")
                await pilot.pause()
                self.assertIn("gateway itself is working", screen.last_status)

    async def test_apply_restart_failure(self):
        app, paths = self.make_app()
        db = engine.load_state(paths)
        engine.add_credentials(db, "gemini", ["GK1-FAKE"])
        engine.set_models(db, "gemini", ["gemini-3.7-flash"])
        engine.save_state(db, paths)
        app.db = engine.load_state(paths)
        with mock.patch.object(engine, "restart_gateway", return_value=False):
            async with app.run_test(size=(100, 50)) as pilot:
                await pilot.pause()
                app.push_screen(DoneScreen())
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, DoneScreen)
                await pilot.click("#apply")
                await _wait_until(lambda: "not ready" in screen.last_status)
                self.assertFalse(screen.applied_ok)

    async def test_free_first_roles_button(self):
        app, paths = self.make_app()
        db = engine.load_state(paths)
        engine.add_credentials(db, "gemini", ["GK1-FAKE"])
        engine.set_models(db, "gemini", ["gemini-3.7-flash", "gemini-3.0-thinking"])
        engine.save_state(db, paths)
        app.db = engine.load_state(paths)
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.pause()
            app.push_screen(DoneScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, DoneScreen)
            # both suggestions visible + button offered
            self.assertIn("google-free-fast", screen.last_content)
            self.assertIn("google-free-smart", screen.last_content)
            btn = screen.query_one("#free-first", tui.Button)
            self.assertTrue(btn.display)
            await pilot.click("#free-first")
            await pilot.pause()
            self.assertIn("Added google-free-fast", screen.last_status)
            roles = app.db[engine._wiz.ROLES_KEY]
            self.assertEqual(roles["google-free-fast"]["pools"],
                             ["gemini-3.7-flash"])
            self.assertEqual(roles["google-free-smart"]["pools"],
                             ["gemini-3.0-thinking"])
            # button gone once applied (roles exist now)
            self.assertFalse(screen.query_one("#free-first", tui.Button).display)

    async def test_home_rows_show_probe_latency(self):
        paths = temp_paths()
        db = engine.load_state(paths)
        engine.add_credentials(db, "gemini", ["GK1-FAKE"])
        db["gemini"]["models"] = ["gemini-3.7-flash"]
        engine._wiz.record_probe_latency(db, "gemini", "gemini-3.7-flash", 2.5)
        engine.save_state(db, paths)
        app = WizardApp(paths=paths, status="unknown", status_auto_refresh=False,
                        auto_probe=False)
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.pause()
            home = app.screen
            assert isinstance(home, HomeScreen)
            rows = home.rows
            self.assertTrue(rows)
            self.assertEqual(rows[0]["latency"], 2.5)
            self.assertEqual(rows[0]["latency_txt"], "2.5s")


class CombiningTest(unittest.IsolatedAsyncioTestCase):
    """Milestone 4: automatic combining across providers + suggested pools."""

    def setUp(self):
        self._get, wizard._get = wizard._get, _no_network
        self._post, wizard._post = wizard._post, _no_network
        import urllib.request
        self._urlopen = urllib.request.urlopen
        urllib.request.urlopen = _no_network_urlopen

    def tearDown(self):
        wizard._get = self._get
        wizard._post = self._post
        import urllib.request
        urllib.request.urlopen = self._urlopen

    def make_app(self):
        paths = temp_paths()
        return WizardApp(paths=paths, status="unknown",
                         status_auto_refresh=False, auto_probe=False), paths

    async def test_two_providers_auto_combine(self):
        app, paths = self.make_app()
        db = engine.load_state(paths)
        engine.add_credentials(db, "gemini", ["GK1-FAKE"])
        engine.set_models(db, "gemini", ["gemini-3.7-flash"])
        engine.save_state(db, paths)
        app.db = engine.load_state(paths)
        validated = ([("OR1-FAKE", True, "fine")], None)
        catalog = [("gemini-3.7-flash", "Gemini Flash")]
        probed = [("gemini-3.7-flash", "OK", "fine")]
        with mock.patch.object(wizard, "validate_keys", return_value=validated), \
                mock.patch.object(wizard, "fetch_catalog", return_value=catalog), \
                mock.patch.object(wizard, "test_models", return_value=probed):
            async with app.run_test(size=(100, 50)) as pilot:
                await pilot.pause()
                await _check_keys(pilot, app, "openrouter",
                                  "OR1-FAKE", validated)
                await pilot.click("#save-continue")
                await _wait_until(lambda: isinstance(app.screen, ModelScreen))
                model = app.screen
                assert isinstance(model, ModelScreen)
                await _wait_until(lambda: model.catalog_state == "ready")
                model.query_one("#model-list", tui.SelectionList).select(
                    "gemini-3.7-flash")
                await pilot.click("#probe")
                await _wait_until(lambda: model.catalog_state == "probed")
                await pilot.click("#keep-passing")
                await _wait_until(lambda: isinstance(app.screen, DoneScreen))
                done = app.screen
                assert isinstance(done, DoneScreen)
                self.assertIn("multiple providers", done.last_status)
                back = engine.load_state(paths)
                members = back["_aliases"]["gemini-3.7-flash"]
                self.assertEqual({m["provider"] for m in members},
                                 {"gemini", "openrouter"})
                _deps, pools, _, errors = engine.compile_config(back)
                self.assertEqual(errors, [])
                self.assertEqual(len(pools["gemini-3.7-flash"]), 2)

    async def test_suggested_pool_grouped_on_review(self):
        app, paths = self.make_app()
        db = engine.load_state(paths)
        engine.add_credentials(db, "openrouter", ["OR1-FAKE"])
        engine.add_credentials(db, "zai", ["Z1-FAKE"])
        engine.set_models(db, "openrouter", ["mimo-v2.5-free"])
        engine.set_models(db, "zai", ["mimo-v2.5-free"])
        engine.save_state(db, paths)
        app.db = engine.load_state(paths)
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.pause()
            app.push_screen(DoneScreen())
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, DoneScreen)
            self.assertIn("mimo-v2.5", screen.last_content)
            await pilot.click("#group-suggested")
            await pilot.pause()
            self.assertIn("Grouped mimo-v2.5", screen.last_status)
            back = engine.load_state(paths)
            self.assertIn("mimo-v2.5", back["_aliases"])
            self.assertFalse(screen.applied_ok)


class GatewayTestScreenTest(unittest.IsolatedAsyncioTestCase):
    """Milestone 4: simple default test, diagnose, park failing connections."""

    def setUp(self):
        self._get, wizard._get = wizard._get, _no_network
        self._post, wizard._post = wizard._post, _no_network
        import urllib.request
        self._urlopen = urllib.request.urlopen
        urllib.request.urlopen = _no_network_urlopen

    def tearDown(self):
        wizard._get = self._get
        wizard._post = self._post
        import urllib.request
        urllib.request.urlopen = self._urlopen

    def make_seeded_app(self):
        paths = temp_paths()
        db = engine.load_state(paths)
        engine.add_credentials(db, "gemini", ["GK1-FAKE"])
        engine.set_models(db, "gemini", ["gemini-3.7-flash"])
        engine.save_state(db, paths)
        engine.write_config(db, paths)
        app = WizardApp(paths=paths, status="unknown",
                        status_auto_refresh=False, auto_probe=False)
        app.db = engine.load_state(paths)
        return app, paths

    async def test_run_all_ok(self):
        app, _ = self.make_seeded_app()
        with mock.patch.object(engine, "probe_gateway_alias",
                               return_value=("OK", "choices OK")):
            async with app.run_test(size=(100, 50)) as pilot:
                await pilot.pause()
                app.push_screen(TestScreen())
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, TestScreen)
                await pilot.click("#run-test")
                await _wait_until(lambda: screen.phase == "done")
                self.assertIn("All models answer", screen.last_status)
                self.assertIn("1 OK", screen.last_results)

    async def test_run_failure_diagnose_park(self):
        app, paths = self.make_seeded_app()
        with mock.patch.object(engine, "probe_gateway_alias",
                               return_value=("AUTH_ERROR", "HTTP 401: bad key")), \
                mock.patch.object(engine, "probe_model",
                                  return_value=("AUTH_ERROR", "bad key")):
            async with app.run_test(size=(100, 50)) as pilot:
                await pilot.pause()
                app.push_screen(TestScreen())
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, TestScreen)
                await pilot.click("#run-test")
                await _wait_until(lambda: screen.phase == "done")
                self.assertIn("failing", screen.last_status)
                await pilot.click("#diagnose")
                await _wait_until(lambda: screen.phase == "diagnosed")
                self.assertTrue(screen._parkable())
                await pilot.click("#park-fixes")
                await pilot.pause()
                self.assertIn("Parked 1 connection", screen.last_status)
                back = engine.load_state(paths)
                creds = list(back["gemini"]["credentials"])
                self.assertTrue(all(c.get("quarantined") for c in creds))
                await pilot.click("#review")
                await pilot.pause()
                self.assertIsInstance(app.screen, DoneScreen)
                # terminal flow: Back goes Home, not back through Test
                await pilot.press("escape")
                await pilot.pause()
                self.assertIsInstance(app.screen, HomeScreen)


class StartupTest(unittest.TestCase):
    """Milestone 5: non-interactive entry points, temp env only."""

    def test_version(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(tui.main(["--version"]), 0)
        self.assertEqual(buf.getvalue().strip(), engine.__version__)

    def test_check_ok(self):
        import io
        import os
        from contextlib import redirect_stdout
        from unittest import mock
        paths = temp_paths()
        env = {"LITELLM_DB_FILE": paths.db_file,
               "LITELLM_YAML_FILE": paths.yaml_file,
               "LITELLM_SECRET_FILE": paths.secret_file,
               "OPENCODE_JSON": paths.opencode_json}
        buf = io.StringIO()
        with mock.patch.dict(os.environ, env), redirect_stdout(buf):
            self.assertEqual(tui.main(["--check"]), 0)
        out = buf.getvalue()
        self.assertIn("providers configured: 0", out)
        self.assertIn("OK", out)

    def test_home_has_quit_binding(self):
        bindings = {b[1]: b[0] for b in HomeScreen.BINDINGS}
        self.assertEqual(bindings.get("quit_app"), "q")


if __name__ == "__main__":
    unittest.main()
