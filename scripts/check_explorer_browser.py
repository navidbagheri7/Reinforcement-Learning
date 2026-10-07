"""Check every concept/preset and independent live Bellman values in Chromium."""

import json
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from decision_lab.artifacts import sha256, write_json
from decision_lab.toytext import MazeEnv, model_from_env


def main():
    errors, checked, live = [], [], []
    env = MazeEnv()
    mdp = model_from_env(env)
    env.close()
    previews = Path("assets/previews")
    previews.mkdir(exist_ok=True)
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport=dict(width=1380, height=1024))
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.goto(Path("docs/explorer.html").resolve().as_uri())
        for gamma in [0.5, 0.75, 0.95, 0.99]:
            for sweeps in [0, 1, 2, 20, 100, 500]:
                page.locator("#gamma").fill(str(gamma))
                page.locator("#sweeps").fill(str(sweeps))
                actual = page.evaluate("window.livePlanning")
                value = np.zeros(mdp.states)
                for _ in range(sweeps):
                    q = mdp.reward + gamma * np.einsum("sak,k->sa", mdp.continuation, value)
                    value = np.where(mdp.legal, q, -np.inf).max(1)
                q = mdp.reward + gamma * np.einsum("sak,k->sa", mdp.continuation, value)
                q = np.where(mdp.legal, q, -np.inf)
                np.testing.assert_allclose(actual["value"], value, rtol=0, atol=2e-12)
                np.testing.assert_allclose(
                    actual["residual"], np.max(abs(q.max(1) - value)), rtol=0, atol=2e-12
                )
                # Exact value/action ties may break differently at roundoff level.
                selected_q = q[np.arange(mdp.states), actual["policy"]]
                np.testing.assert_allclose(selected_q, q.max(1), rtol=0, atol=2e-12)
                live.append(dict(gamma=gamma, sweeps=sweeps))
        page.screenshot(path=str(previews / "planning_explorer_desktop.png"), full_page=True)
        groups = page.locator("nav button").all_text_contents()
        for group in groups:
            page.get_by_role("tab", name=group, exact=True).click()
            if group == "Planning and certificates":
                page.set_viewport_size(dict(width=390, height=844))
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), group
                assert page.locator("#pareto-legend span").count() == 8
                page.set_viewport_size(dict(width=1380, height=1024))
                continue
            tasks = page.locator("#task option").all_text_contents()
            for task in tasks:
                page.select_option("#task", task)
                methods = page.locator("#method option").all_text_contents()
                for method in methods:
                    page.select_option("#method", method)
                    for seed in page.locator("#seed option").all_text_contents():
                        page.select_option("#seed", seed)
                        assert page.locator("#title").inner_text() == method
                        assert page.locator("#curves svg path").count() >= 1
                        if page.locator("#media img").count():
                            page.wait_for_function(
                                "document.querySelector('#media img').complete && document.querySelector('#media img').naturalWidth>0"
                            )
                        assert page.locator("#comparison tr").count() >= 1
                        checked.append(dict(group=group, task=task, method=method, seed=seed))
            page.set_viewport_size(dict(width=390, height=844))
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), group
            page.set_viewport_size(dict(width=1380, height=1024))
        page.get_by_role("tab", name="Offline learning", exact=True).click()
        page.screenshot(path=str(previews / "offline_explorer_desktop.png"), full_page=True)
        page.set_viewport_size(dict(width=390, height=844))
        page.screenshot(path=str(previews / "concept_explorer_mobile.png"), full_page=True)
        browser.close()
    assert not errors, errors
    manifest = json.loads(Path("docs/explorer_manifest.json").read_text())
    write_json(
        "docs/explorer_browser_validation.json",
        dict(
            status="passed" if manifest["status"] == "complete" else "partial",
            html_sha256=sha256("docs/explorer.html"),
            checked_presets=checked,
            live_bellman_checks=live,
            mobile_concepts=8,
            javascript_errors=errors,
        ),
    )
    print("Passed", len(checked), "presets and", len(live), "independent live Bellman checks")


if __name__ == "__main__":
    main()
