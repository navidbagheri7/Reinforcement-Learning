"""Optional real-browser checks for gallery controls, images and learning curves."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

from decision_lab.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    errors = []
    cases = []
    with sync_playwright() as browser_runtime:
        browser = browser_runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport=dict(width=1280, height=960))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(Path("docs/simulation_dashboard.html").resolve().as_uri())
        page.wait_for_selector(".card img")
        environments = page.locator("#environment option").evaluate_all("xs=>xs.map(x=>x.value)")
        for env in environments:
            page.select_option("#environment", env)
            methods = page.locator("#method option").evaluate_all("xs=>xs.map(x=>x.value)")
            for method in methods:
                page.select_option("#method", method)
                page.wait_for_function(
                    "[...document.querySelectorAll('.card img')].every(x=>x.complete && x.naturalWidth>0)"
                )
                assert page.locator(".card").count() == 3
                assert page.locator("#curves svg path").count() == 3
                assert page.locator("#summary tr").count() >= (2 if args.allow_partial else 3)
                cases.append(dict(environment=env, method=method, fitting_seeds=3))
        assert args.allow_partial or len(cases) == 8
        page.select_option("#environment", "MiniGrid-DoorKey-6x6-v0")
        page.select_option("#method", "PPO_potential")
        page.screenshot(path="/tmp/rl_simulation_dashboard_desktop.png", full_page=True)
        page.set_viewport_size(dict(width=390, height=844))
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
            "Mobile page overflows"
        )
        assert page.locator(".card").count() == 3
        page.screenshot(path="/tmp/rl_simulation_dashboard_mobile.png", full_page=True)
        browser.close()
    assert not errors, errors
    write_json(
        "docs/simulation_browser_validation.json",
        dict(
            status="passed" if len(cases) == 8 else "partial",
            checked_presets=cases,
            desktop_viewport=[1280, 960],
            mobile_viewport=[390, 844],
            javascript_errors=errors,
        ),
    )
    print(
        "Browser checks passed:",
        len(cases),
        "presets; three actors/images/curves each; mobile fits",
    )


if __name__ == "__main__":
    main()
