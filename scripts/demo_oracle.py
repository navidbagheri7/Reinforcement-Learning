"""Source-bound numerical assertions for the currently selected demo controls."""

import math
import re

import numpy as np

_CACHE = None


def close(text, expected):
    if expected is None:
        assert text == "—", (text, expected)
        return
    if isinstance(expected, str):
        assert text == expected, (text, expected)
        return
    found = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?", text, re.I)
    assert found, (text, expected)
    token = found.group().replace(",", "")
    places = len(token.split("e")[0].partition(".")[2])
    exponent = int(token.lower().partition("e")[2] or "0")
    tolerance = 0.501 * 10 ** (exponent - places)
    assert math.isclose(float(token), expected, rel_tol=1e-12, abs_tol=tolerance), (text, expected)


def cards(page, selector, expected):
    actual = page.locator(selector).all_text_contents()
    assert len(actual) == len(expected), (actual, expected)
    for text, value in zip(actual, expected, strict=True):
        close(text, value)


def image_sources(page, selector, expected):
    actual = page.locator(selector).evaluate_all('xs=>xs.map(x=>x.getAttribute("src"))')
    assert actual == expected, (actual, expected)


def check(page):
    global _CACHE
    if _CACHE is None:
        _CACHE = page.evaluate("evidence")
    e = _CACHE
    if page.locator("#environment").count():
        env = page.locator("#environment").input_value()
        method = page.locator("#method").input_value()
        rows = sorted(
            [r for r in e["fits"] if r["env"] == env and r["key"] == method],
            key=lambda r: r["seed"],
        )
        image_sources(page, ".card img", [r["gif"] for r in rows])
        cards(page, ".card .metric", [r["test"]["mean"] for r in rows])
        return
    if not page.locator("#seed:visible").count():
        return
    group = page.locator('nav [aria-selected="true"]').inner_text()
    task = page.locator("#task").input_value()
    method = page.locator("#method").input_value()
    seed = page.locator("#seed").input_value()
    r = next(
        r
        for r in e["rows"]
        if r["group"] == group
        and r["task"] == task
        and r["method"] == method
        and str(r["seed"]) == seed
    )
    actual = page.locator("#metrics .number").all_text_contents()
    expected = [
        r["score"],
        100 * r["success"] if r["success"] is not None else None,
        r["gap"],
        100 * r["coverage"] if r["coverage"] is not None else None,
        r["selected"] if r["selected"] is not None else "Fixed final",
    ]
    assert len(actual) == 6
    for text, value in zip(actual[:5], expected):
        close(text, value)
    if r["ci"]:
        for text, value in zip(actual[5].split(" / "), r["ci"]):
            close(text, value)
    else:
        assert actual[5] == "Across instances"
    if r["gif"]:
        image_sources(page, "#media img", [r["gif"]])
    rows = [
        x
        for x in e["rows"]
        if x["group"] == group
        and x["task"] == task
        and (group == "Prediction and bandits" or x["method"] == method)
    ]
    curves = [x["curve"] for x in rows]
    paths = page.locator("#curves svg path").all()
    assert len(paths) == len(curves)
    all_points = np.concatenate([np.asarray(c, dtype=np.float64) for c in curves])
    xmin, ymin = all_points.min(0)
    xmax, ymax = all_points.max(0)
    xmax = max(xmax, xmin + 1)
    ymax = max(ymax, ymin + 1)
    for path, curve in zip(paths, curves):
        observed = np.asarray(
            [
                [float(a), float(b)]
                for a, b in re.findall(r"[ML]([^, ]+),([^ ML]+)", path.get_attribute("d"))
            ]
        )
        original = np.asarray(curve, dtype=np.float64)
        expected = np.column_stack(
            [
                85 + 690 * (original[:, 0] - xmin) / (xmax - xmin),
                245 - 220 * (original[:, 1] - ymin) / (ymax - ymin),
            ]
        )
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-9)
