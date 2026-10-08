# Use, verify and share the interactive demos

From the repository root, run `python3 -m http.server 8000 --bind 127.0.0.1`
and open `http://127.0.0.1:8000/docs/explorer.html`. When using remote VS Code,
forward that terminal's port and open its forwarded browser address. The
simulation dashboard is at `/docs/simulation_dashboard.html`. Keep the server
running while browsing. Both pages also work as local HTML files with their
relative assets present.

The explorer has eight concept tabs. Planning exposes a known-model maze:
discount and sweep controls perform actual Bellman calculations in the browser.
The remaining tabs select recorded tasks, methods and fitting seeds. Changing
those selectors changes the corresponding measured curves, statistics, source
links and predetermined rollouts. Selection controls do not train a new policy.
The simulation dashboard shows all three fitted actors for a selected preset.

Return intervals condition on a fitted policy; fitting variation is separate.
The global rendering of a partial-view actor does not change its observations.
Official Empty resets repeat a fixed map, start and goal. Full-map, shaped and
local-view presets have distinct information and reward contracts.

To regenerate the pages, run `python scripts/create_explorer.py` and
`python scripts/create_simulation_dashboard.py`. With Playwright and Chromium
installed, the following commands verify their interaction and numerical scope:

```bash
python scripts/check_explorer_browser.py
python scripts/check_simulation_dashboard.py
python scripts/check_demo_controls.py
python scripts/check_demo_controls.py --page docs/simulation_dashboard.html --report docs/simulation_controls_validation.json
```

The lab-specific browser check verifies Bellman iterates. The HTTP control
checker covers every enabled dropdown combination, representative joint slider
settings, selected numerical fields and curves, image bytes, local links and
mobile layouts. Continuous sliders use boundaries and middle values; this is
not a claim to test every real-valued parameter or browser implementation.

Use [publication.md](publication.md) for GitHub Pages and
[reproduction.md](reproduction.md) for fitting the underlying experiments.

See the [user interaction audit](demo_review.md) for the HTTP selector, slider,
asset and numerical verification commands and their precise scope.
