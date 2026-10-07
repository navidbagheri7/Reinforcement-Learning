"use strict";
const evidence = JSON.parse(document.getElementById("evidence").textContent);
const groups = ["Planning and certificates", "Tabular control", "Deep and distributional control", "Offline learning", "Standard control", "Linear approximation", "Prediction and bandits", "Simulations and memory"];
const colors = ["#087f8c", "#ac4b7f", "#c18317", "#4360a5", "#569555", "#815cb4", "#c65a3e", "#596874"];
const $ = id => document.getElementById(id);
let group = groups[0];

function node(tag, text, cls) {
    let el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (cls) el.className = cls;
    return el;
}

function mean(xs) {
    return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function sd(xs) {
    return xs.length > 1 ? Math.sqrt(xs.reduce((s, x) => s + (x - mean(xs)) ** 2, 0) / (xs.length - 1)) : 0;
}

function format(x) {
    return x === null || x === undefined ? "—" : Math.abs(x) > 1e6 || Math.abs(x) < .001 && x !== 0 ? Number(x).toExponential(3) : Number(x).toFixed(3);
}

function options(id, xs) {
    $(id).replaceChildren(...xs.map(x => {
        let el = node("option", x);
        el.value = x;
        return el;
    }));
}

function svgNode(tag, attrs, text) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
    if (text !== undefined) el.textContent = text;
    return el;
}

function chart(id, series, xlabel, ylabel) {
    const width = 820,
        height = 310,
        svg = svgNode("svg", {
            viewBox: `0 0 ${width} ${height}`,
            role: "img",
            "aria-label": ylabel
        });
    const points = series.flatMap(s => s.points);
    if (!points.length) {
        $(id).replaceChildren(node("p", "No recorded curve for this preset", "muted"));
        return;
    }
    const xmin = Math.min(...points.map(p => p[0])),
        xmax = Math.max(...points.map(p => p[0]), xmin + 1),
        ymin = Math.min(...points.map(p => p[1])),
        ymax = Math.max(...points.map(p => p[1]), ymin + 1),
        px = x => 85 + 690 * (x - xmin) / (xmax - xmin),
        py = y => 245 - 220 * (y - ymin) / (ymax - ymin);
    for (let i = 0; i < 5; i++) {
        let y = ymin + (ymax - ymin) * i / 4;
        svg.append(svgNode("line", {
            x1: 85,
            x2: 775,
            y1: py(y),
            y2: py(y),
            stroke: "#d9e6e9"
        }), svgNode("text", {
            x: 2,
            y: py(y) + 4,
            "font-size": 11,
            fill: "#536c7d"
        }, format(y)));
    }
    series.forEach((s, i) => {
        svg.append(svgNode("path", {
            d: s.points.map((p, j) => `${j?"L":"M"}${px(p[0])},${py(p[1])}`).join(" "),
            stroke: colors[i % colors.length],
            "stroke-width": 2.2,
            fill: "none"
        }));
    });
    for (let i = 0; i < 3; i++) {
        const x = xmin + (xmax - xmin) * i / 2;
        svg.append(svgNode("text", {
            x: px(x),
            y: 270,
            "text-anchor": "middle",
            "font-size": 11,
            fill: "#536c7d"
        }, Math.abs(x) >= 1000 ? Math.round(x).toLocaleString() : x.toFixed(2)));
    }
    svg.append(svgNode("text", {
        x: 420,
        y: 300,
        "text-anchor": "middle",
        "font-size": 13,
        fill: "#536c7d"
    }, xlabel));
    $(id).replaceChildren(svg);
}

function fits() {
    return evidence.rows.filter(r => r.group === group && r.task === $("task").value);
}

function render() {
    const rows = fits(),
        methodRows = rows.filter(r => r.method === $("method").value),
        row = methodRows.find(r => String(r.seed) === $("seed").value);
    $("title").textContent = row.method;
    $("profile").textContent = row.profile === "partial" ? "Official egocentric 7×7 symbolic view and direction. The animation reveals the global scene for readers; the agent never sees that map. Memory resets on each episode." : row.profile.startsWith("potential") ? "Full-map symbolic observations. Shaping/geometry/curriculum are declared training choices; evaluation keeps the official unmodified reward." : group === "Offline learning" ? "Pure logged-data learning. Identical logged batches per coverage and seed; final fixed-budget checkpoints. Online test episodes do not select or tune the policy." : row.profile;
    if (row.task === "MiniGrid-Empty-8x8-v0") {
        $("profile").textContent += " The official Empty preset fixes its map, start, direction and goal. Deterministic resets repeat a trajectory; a zero episode interval does not measure unseen-layout generalization.";
    }
    const lecture = group === "Prediction and bandits";
    $("metrics").replaceChildren(...[
        [lecture ? "Final diagnostic" : "Mean held-out return", format(row.score)],
        ["Goal success", row.success === null ? "—" : (100 * row.success).toFixed(1) + "%"],
        ["Exact normalized gap", format(row.gap)],
        ["Visited legal fraction", row.coverage === null ? "—" : (100 * row.coverage).toFixed(1) + "%"],
        [row.selected === null ? "Checkpoint selection" : row.selection_unit === "episode" ? "Selected training episode" : "Selected training interaction", row.selected === null ? "Fixed final" : row.selected.toLocaleString()],
        ["Episode CI", row.ci ? row.ci.map(format).join(" / ") : "Across instances"]
    ].map(([title, value]) => {
        let box = node("div");
        box.append(node("div", value, "number"), node("div", title, "muted"));
        return box;
    }));
    const curveRows = lecture ? rows : methodRows;
    chart("curves", curveRows.map(r => ({
        name: lecture ? r.method : `Seed ${r.seed}`,
        points: r.curve
    })), row.curve_x, row.curve_label);
    $("curve-caption").textContent = row.curve_label + (lecture ? ". Baird is an expected-update counterexample; axes are linear and its divergent values remain unclipped. Other lecture curves are recorded means at up to 200 sampled time indices." : ". All fitting seeds shown; selected test episodes never choose checkpoints.");
    $("legend").replaceChildren(...curveRows.map((r, i) => {
        let el = node("span", (lecture ? r.method : `Seed ${r.seed}`) + "  ");
        el.style.color = colors[i % colors.length];
        return el;
    }));
    $("media").replaceChildren();
    if (row.gif) {
        let img = node("img");
        img.src = row.gif;
        img.alt = `${row.task}, ${row.method}, fitting seed ${row.seed}, fixed test reset 300000`;
        $("media").append(img);
        $("caption").textContent = "Predetermined first held-out episode. Failures and time-limit truncations are retained. The complete score averages 100 resets.";
    } else {
        $("media").append(node("p", lecture ? "This concept is a numerical study. Curves explain prediction error, exploration regret or instability." : "No animation is published for this preset. The numeric held-out episodes and full method comparisons are available below.", "notice"));
        $("caption").textContent = "An absent video is not replaced with a different fitted policy.";
    }
    chart("episode-chart", row.returns.length ? [{
        points: row.returns.map((x, i) => [i + 1, x])
    }] : [], "Held-out episode index", "Recorded official return");
    $("source-links").replaceChildren();
    for (const [title, href] of [
            ["Run metrics", row.metrics],
            ["Numeric source", row.source]
        ]) {
        let link = node("a", title);
        link.href = href;
        $("source-links").append(link, document.createTextNode(" · "));
    }
    const body = $("comparison");
    body.replaceChildren();
    for (const method of [...new Set(rows.map(r => r.method))]) {
        let subset = rows.filter(r => r.method === method),
            xs = subset.map(r => r.score),
            tr = node("tr");
        for (const v of [method, lecture ? subset.reduce((n, r) => n + r.instances, 0) : subset.length, format(mean(xs)), subset.length > 1 ? format(sd(xs)) : "—", `${format(Math.min(...xs))} / ${format(Math.max(...xs))}`]) tr.append(node("td", String(v)));
        body.append(tr);
    }
}

function setSeeds() {
    options("seed", fits().filter(r => r.method === $("method").value).map(r => String(r.seed)));
    render();
}

function setMethods() {
    options("method", [...new Set(fits().map(r => r.method))]);
    setSeeds();
}

function selectGroup(value) {
    group = value;
    document.querySelectorAll("nav button").forEach(b => b.setAttribute("aria-selected", String(b.textContent === group)));
    $("planning").classList.toggle("hidden", group !== groups[0]);
    $("measured").classList.toggle("hidden", group === groups[0]);
    if (group === groups[0]) {
        liveMaze();
        pareto();
        return;
    }
    options("task", [...new Set(evidence.rows.filter(r => r.group === group).map(r => r.task))]);
    setMethods();
}

function liveMaze() {
    const maze = evidence.maze,
        gamma = Number($("gamma").value),
        sweeps = Number($("sweeps").value),
        n = maze.positions.length;
    let value = Array(n).fill(0);

    function q(v, s, a) {
        return maze.reward[s][a] + gamma * maze.transitions[s][a].reduce((sum, [j, p, c]) => sum + c * v[j], 0);
    }

    function bellman(v) {
        return v.map((_, s) => Math.max(...maze.legal[s].map((legal, a) => legal ? q(v, s, a) : -Infinity)));
    }
    for (let k = 0; k < sweeps; k++) value = bellman(value);
    let next = bellman(value),
        residual = Math.max(...next.map((x, i) => Math.abs(x - value[i]))),
        policy = value.map((_, s) => {
            let scores = maze.legal[s].map((legal, a) => legal ? q(value, s, a) : -Infinity);
            return scores.indexOf(Math.max(...scores));
        });
    window.livePlanning = {
        value,
        policy,
        residual,
        gamma,
        sweeps
    };
    $("gamma-value").textContent = gamma.toFixed(2);
    $("sweeps-value").textContent = sweeps;
    $("certificate").textContent = `Bellman residual ${residual.toExponential(3)}; sup-norm optimal-value error ≤ ${(residual/(1-gamma)).toExponential(3)}. Discounted known model, slip 0.1; terminal transitions do not bootstrap. Start value ${format(value[maze.start])}. Floating-point diagnostic; the theoretical bound excludes roundoff.`;
    const cell = 48,
        svg = svgNode("svg", {
            viewBox: `0 0 ${maze.layout[0].length*cell} ${maze.layout.length*cell}`,
            role: "img",
            "aria-label": "Live value and greedy action map"
        }),
        vmin = Math.min(...value),
        vmax = Math.max(...value, vmin + 1e-8),
        arrows = ["←", "↓", "→", "↑"];
    maze.layout.forEach((line, i) => [...line].forEach((char, j) => {
        svg.append(svgNode("rect", {
            x: j * cell,
            y: i * cell,
            width: cell,
            height: cell,
            fill: char === "#" ? "#173245" : "#eef7f6",
            stroke: "white"
        }));
    }));
    maze.positions.forEach(([i, j], s) => {
        let color = `hsl(181 40% ${94-44*(value[s]-vmin)/(vmax-vmin)}%)`;
        svg.append(svgNode("rect", {
            x: j * cell,
            y: i * cell,
            width: cell,
            height: cell,
            fill: color,
            stroke: "white"
        }), svgNode("text", {
            x: j * cell + 24,
            y: i * cell + 21,
            "text-anchor": "middle",
            "font-size": 16,
            fill: "#173245"
        }, s === maze.goal ? "Goal" : s === maze.start ? "Start" : arrows[policy[s]]), svgNode("text", {
            x: j * cell + 24,
            y: i * cell + 39,
            "text-anchor": "middle",
            "font-size": 10,
            fill: "#173245"
        }, value[s].toFixed(2)));
    });
    $("maze").replaceChildren(svg);
}

function pareto() {
    let series = [];
    for (const profile of ["per_user", "global"])
        for (const weighted of [false, true])
            for (const fair of [false, true]) {
                let rows = evidence.planning.pareto.filter(r => r.profile === profile && (r.priorities[1] > 1) === weighted && r.fairness_constraint === fair);
                series.push({
                    name: `${profile}; w2=${weighted?3:1}; fair=${fair}`,
                    points: rows.map(r => [r.normalized_energy, r.normalized_weighted_loss])
                });
            }
    chart("pareto", series, "Normalized discounted energy", "Weighted packet loss");
    $("pareto-legend").replaceChildren(...series.map((s, i) => {
        let el = node("span", s.name + "  ");
        el.style.color = colors[i];
        return el;
    }));
    let table = node("table"),
        header = node("tr");
    ["Profile", "VI–LP difference", "Discounted gap", "Flow residual"].forEach(x => header.append(node("th", x)));
    table.append(header);
    evidence.planning.rows.forEach(r => {
        let tr = node("tr");
        [r.profile, format(r.VI_LP_max_value_difference), format(r.discounted_primal_dual_gap), format(r.occupancy_flow_residual)].forEach(x => tr.append(node("td", String(x))));
        table.append(tr);
    });
    $("planning-summary").replaceChildren(table);
}
groups.forEach(g => {
    let b = node("button", g);
    b.setAttribute("role", "tab");
    b.addEventListener("click", () => selectGroup(g));
    $("concepts").append(b);
});
$("task").addEventListener("change", setMethods);
$("method").addEventListener("change", setSeeds);
$("seed").addEventListener("change", render);
$("gamma").addEventListener("input", liveMaze);
$("sweeps").addEventListener("input", liveMaze);
selectGroup(groups[0]);
