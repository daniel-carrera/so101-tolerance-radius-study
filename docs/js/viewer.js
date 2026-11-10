// SO-101 trial map. Plain JavaScript, no framework and no build step.
// Reads data/trials.json, or its twin data/trials.js when the page is opened from a file URL (a browser will not
// fetch a JSON file from disk). Every trial, position, condition and outcome comes from that file; every count on the
// page is made from it here.
(function () {
  "use strict";

  // The split the study's results table uses: within 20 mm of a demonstration, and beyond.
  const RING_MM = 20;
  const SVGNS = "http://www.w3.org/2000/svg";

  const CONDITIONS = {
    act_clustered: { label: "Clustered policy", chip: "Policy", shape: "circle" },
    replay: { label: "Replay", chip: "Replay", shape: "diamond" },
    act_spread: { label: "Spread policy", chip: "Policy", shape: "circle" },
  };
  const SETS = {
    clustered: { label: "Clustered set", conditions: ["act_clustered", "replay"] },
    spread: { label: "Spread set", conditions: ["act_spread"] },
  };
  const OUTCOMES = { success: "Succeeded", failure: "Missed" };

  const $ = (id) => document.getElementById(id);
  const fmt = (v) => v.toFixed(1);
  const byId = (a, b) => a.id.localeCompare(b.id, "en", { numeric: true });

  function svg(tag, attrs, parent) {
    const node = document.createElementNS(SVGNS, tag);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function html(tag, attrs, parent, text) {
    const node = document.createElement(tag);
    for (const key in attrs || {}) {
      if (key === "class") node.className = attrs[key];
      else node.setAttribute(key, attrs[key]);
    }
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }
  function clearSvg(node) {
    for (const child of Array.from(node.childNodes)) {
      if (child.nodeName !== "title" && child.nodeName !== "desc") node.removeChild(child);
    }
  }

  // ----------------------------------------------------------------------------------------------------- loading

  function loadData() {
    const fromScript = () => new Promise((resolve, reject) => {
      if (window.TRIAL_DATA) { resolve(window.TRIAL_DATA); return; }
      const tag = document.createElement("script");
      tag.src = "data/trials.js";
      tag.onload = () => (window.TRIAL_DATA ? resolve(window.TRIAL_DATA) : reject(new Error("data/trials.js holds no data")));
      tag.onerror = () => reject(new Error("data/trials.js did not load"));
      document.head.appendChild(tag);
    });
    if (location.protocol === "file:" || !window.fetch) return fromScript();
    return fetch("data/trials.json")
      .then((response) => { if (!response.ok) throw new Error("HTTP " + response.status); return response.json(); })
      .catch(fromScript);
  }

  // ------------------------------------------------------------------------------------------------------- state

  let D, TABLE, W, H, TURN, TRIALS, BY_ID, DEMOS, POINTS;

  // Demonstrations aimed at the same planned start (five at each point of the clustered set) are drawn as one cross
  // at the average of their measured starts, with one ring: their measured starts lie a millimeter or so apart, and
  // drawn one by one they smear into each other. Distances on the page stay those of the log, to each measured start.
  function demoPoints(demos) {
    const groups = new Map();
    for (const d of demos) {
      const at = d.planned_mm || [d.x_mm, d.y_mm];
      const key = at[0].toFixed(3) + "," + at[1].toFixed(3);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(d);
    }
    return [...groups.values()].map((members) => {
      const x = members.reduce((sum, d) => sum + d.x_mm, 0) / members.length;
      const y = members.reduce((sum, d) => sum + d.y_mm, 0) / members.length;
      const spread = Math.max(...members.map((d) => Math.hypot(d.x_mm - x, d.y_mm - y)));
      return { x_mm: x, y_mm: y, n: members.length, spread_mm: spread, episodes: members.map((d) => d.episode) };
    });
  }
  const state = {
    set: "clustered",
    show: { act_clustered: true, replay: true, act_spread: true },
    outcome: { success: true, failure: true },
    selected: null,
    hover: null,
  };
  // Map geometry of the last draw: millimetres per screen pixel, and the marks drawn.
  let mapK = 0.5;

  // Table frame to map: turned the way the front camera sees the table.
  const X = (x) => (TURN.x_runs_left ? W - x : x);
  const Y = (y) => (TURN.y_runs_down ? y : H - y);

  const inSet = (t) => SETS[state.set].conditions.includes(t.condition);
  const visible = () => TRIALS.filter((t) => inSet(t) && state.show[t.condition] && state.outcome[t.outcome]);

  function who(t) {
    if (t.condition === "replay") return "Replay of demonstration " + t.replay_episode;
    return CONDITIONS[t.condition].label;
  }
  function distanceText(t) {
    if (t.condition === "replay") return fmt(t.d_mm) + "\u00a0mm from the start of demonstration " + t.replay_episode;
    return fmt(t.d_mm) + "\u00a0mm from the nearest demonstration it was trained on";
  }
  function shortDistance(t) {
    return t.condition === "replay"
      ? fmt(t.d_mm) + "\u00a0mm from its demonstration's start"
      : fmt(t.d_mm) + "\u00a0mm from the nearest demonstration";
  }

  // ------------------------------------------------------------------------------------------------- the marks

  // One trial's mark: shape by who ran it, filled blue for a success, hollow orange for a miss.
  function drawMark(parent, t, cx, cy, k, extra) {
    const g = svg("g", { class: "trial", "data-id": t.id }, parent);
    const ok = t.outcome === "success";
    const shape = CONDITIONS[t.condition].shape;
    const cls = ok ? "ok-fill" : "miss-ring";
    const strokePx = ok ? 1.5 : 2.2;
    if (shape === "circle") {
      svg("circle", { cx, cy, r: 5 * k, class: cls, "stroke-width": strokePx }, g);
    } else {
      const s = 6.2 * k;
      svg("path", { d: `M${cx},${cy - s}L${cx + s},${cy}L${cx},${cy + s}L${cx - s},${cy}Z`, class: cls, "stroke-width": strokePx }, g);
    }
    if (extra && t.clip) svg("circle", { cx, cy, r: 10 * k, class: "clip-mark", "stroke-width": 1.2 }, g);
    return g;
  }

  function iconSvg(kind, size) {
    const s = size || 14;
    const node = svg("svg", { width: s, height: s, viewBox: "-8 -8 16 16", "aria-hidden": "true", focusable: "false" });
    if (kind === "circle") svg("circle", { r: 5, fill: "currentColor" }, node);
    else if (kind === "diamond") svg("path", { d: "M0,-6.5L6.5,0L0,6.5L-6.5,0Z", fill: "currentColor" }, node);
    else if (kind === "success") svg("circle", { r: 5, class: "ok-fill" }, node);
    else if (kind === "failure") svg("circle", { r: 4.6, class: "miss-ring", "stroke-width": 2.2 }, node);
    else if (kind === "demo") {
      svg("path", { d: "M-5,0H5M0,-5V5", class: "demo", "stroke-width": 1.4 }, node);
    } else if (kind === "clip") {
      svg("circle", { r: 6.5, class: "clip-mark", "stroke-width": 1.2 }, node);
      svg("circle", { r: 3.4, class: "ok-fill" }, node);
    }
    return node;
  }

  // ------------------------------------------------------------------------------------------------- controls

  function counts(list) {
    return { n: list.length, ok: list.filter((t) => t.outcome === "success").length };
  }

  function drawControls() {
    const sets = $("sets");
    sets.textContent = "";
    for (const key of Object.keys(SETS)) {
      const n = TRIALS.filter((t) => SETS[key].conditions.includes(t.condition)).length;
      const b = html("button", { type: "button", "data-set": key, "aria-pressed": String(state.set === key) }, sets);
      html("span", {}, b, SETS[key].label);
      html("span", { class: "count" }, b, n + " trials");
    }

    const conds = $("conditions");
    conds.textContent = "";
    for (const c of SETS[state.set].conditions) {
      const n = TRIALS.filter((t) => t.condition === c).length;
      const b = html("button", { type: "button", "data-condition": c, "aria-pressed": String(state.show[c]) }, conds);
      b.appendChild(iconSvg(CONDITIONS[c].shape));
      html("span", {}, b, state.set === "clustered" ? CONDITIONS[c].chip : CONDITIONS[c].label);
      html("span", { class: "count" }, b, String(n));
    }

    const outs = $("outcomes");
    outs.textContent = "";
    const inThisSet = TRIALS.filter(inSet);
    for (const o of Object.keys(OUTCOMES)) {
      const n = inThisSet.filter((t) => t.outcome === o).length;
      const b = html("button", { type: "button", "data-outcome": o, "aria-pressed": String(state.outcome[o]) }, outs);
      b.appendChild(iconSvg(o));
      html("span", {}, b, OUTCOMES[o]);
      html("span", { class: "count" }, b, String(n));
    }

    const demos = DEMOS[state.set];
    const perCell = {};
    for (const d of demos) perCell[d.cell] = (perCell[d.cell] || 0) + 1;
    const cells = Object.keys(perCell).length;
    const sizes = new Set(Object.values(perCell));
    const distinct = new Set(demos.map((d) => d.x_mm.toFixed(1) + "," + d.y_mm.toFixed(1))).size;
    let note;
    const points = POINTS[state.set];
    const shared = points.filter((p) => p.n > 1);
    if (state.set === "clustered") {
      const each = sizes.size === 1 ? `${[...sizes][0]} from each of ${cells} points` : `from ${cells} points`;
      note = `The clustered set has ${demos.length} demonstrations, ${each}. The clustered policy, an ACT policy, learned from them. A replay trial plays one of them back as recorded, with no camera input.`;
      if (shared.length) {
        const most = Math.max(...shared.map((p) => p.spread_mm));
        note += ` Each cross is the average of the measured starts at its point, which all lie within ${fmt(Math.ceil(most * 10) / 10)}\u00a0mm of it.`;
      }
    } else {
      const pairs = new Set(TRIALS.filter((t) => t.condition === "act_spread" && t.pair_id).map((t) => t.pair_id)).size;
      const starts = distinct === demos.length ? "each from a different start" : `from ${distinct} different starts`;
      note = `The spread set has ${demos.length} demonstrations, ${starts}, over the same ${cells} cells. The spread policy, an ACT policy with the same settings, learned from them. It ran at the same ${pairs} placements as ${pairs} of the clustered policy's trials.`;
    }
    $("set-note").textContent = note;
    const keyDemo = $("key-demo");
    if (keyDemo) {
      const n = new Set(shared.map((p) => p.n));
      keyDemo.textContent = shared.length && n.size === 1
        ? `Where demonstrations started, ${[...n][0]} at each cross`
        : "Where a demonstration started";
    }
  }

  // ------------------------------------------------------------------------------------------------------ map

  function drawMap() {
    const map = $("map");
    clearSvg(map);
    const c = TABLE.cell_mm;
    const pad = 6;
    const vb = { x: -pad, y: -pad, w: W + 2 * pad, h: H + 2 * pad };
    map.setAttribute("viewBox", `${vb.x} ${vb.y} ${vb.w} ${vb.h}`);
    const width = map.getBoundingClientRect().width || 640;
    const k = (mapK = vb.w / width);

    const defs = svg("defs", {}, map);
    const hatch = svg("pattern", { id: "hatch", patternUnits: "userSpaceOnUse", width: 6 * k, height: 6 * k, patternTransform: "rotate(45)" }, defs);
    svg("line", { x1: 0, y1: 0, x2: 0, y2: 6 * k, class: "hatch-line" }, hatch);
    const clip = svg("clipPath", { id: "grid-clip" }, defs);
    svg("rect", { x: 0, y: 0, width: W, height: H }, clip);
    const mask = svg("mask", { id: "ring-mask", maskUnits: "userSpaceOnUse", x: vb.x, y: vb.y, width: vb.w, height: vb.h }, defs);
    svg("rect", { x: vb.x, y: vb.y, width: vb.w, height: vb.h, fill: "white" }, mask);

    // Cells, held-out cells hatched.
    const cells = svg("g", {}, map);
    for (let i = 0; i < TABLE.cols; i++) {
      for (let j = 0; j < TABLE.rows; j++) {
        const label = String.fromCharCode(65 + i) + (j + 1);
        const held = TABLE.held_out_cells.includes(label);
        const x0 = Math.min(X(i * c), X((i + 1) * c));
        const y0 = Math.min(Y(j * c), Y((j + 1) * c));
        if (held) svg("rect", { x: x0, y: y0, width: c, height: c, class: "held" }, cells);
        svg("rect", { x: x0, y: y0, width: c, height: c, class: "cell" }, cells);
        const text = svg("text", { x: x0 + 4 * k, y: y0 + 13 * k, class: "cell-label" + (held ? " is-held" : ""), "font-size": 11 * k }, cells);
        text.textContent = label;
      }
    }

    // The 20 mm around each demonstration start, as one shaded region with its outer edge drawn.
    const points = POINTS[state.set];
    const rings = svg("g", { "clip-path": "url(#grid-clip)" }, map);
    const union = svg("g", { class: "ring-union" }, rings);
    const edge = svg("g", { class: "ring-edge", mask: "url(#ring-mask)" }, rings);
    for (const d of points) {
      svg("circle", { cx: X(d.x_mm), cy: Y(d.y_mm), r: RING_MM }, union);
      svg("circle", { cx: X(d.x_mm), cy: Y(d.y_mm), r: RING_MM, "stroke-width": 2.4 }, edge);
      svg("circle", { cx: X(d.x_mm), cy: Y(d.y_mm), r: RING_MM, fill: "black" }, mask);
    }
    svg("rect", { x: 0, y: 0, width: W, height: H, class: "outline" }, map);

    // Demonstration starts.
    const demoLayer = svg("g", {}, map);
    const arm = 4.5 * k;
    for (const d of points) {
      const x = X(d.x_mm), y = Y(d.y_mm);
      svg("path", { d: `M${x - arm},${y}H${x + arm}M${x},${y - arm}V${y + arm}`, class: "demo", "stroke-width": 1.4 }, demoLayer);
    }

    svg("g", { id: "sel-under" }, map);
    const marks = svg("g", { id: "marks" }, map);
    // Misses first, so that a success is not hidden under a hollow mark.
    const shown = visible().sort((a, b) => (a.outcome === b.outcome ? 0 : a.outcome === "failure" ? -1 : 1));
    for (const t of shown) drawMark(marks, t, X(t.x_mm), Y(t.y_mm), k, true);
    svg("g", { id: "sel-over" }, map);
    svg("g", { id: "hover-layer" }, map);

    const xSide = TURN.x_runs_left ? "right" : "left";
    const ySide = TURN.y_runs_down ? "top" : "bottom";
    $("orient").textContent = `Drawn the way the front camera sees the table: column A is on the ${xSide} and row 1 at the ${ySide}. ` +
      `Showing ${shown.length} of ${TRIALS.filter(inSet).length} trials in this set.`;
    drawSelection();
  }

  function drawSelection() {
    const under = $("sel-under"), over = $("sel-over");
    if (!under) return;
    under.textContent = "";
    over.textContent = "";
    const t = state.selected && BY_ID[state.selected];
    if (!t || !inSet(t)) return;
    const k = mapK, x = X(t.x_mm), y = Y(t.y_mm);
    const nx = X(t.nearest_xy_mm[0]), ny = Y(t.nearest_xy_mm[1]);
    const half = TABLE.cube_edge_mm / 2;
    svg("rect", { x: x - half, y: y - half, width: 2 * half, height: 2 * half, class: "footprint", "stroke-width": 1.2 }, under);
    const point = POINTS[t.demo_set].find((p) => p.episodes.includes(t.nearest_episode));
    if (point) svg("circle", { cx: X(point.x_mm), cy: Y(point.y_mm), r: RING_MM, class: "near-ring", "stroke-width": 1.8 }, under);
    svg("line", { x1: x, y1: y, x2: nx, y2: ny, class: "tether", "stroke-width": 1.5 }, under);
    svg("circle", { cx: x, cy: y, r: 9.5 * k, class: "halo", "stroke-width": 2 }, over);
    drawMark(over, t, x, y, k, false);
  }

  function drawHover() {
    const layer = $("hover-layer");
    if (!layer) return;
    layer.textContent = "";
    const t = state.hover && BY_ID[state.hover];
    if (!t || !inSet(t) || t.id === state.selected) return;
    svg("circle", { cx: X(t.x_mm), cy: Y(t.y_mm), r: 9 * mapK, class: "halo", "stroke-width": 1.5 }, layer);
  }

  // The trial nearest the pointer, if it is close enough to mean it.
  function nearestOnMap(event) {
    const map = $("map");
    const ctm = map.getScreenCTM();
    if (!ctm) return null;
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    const reach = (event.pointerType === "touch" ? 22 : 14) * mapK;
    let best = null, bestD = Infinity;
    for (const t of visible()) {
      const d = Math.hypot(X(t.x_mm) - p.x, Y(t.y_mm) - p.y);
      if (d < bestD) { best = t; bestD = d; }
    }
    return bestD <= reach ? best : null;
  }

  function markClientPoint(t) {
    const map = $("map");
    const ctm = map.getScreenCTM();
    const p = new DOMPoint(X(t.x_mm), Y(t.y_mm)).matrixTransform(ctm);
    return { x: p.x, y: p.y };
  }

  // ------------------------------------------------------------------------------------------------ the readout

  function showTip(t, at) {
    const tip = $("tip");
    if (!t) { tip.hidden = true; return; }
    tip.textContent = "";
    const head = html("strong", {}, tip, t.id + " \u00b7 " + OUTCOMES[t.outcome]);
    head.className = "";
    html("span", { class: "tip-sub" }, tip, who(t));
    html("span", { class: "tip-sub" }, tip, shortDistance(t));
    html("span", { class: "tip-sub" }, tip, "Cell " + t.cell + (t.held_out ? ", held out" : ""));
    tip.hidden = false;
    const tw = tip.offsetWidth, th = tip.offsetHeight, margin = 8;
    let left = at.x + 14;
    let top = at.y - th - 12;
    if (left + tw > window.innerWidth - margin) left = at.x - tw - 14;
    left = Math.max(margin, Math.min(window.innerWidth - tw - margin, left));
    if (top < margin) top = at.y + 16;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  function setHover(t, at) {
    const id = t ? t.id : null;
    if (id !== state.hover) {
      state.hover = id;
      drawHover();
      drawStripHover();
    }
    showTip(t, at);
  }

  // ------------------------------------------------------------------------------------------------- the panel

  const level = (v) => (Number.isInteger(v) ? String(v) : fmt(v));

  function startText(t) {
    if (t.pair_id) return "Drawn at random for the workspace test, and run with both policies at the same spot.";
    if (t.staircase_id) {
      return `A staircase asked for ${level(t.level_mm)}\u00a0mm. It moves the cube farther after a success and nearer after a miss.`;
    }
    if (t.level_mm === 0) return "Asked for 0\u00a0mm: the start of a demonstration itself.";
    if (t.level_mm !== null && t.level_mm !== undefined) return `Asked for ${level(t.level_mm)}\u00a0mm, fixed in advance.`;
    return "Not recorded.";
  }

  function outcomeText(t) {
    if (t.outcome === "success") return `The cube ended in the basket. Graded progress: ${t.subgoal} of 5.`;
    const kind = t.failure ? ` Failure type: ${t.failure}.` : "";
    return `The arm missed.${kind} Graded progress: ${t.subgoal} of 5.`;
  }

  function fact(dl, term, value) {
    const row = html("div", {}, dl);
    html("dt", {}, row, term);
    html("dd", {}, row, value);
  }

  function drawPanel() {
    const panel = $("panel");
    panel.textContent = "";
    const t = state.selected && BY_ID[state.selected];
    if (!t || !inSet(t)) { drawEmptyPanel(panel); return; }

    const head = html("div", { class: "panel-head" }, panel);
    html("h2", {}, head, t.id);
    const ok = t.outcome === "success";
    const badge = html("span", { class: "badge " + (ok ? "ok" : "miss") }, head);
    badge.appendChild(iconSvg(t.outcome, 12));
    html("span", {}, badge, OUTCOMES[t.outcome]);
    html("p", { class: "who" }, panel, who(t));

    const dl = html("dl", { class: "facts" }, panel);
    fact(dl, "Distance", distanceText(t));
    fact(dl, "Where the cube started", `Cell ${t.cell}${t.held_out ? ", a held-out cell" : ""}, at x\u00a0${fmt(t.x_mm)}, y\u00a0${fmt(t.y_mm)}\u00a0mm`);
    fact(dl, "Reset", `Placed with the overlay ${fmt(resetError(t))}\u00a0mm from its planned target.`);
    fact(dl, "How the start was chosen", startText(t));
    fact(dl, "Outcome", outcomeText(t));
    fact(dl, "Block", `${t.block}: ${D.blocks[t.block] || ""}`);

    const note = html("p", { class: "note" }, panel);
    html("small", {}, note, "Scorer's note from the rig, spelling corrected");
    note.appendChild(document.createTextNode(t.notes ? t.notes : "No note."));

    if (t.pair_id) {
      const twin = TRIALS.find((u) => u.pair_id === t.pair_id && u.id !== t.id);
      if (twin) {
        const p = html("p", { class: "pair" }, panel);
        const b = html("button", { type: "button", class: "pick", "data-go": twin.id }, p);
        b.appendChild(iconSvg(twin.outcome, 12));
        html("span", {}, b, `Same placement, ${CONDITIONS[twin.condition].label.toLowerCase()}: ${twin.id}, ${OUTCOMES[twin.outcome].toLowerCase()}`);
      }
    }

    if (t.clip) {
      const fig = html("figure", { class: "clip" }, panel);
      const video = html("video", {
        controls: "", muted: "", playsinline: "", loop: "", preload: "metadata",
        poster: `assets/${t.clip}.jpg`, width: "640", height: "480",
        "aria-label": `Front camera video of trial ${t.id}: ${who(t).toLowerCase()}, ${shortDistance(t)}, ${OUTCOMES[t.outcome].toLowerCase()}.`,
      }, fig);
      video.muted = true;
      html("source", { src: `assets/${t.clip}.mp4`, type: "video/mp4" }, video);
      html("figcaption", {}, fig, "Front camera.");
    } else {
      html("p", { class: "no-clip" }, panel, "No video of this trial is on this page.");
    }

    const list = visible().sort(byId);
    const i = list.findIndex((u) => u.id === t.id);
    const nav = html("div", { class: "panel-nav" }, panel);
    html("span", { class: "where" }, nav, i >= 0 ? `${i + 1} of ${list.length} shown` : "Hidden by the filters");
    html("button", { type: "button", class: "step", "data-step": "-1", "aria-label": "Previous trial" }, nav, "\u2190 Previous");
    html("button", { type: "button", class: "step", "data-step": "1", "aria-label": "Next trial" }, nav, "Next \u2192");
    html("button", { type: "button", class: "step", "data-clear": "" }, nav, "Close");
  }

  function drawEmptyPanel(panel) {
    const box = html("div", { class: "empty" }, panel);
    html("h2", {}, box, "Pick a trial");
    html("p", { class: "lead" }, box, "Hover over a dot or tap it to see where the cube started, how far that was from a demonstration, and how the trial ended.");
    const clips = TRIALS.filter((t) => t.clip).sort(byId);
    if (!clips.length) return;
    html("p", {}, box, clips.length === 1 ? "One trial has a video:" : `${clips.length} trials have a video:`);
    for (const t of clips) {
      const b = html("button", { type: "button", class: "pick", "data-go": t.id }, box);
      b.appendChild(iconSvg(t.outcome, 12));
      const label = html("span", {}, b);
      label.appendChild(document.createTextNode(t.id + " "));
      html("small", {}, label, `${CONDITIONS[t.condition].label.toLowerCase()}, ${shortDistance(t)}, ${OUTCOMES[t.outcome].toLowerCase()}`);
    }
  }

  // ------------------------------------------------------------------------------------------ by distance strip

  let stripGeom = [];  // per row: { svg, condition, points: [{ t, x, y }] }

  function drawStrip() {
    const box = $("strip");
    box.textContent = "";
    stripGeom = [];
    const conditions = SETS[state.set].conditions.filter((c) => state.show[c]);
    const all = TRIALS;
    const maxD = Math.ceil(Math.max(...all.map((t) => t.d_mm)) / 10) * 10;
    const width = box.getBoundingClientRect().width || 600;
    const left = 10, right = 14;
    const sx = (d) => left + (d / maxD) * (width - left - right);
    const r = 5, step = 2 * r + 2;

    $("distance-note").textContent = state.set === "clustered"
      ? `The same trials, laid out by how far the cube started from the nearest demonstration. For a replay it is the demonstration it plays back. The dashed line is at ${RING_MM}\u00a0mm.`
      : `The same trials, laid out by how far the cube started from the nearest demonstration in the spread set. The dashed line is at ${RING_MM}\u00a0mm.`;

    for (const c of conditions) {
      const mine = all.filter((t) => t.condition === c);
      const near = counts(mine.filter((t) => t.d_mm <= RING_MM));
      const far = counts(mine.filter((t) => t.d_mm > RING_MM));
      const row = html("div", { class: "strip-row" }, box);
      const head = html("div", { class: "strip-head" }, row);
      const name = html("span", {}, head);
      name.appendChild(iconSvg(CONDITIONS[c].shape, 12));
      name.appendChild(document.createTextNode(" " + CONDITIONS[c].label));
      const tally = html("span", { class: "tally" }, head);
      tally.appendChild(document.createTextNode(`Within ${RING_MM}\u00a0mm: `));
      html("b", {}, tally, `${near.ok} of ${near.n}`);
      tally.appendChild(document.createTextNode(" succeeded. Beyond: "));
      html("b", {}, tally, `${far.ok} of ${far.n}`);
      tally.appendChild(document.createTextNode("."));

      // A simple swarm: each dot takes the first lane, from the middle outward, where it overlaps no other.
      const shown = mine.filter((t) => state.outcome[t.outcome]).sort((a, b) => a.d_mm - b.d_mm || byId(a, b));
      const lanes = new Map();
      const placed = [];
      let minLane = 0, maxLane = 0;
      for (const t of shown) {
        const x = sx(t.d_mm);
        for (let n = 0; ; n++) {
          const lane = n === 0 ? 0 : (n % 2 ? (n + 1) / 2 : -n / 2);
          const last = lanes.get(lane);
          if (last === undefined || x - last >= step) {
            lanes.set(lane, x);
            placed.push({ t, x, lane });
            minLane = Math.min(minLane, lane);
            maxLane = Math.max(maxLane, lane);
            break;
          }
        }
      }
      const height = (maxLane - minLane + 1) * step + 8;
      const lane = svg("svg", { class: "lane", width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
        "aria-label": `${CONDITIONS[c].label}: within ${RING_MM} mm ${near.ok} of ${near.n} succeeded, beyond ${far.ok} of ${far.n}.` }, row);
      const mid = 4 + (0 - minLane) * step + step / 2;
      svg("line", { x1: left, x2: width - right, y1: mid, y2: mid, class: "lane-line" }, lane);
      svg("line", { x1: sx(RING_MM), x2: sx(RING_MM), y1: 0, y2: height, class: "cut-line" }, lane);
      const points = [];
      const missesFirst = placed.slice().sort((a, b) => (a.t.outcome === b.t.outcome ? 0 : a.t.outcome === "failure" ? -1 : 1));
      for (const p of missesFirst) {
        const y = mid + p.lane * step;
        drawMark(lane, p.t, p.x, y, 1, false);
        points.push({ t: p.t, x: p.x, y });
      }
      svg("g", { class: "strip-sel" }, lane);
      stripGeom.push({ svg: lane, condition: c, points });
    }

    // The axis, shared by the rows.
    const axis = svg("svg", { class: "strip-axis", width, height: 34, viewBox: `0 0 ${width} 34`, "aria-hidden": "true" }, box);
    svg("line", { x1: left, x2: width - right, y1: 4, y2: 4, class: "axis-line" }, axis);
    for (let d = 0; d <= maxD; d += 10) {
      svg("line", { x1: sx(d), x2: sx(d), y1: 4, y2: 9, class: "axis-line" }, axis);
      const label = svg("text", { x: sx(d), y: 22, "text-anchor": d === 0 ? "start" : d === maxD ? "end" : "middle", class: "axis-tick" }, axis);
      label.textContent = d === maxD ? d + " mm" : String(d);
    }
    drawStripHover();
  }

  function drawStripHover() {
    for (const row of stripGeom) {
      const layer = row.svg.querySelector(".strip-sel");
      layer.textContent = "";
      for (const p of row.points) {
        if (p.t.id === state.selected) svg("circle", { cx: p.x, cy: p.y, r: 9.5, class: "halo", "stroke-width": 2 }, layer);
        else if (p.t.id === state.hover) svg("circle", { cx: p.x, cy: p.y, r: 9, class: "halo", "stroke-width": 1.5 }, layer);
      }
    }
  }

  function nearestOnStrip(event) {
    for (const row of stripGeom) {
      if (row.svg !== event.currentTarget) continue;
      const box = row.svg.getBoundingClientRect();
      const px = event.clientX - box.left, py = event.clientY - box.top;
      const reach = event.pointerType === "touch" ? 18 : 11;
      let best = null, bestD = Infinity;
      for (const p of row.points) {
        const d = Math.hypot(p.x - px, p.y - py);
        if (d < bestD) { best = p; bestD = d; }
      }
      return bestD <= reach ? { t: best.t, at: { x: box.left + best.x, y: box.top + best.y } } : null;
    }
    return null;
  }

  // ------------------------------------------------------------------------------------------------- the table

  function drawTable() {
    const body = $("rows");
    body.textContent = "";
    const list = visible().sort(byId);
    $("table-summary").textContent = `The same ${list.length} trials as a table`;
    for (const t of list) {
      const tr = html("tr", { "data-id": t.id, "aria-current": String(t.id === state.selected) }, body);
      const first = html("td", {}, tr);
      html("button", { type: "button", "data-go": t.id }, first, t.id);
      html("td", {}, tr, who(t));
      html("td", {}, tr, t.cell + (t.held_out ? " (held out)" : ""));
      html("td", { class: "num" }, tr, fmt(t.d_mm) + "\u00a0mm");
      html("td", { class: t.outcome === "success" ? "out-ok" : "out-miss" }, tr, OUTCOMES[t.outcome]);
      html("td", { class: "note-cell" }, tr, t.notes || "");
    }
  }

  // ------------------------------------------------------------------------------------------------ the overview

  function drawOverview() {
    const node = $("overview");
    clearSvg(node);
    const pts = [[0, 0], [W, H]];
    for (const tag of D.tags) for (const p of tag.corners_mm) pts.push([X(p[0]), Y(p[1])]);
    for (const p of pts.slice(0, 2)) { p[0] = X(p[0]); p[1] = Y(p[1]); }
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const pad = 30;
    const vb = [Math.min(...xs) - pad - 20, Math.min(...ys) - pad - 20, Math.max(...xs) - Math.min(...xs) + 2 * pad + 40, Math.max(...ys) - Math.min(...ys) + 2 * pad + 40];
    node.setAttribute("viewBox", vb.join(" "));
    const defs = svg("defs", {}, node);
    const hatch = svg("pattern", { id: "hatch-overview", patternUnits: "userSpaceOnUse", width: 9, height: 9, patternTransform: "rotate(45)" }, defs);
    svg("line", { x1: 0, y1: 0, x2: 0, y2: 9, class: "hatch-line" }, hatch);
    const c = TABLE.cell_mm;
    for (let i = 0; i < TABLE.cols; i++) {
      for (let j = 0; j < TABLE.rows; j++) {
        const label = String.fromCharCode(65 + i) + (j + 1);
        const x0 = Math.min(X(i * c), X((i + 1) * c)), y0 = Math.min(Y(j * c), Y((j + 1) * c));
        if (TABLE.held_out_cells.includes(label)) svg("rect", { x: x0, y: y0, width: c, height: c, fill: "url(#hatch-overview)" }, node);
        svg("rect", { x: x0, y: y0, width: c, height: c, class: "cell" }, node);
      }
    }
    svg("rect", { x: 0, y: 0, width: W, height: H, class: "outline" }, node);
    const gridLabel = svg("text", { x: W / 2, y: -12, "text-anchor": "middle", class: "tag-label" }, node);
    gridLabel.textContent = `Grid, ${W} \u00d7 ${H} mm`;
    for (const tag of D.tags) {
      const corners = tag.corners_mm.map((p) => [X(p[0]), Y(p[1])]);
      const cx = X(tag.center_mm[0]), cy = Y(tag.center_mm[1]);
      const scaled = (f) => corners.map(([x, y]) => [cx + (x - cx) * f, cy + (y - cy) * f]).map((p) => p.join(",")).join(" ");
      svg("polygon", { points: scaled(1), class: "tag-body" }, node);
      svg("polygon", { points: scaled(0.72), class: "tag-hole" }, node);
      svg("polygon", { points: scaled(0.36), class: "tag-body" }, node);
      const bottom = Math.max(...corners.map((p) => p[1]));
      const label = svg("text", { x: cx, y: bottom + 26, "text-anchor": "middle", class: "tag-label" }, node);
      label.textContent = "AprilTag " + tag.id;
    }
  }

  // ------------------------------------------------------------------------------------------------- behavior

  function redraw() {
    drawControls();
    drawMap();
    drawPanel();
    drawStrip();
    drawTable();
  }

  function select(id, options) {
    const t = id && BY_ID[id];
    if (!t) {
      state.selected = null;
      try { history.replaceState(null, "", location.pathname + location.search); } catch (e) { /* file URL */ }
      drawSelection(); drawPanel(); drawStripHover(); drawTable();
      return;
    }
    let full = false;
    if (!inSet(t)) { state.set = SETS.clustered.conditions.includes(t.condition) ? "clustered" : "spread"; full = true; }
    if (!state.show[t.condition]) { state.show[t.condition] = true; full = true; }
    if (!state.outcome[t.outcome]) { state.outcome[t.outcome] = true; full = true; }
    state.selected = t.id;
    try { history.replaceState(null, "", "#" + t.id); } catch (e) { /* file URL */ }
    if (full) redraw();
    else { drawSelection(); drawPanel(); drawStripHover(); drawTable(); }
    if (options && options.reveal) revealPanel();
  }

  // On a phone the panel sits under the map: bring it into view after a tap.
  function revealPanel() {
    if (!window.matchMedia("(max-width: 61.99rem)").matches) return;
    const box = $("panel").getBoundingClientRect();
    if (box.top > window.innerHeight * 0.8 || box.bottom < 0) {
      const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      $("panel").scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
    }
  }

  function step(by) {
    const list = visible().sort(byId);
    if (!list.length) return;
    const i = list.findIndex((t) => t.id === state.selected);
    const next = i < 0 ? (by > 0 ? 0 : list.length - 1) : (i + by + list.length) % list.length;
    select(list[next].id);
  }

  function wire() {
    $("sets").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-set]");
      if (!b || b.dataset.set === state.set) return;
      state.set = b.dataset.set;
      if (state.selected && !inSet(BY_ID[state.selected])) {
        state.selected = null;
        try { history.replaceState(null, "", location.pathname + location.search); } catch (err) { /* file URL */ }
      }
      redraw();
    });
    $("conditions").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-condition]");
      if (!b) return;
      const c = b.dataset.condition;
      const others = SETS[state.set].conditions.filter((x) => x !== c && state.show[x]);
      if (state.show[c] && !others.length) return;  // keep at least one condition on
      state.show[c] = !state.show[c];
      redraw();
    });
    $("outcomes").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-outcome]");
      if (!b) return;
      const o = b.dataset.outcome;
      const other = o === "success" ? "failure" : "success";
      if (state.outcome[o] && !state.outcome[other]) return;  // keep at least one outcome on
      state.outcome[o] = !state.outcome[o];
      redraw();
    });

    const map = $("map");
    map.addEventListener("pointermove", (e) => {
      const t = nearestOnMap(e);
      map.classList.toggle("over-trial", Boolean(t));
      if (e.pointerType === "touch") return;
      setHover(t, t ? markClientPoint(t) : null);
    });
    map.addEventListener("pointerleave", () => { map.classList.remove("over-trial"); setHover(null); });
    map.addEventListener("click", (e) => {
      const t = nearestOnMap(e);
      if (t) { setHover(null); select(t.id, { reveal: true }); }
    });
    map.addEventListener("keydown", (e) => {
      const go = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (go) { e.preventDefault(); step(go); }
      else if (e.key === "Escape") select(null);
    });

    $("strip").addEventListener("pointermove", (e) => {
      const lane = e.target.closest("svg.lane");
      if (!lane) return;
      const hit = nearestOnStrip({ currentTarget: lane, clientX: e.clientX, clientY: e.clientY, pointerType: e.pointerType });
      lane.classList.toggle("over-trial", Boolean(hit));
      if (e.pointerType === "touch") return;
      setHover(hit ? hit.t : null, hit ? hit.at : null);
    });
    $("strip").addEventListener("pointerleave", () => setHover(null));
    $("strip").addEventListener("click", (e) => {
      const lane = e.target.closest("svg.lane");
      if (!lane) return;
      const hit = nearestOnStrip({ currentTarget: lane, clientX: e.clientX, clientY: e.clientY, pointerType: e.pointerType || "mouse" });
      if (hit) { setHover(null); select(hit.t.id); }
    });

    document.addEventListener("click", (e) => {
      const go = e.target.closest("[data-go]");
      if (go) { select(go.dataset.go, { reveal: true }); return; }
      const link = e.target.closest("a[data-trial]");
      if (link) {
        e.preventDefault();
        select(link.dataset.trial);
        $("map-wrap").scrollIntoView({ behavior: "smooth", block: "center" });
        return;
      }
      const st = e.target.closest("button[data-step]");
      if (st) { step(Number(st.dataset.step)); return; }
      if (e.target.closest("button[data-clear]")) select(null);
    });

    window.addEventListener("scroll", () => { if (!$("tip").hidden) setHover(null); }, { passive: true });

    window.addEventListener("hashchange", () => {
      const id = decodeURIComponent(location.hash.slice(1));
      if (BY_ID[id] && id !== state.selected) select(id);
    });

    let pending = 0;
    const onResize = () => {
      cancelAnimationFrame(pending);
      pending = requestAnimationFrame(() => { setHover(null); drawMap(); drawStrip(); });
    };
    if ("ResizeObserver" in window) {
      let lastW = 0;
      new ResizeObserver((entries) => {
        const w = Math.round(entries[0].contentRect.width);
        if (w !== lastW) { lastW = w; onResize(); }
      }).observe($("explore"));
    } else {
      window.addEventListener("resize", onResize);
    }
  }

  function themeToggle() {
    const root = document.documentElement;
    const toggle = document.querySelector(".theme-toggle");
    if (!toggle) return;
    const label = () => {
      const light = root.getAttribute("data-theme") === "light";
      toggle.textContent = light ? "Dark mode" : "Light mode";
      toggle.setAttribute("aria-pressed", light ? "true" : "false");
      const meta = document.querySelector('meta[name="theme-color"]');
      if (meta) meta.setAttribute("content", light ? "#f4f6f8" : "#0e1318");
    };
    toggle.hidden = false;
    label();
    toggle.addEventListener("click", () => {
      const light = root.getAttribute("data-theme") !== "light";
      if (light) root.setAttribute("data-theme", "light");
      else root.removeAttribute("data-theme");
      try { localStorage.setItem("theme", light ? "light" : "dark"); } catch (e) { /* not saved */ }
      label();
    });
  }

  function fillText() {
    for (const node of document.querySelectorAll("[data-fill]")) {
      const key = node.getAttribute("data-fill");
      if (key === "cell-mm") node.textContent = String(TABLE.cell_mm);
      if (key === "n-trials") node.textContent = String(TRIALS.length);
      if (key === "tolerance-mm") node.textContent = String(TABLE.placement_tolerance_mm);
    }
  }

  // ---------------------------------------------------------------------------------------------- the resets

  const resetError = (t) => Math.hypot(t.x_mm - t.target_mm[0], t.y_mm - t.target_mm[1]);
  function median(values) {
    const v = values.slice().sort((a, b) => a - b);
    const m = Math.floor(v.length / 2);
    return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
  }

  // Every figure in the reset section, counted from the data file.
  function drawResets() {
    const trialErrors = TRIALS.map(resetError);
    const demoErrors = Object.values(DEMOS).flat().filter((d) => d.planned_mm)
      .map((d) => Math.hypot(d.x_mm - d.planned_mm[0], d.y_mm - d.planned_mm[1]));
    const eye = D.resets_by_eye_mm || [];
    const numbers = $("reset-numbers");
    numbers.textContent =
      `The ${trialErrors.length} trial resets were accepted a median ${fmt(median(trialErrors))}\u00a0mm from their targets, ` +
      `and none more than ${fmt(Math.max(...trialErrors))}\u00a0mm. ` +
      `The ${demoErrors.length} demonstration starts, measured in each episode's first frame, were a median ` +
      `${fmt(median(demoErrors))}\u00a0mm from theirs.` +
      (eye.length ? ` When I placed the cube ${eye.length} times by eye from a reference picture, with no overlay, the median miss was ${fmt(median(eye))}\u00a0mm.` : "");

    const band = (condition, lo, hi) => counts(TRIALS.filter((t) => t.condition === condition && t.d_mm > lo && t.d_mm <= hi));
    const cMid = band("act_clustered", 10, 20), rMid = band("replay", 10, 20);
    const cNear = band("act_clustered", -1, 10), rNear = band("replay", -1, 10);
    $("reset-band").textContent = eye.length
      ? `That is inside the 10 to 20\u00a0mm band, where the clustered policy succeeded in ${cMid.ok} of ${cMid.n} trials and the replay in ${rMid.ok} of ${rMid.n}. ` +
        `Within 10\u00a0mm they succeeded in ${cNear.ok} of ${cNear.n} and ${rNear.ok} of ${rNear.n}.`
      : "";
  }

  // The reset clip plays muted while it is on screen. Readers who ask for reduced motion get the controls instead.
  function autoplayClips() {
    const clips = document.querySelectorAll("video[data-autoplay]");
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    for (const v of clips) {
      v.muted = true;
      v.addEventListener("click", () => { v.controls = true; });
    }
    if (reduce || !("IntersectionObserver" in window)) {
      for (const v of clips) v.controls = true;
      return;
    }
    const seen = new IntersectionObserver((entries) => {
      for (const e of entries) {
        const v = e.target;
        if (e.isIntersecting) {
          const p = v.play();
          if (p && p.catch) p.catch(() => { v.controls = true; });
        } else {
          v.pause();
        }
      }
    }, { threshold: 0.4 });
    for (const v of clips) seen.observe(v);
  }

  function start(data) {
    D = data;
    TABLE = D.table;
    W = TABLE.cols * TABLE.cell_mm;
    H = TABLE.rows * TABLE.cell_mm;
    TURN = D.camera_view;
    TRIALS = D.trials.slice();
    BY_ID = Object.fromEntries(TRIALS.map((t) => [t.id, t]));
    DEMOS = D.demos;
    POINTS = Object.fromEntries(Object.entries(DEMOS).map(([name, list]) => [name, demoPoints(list)]));

    fillText();
    for (const id of ["controls", "explore", "distance", "resets", "where", "list"]) $(id).hidden = false;

    const key = $("key");
    const items = [
      ["demo", "Where a demonstration started"],
      ["ring", `Within ${RING_MM}\u00a0mm of a demonstration (shaded)`],
      ["held", "Held-out cell: no demonstration started there"],
      ["clip", "Trial with a video"],
    ];
    for (const [kind, text] of items) {
      const li = html("li", {}, key);
      if (kind === "held") {
        const s = svg("svg", { width: 16, height: 16, viewBox: "0 0 16 16", "aria-hidden": "true" }, li);
        const defs = svg("defs", {}, s);
        const p = svg("pattern", { id: "hatch-key", patternUnits: "userSpaceOnUse", width: 4, height: 4, patternTransform: "rotate(45)" }, defs);
        svg("line", { x1: 0, y1: 0, x2: 0, y2: 4, class: "hatch-line" }, p);
        svg("rect", { x: 0.5, y: 0.5, width: 15, height: 15, fill: "url(#hatch-key)", class: "cell" }, s);
      } else if (kind === "ring") {
        const s = svg("svg", { width: 16, height: 16, viewBox: "-8 -8 16 16", "aria-hidden": "true" }, li);
        svg("circle", { r: 7, class: "ring-swatch" }, s);
      } else {
        li.appendChild(iconSvg(kind, 16));
      }
      html("span", kind === "demo" ? { id: "key-demo" } : {}, li, text);
    }

    themeToggle();
    wire();
    redraw();
    drawOverview();
    drawResets();
    autoplayClips();
    const id = decodeURIComponent(location.hash.slice(1));
    if (BY_ID[id]) select(id);
  }

  loadData().then(start).catch((error) => {
    const box = $("load-error");
    box.hidden = false;
    box.textContent = "The trial data did not load (" + error.message + "). Keep data/trials.json and data/trials.js next to this page, or open it through a web server.";
  });
})();
