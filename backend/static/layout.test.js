/* layout.test.js · agent observatory
 *
 * Dependency-free harness for layout.js. Renders four graph shapes to SVG
 * strings and checks the invariants that matter for a live agent DAG:
 *   1. no two node boxes overlap
 *   2. every edge endpoint lands on the node it claims to touch
 *   3. layout is deterministic: same input, byte-identical SVG
 *   4. forward edges always run down the flow, back edges run against it
 *
 * Run:  node backend/static/layout.test.js
 */
"use strict";

const path = require("path");
const ObsLayout = require(path.join(__dirname, "layout.js"));

let checks = 0, failures = [];

function ok(cond, msg) {
  checks++;
  if (!cond) failures.push(msg);
}

/* ------------------------------------------------------------------ *
 * fixtures
 * ------------------------------------------------------------------ */

const linear = {
  nodes: ["intake", "plan", "act", "report"],
  edges: [
    { s: "intake", t: "plan" },
    { s: "plan", t: "act" },
    { s: "act", t: "report" },
  ],
};

// the real observatory pipeline, fan-out then join
const branching = {
  nodes: [
    { id: "supervisor", label: "Supervisor", role: "plans & delegates" },
    { id: "researcher", label: "Researcher", role: "gathers facts" },
    { id: "analyst", label: "Analyst", role: "weighs trade-offs" },
    { id: "writer", label: "Writer", role: "drafts answer" },
    { id: "critic", label: "Critic", role: "reviews & gates" },
  ],
  edges: [
    { s: "supervisor", t: "researcher" },
    { s: "supervisor", t: "analyst" },
    { s: "researcher", t: "writer" },
    { s: "analyst", t: "writer" },
    { s: "writer", t: "critic" },
    { s: "critic", t: "writer", cond: true, label: "revise" },
    { s: "critic", t: "end", cond: true, label: "approve" },
  ],
};

// a cycle plus a self loop, the LangGraph tool-calling shape
const loopy = {
  nodes: ["agent", "tools", "reflect", "finish"],
  edges: [
    { s: "agent", t: "tools" },
    { s: "tools", t: "reflect" },
    { s: "reflect", t: "agent", cond: true, label: "retry" },
    { s: "reflect", t: "finish", cond: true, label: "done" },
    { s: "tools", t: "tools", label: "retry tool" },
  ],
};

// 20-node fan out then back in
const fan = (() => {
  const nodes = ["root"], edges = [];
  for (let i = 1; i <= 20; i++) {
    nodes.push("w" + i);
    edges.push({ s: "root", t: "w" + i });
    edges.push({ s: "w" + i, t: "sink" });
  }
  nodes.push("sink");
  return { nodes: nodes, edges: edges };
})();

const single = { nodes: ["solo"], edges: [] };

const SHAPES = [
  ["linear", linear, {}],
  ["branching", branching, {}],
  ["loop", loopy, {}],
  ["fan-20", fan, {}],
  ["single", single, {}],
  ["branching-LR", branching, { direction: "LR" }],
];

/* ------------------------------------------------------------------ *
 * assertions
 * ------------------------------------------------------------------ */

function overlaps(a, b) {
  const pad = 1; // touching edges is fine, crossing is not
  return a.x + a.w - pad > b.x && b.x + b.w - pad > a.x &&
         a.y + a.h - pad > b.y && b.y + b.h - pad > a.y;
}

function pathEndpoints(d) {
  const nums = d.match(/-?\d+(?:\.\d+)?/g).map(Number);
  return [
    [nums[0], nums[1]],
    [nums[nums.length - 2], nums[nums.length - 1]],
  ];
}

function onNode(pt, n, tol) {
  return pt[0] >= n.x - tol && pt[0] <= n.x + n.w + tol &&
         pt[1] >= n.y - tol && pt[1] <= n.y + n.h + tol;
}

function checkShape(name, spec, opts) {
  const lay = ObsLayout.layout(spec, opts);
  const svg = ObsLayout.renderSvg(spec, opts);
  const byId = new Map(lay.nodes.map((n) => [n.id, n]));
  const horiz = lay.direction === "LR";

  // every declared node is placed
  const declared = new Set(spec.nodes.map((n) => (typeof n === "string" ? n : n.id)));
  declared.forEach((id) => ok(byId.has(id), name + ": node " + id + " was not placed"));

  // 1. no two boxes overlap
  const clashes = [];
  for (let i = 0; i < lay.nodes.length; i++) {
    for (let j = i + 1; j < lay.nodes.length; j++) {
      if (overlaps(lay.nodes[i], lay.nodes[j])) {
        clashes.push(lay.nodes[i].id + "/" + lay.nodes[j].id);
      }
    }
  }
  ok(clashes.length === 0,
     name + ": " + clashes.length + " overlapping box pair(s) · " +
     clashes.slice(0, 3).join(", "));

  // nodes stay inside the canvas
  lay.nodes.forEach((n) => {
    ok(n.x >= 0 && n.y >= 0 && n.x + n.w <= lay.width + 0.01 &&
       n.y + n.h <= lay.height + 0.01,
       name + ": node " + n.id + " falls outside the canvas");
  });

  // 2. every edge endpoint lands on its node · read back out of the SVG string
  const re = /<path class="edge[^"]*" id="e-([^"]+)" d="([^"]+)"/g;
  const rendered = [];
  let m;
  while ((m = re.exec(svg)) !== null) rendered.push({ id: m[1], d: m[2] });
  ok(rendered.length === lay.edges.length,
     name + ": rendered " + rendered.length + " edge paths, expected " + lay.edges.length);

  rendered.forEach(({ id, d }) => {
    const edge = lay.edges.find((e) => e.s + "-" + e.t === id);
    ok(!!edge, name + ": rendered edge " + id + " has no layout entry");
    if (!edge) return;
    const s = byId.get(edge.s), t = byId.get(edge.t);
    const [p0, p1] = pathEndpoints(d);
    ok(onNode(p0, s, 1.5),
       name + ": edge " + id + " starts at " + p0 + ", off node " + edge.s);
    ok(onNode(p1, t, 1.5),
       name + ": edge " + id + " ends at " + p1 + ", off node " + edge.t);
  });

  // 4. flow direction: forward edges advance a layer, back edges do not
  lay.edges.forEach((e) => {
    const s = byId.get(e.s), t = byId.get(e.t);
    if (e.kind === "forward") {
      ok(t.layer > s.layer,
         name + ": forward edge " + e.s + "->" + e.t + " does not advance a layer");
      const adv = horiz ? t.x - (s.x + s.w) : t.y - (s.y + s.h);
      ok(adv > 0, name + ": forward edge " + e.s + "->" + e.t + " runs backwards");
    } else {
      ok(t.layer <= s.layer,
         name + ": back edge " + e.s + "->" + e.t + " is not a back edge");
    }
  });

  // 3. determinism
  ok(ObsLayout.renderSvg(spec, opts) === svg, name + ": render is not deterministic");
  ok(JSON.stringify(ObsLayout.layout(spec, opts)) === JSON.stringify(lay),
     name + ": layout is not deterministic");
  ok(!/NaN|Infinity|undefined/.test(svg), name + ": svg contains a non-finite number");

  return { lay: lay, svg: svg };
}

/* ------------------------------------------------------------------ *
 * run
 * ------------------------------------------------------------------ */

const rows = [];
SHAPES.forEach(([name, spec, opts]) => {
  const before = failures.length;
  const { lay, svg } = checkShape(name, spec, opts);
  const backs = lay.edges.filter((e) => e.kind !== "forward").length;
  rows.push([
    name,
    String(lay.nodes.length),
    String(lay.edges.length),
    String(lay.layers.length),
    String(backs),
    lay.width + "x" + lay.height,
    String(svg.length) + "b",
    failures.length === before ? "ok" : "FAIL",
  ]);
});

const head = ["shape", "nodes", "edges", "layers", "arcs", "viewBox", "svg", ""];
const widths = head.map((h, i) =>
  Math.max(h.length, ...rows.map((r) => r[i].length)));
const line = (cells) => cells.map((c, i) => c.padEnd(widths[i])).join("  ");

console.log(line(head));
console.log(widths.map((w) => "-".repeat(w)).join("  "));
rows.forEach((r) => console.log(line(r)));
console.log("");

if (failures.length) {
  console.log(failures.length + " failure(s) of " + checks + " checks:");
  failures.slice(0, 20).forEach((f) => console.log("  - " + f));
  process.exit(1);
}
console.log("all " + checks + " checks passed across " + SHAPES.length + " shapes");
