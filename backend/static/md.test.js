/* md.test.js · agent observatory
 *
 * Dependency-free harness for md.js. Feeds the renderer hostile and
 * ordinary chat-feed text and checks the invariants that matter:
 *   1. no input, however hostile, reaches the output as live markup
 *   2. the small supported syntax set renders what it promises
 *   3. fenced code protects its body from every other transform
 *   4. pretty() de-uglifies tool payloads without ever returning HTML
 *   5. garbage never throws and rendering is deterministic
 *
 * Run:  node backend/static/md.test.js
 */
"use strict";

const path = require("path");
const ObsMd = require(path.join(__dirname, "md.js"));

let checks = 0, failures = [];

function ok(cond, msg) {
  checks++;
  if (!cond) failures.push(msg);
}

function has(html, part, msg) {
  ok(html.indexOf(part) !== -1, msg + " · missing " + JSON.stringify(part));
}

function lacks(html, part, msg) {
  ok(html.indexOf(part) === -1, msg + " · contains " + JSON.stringify(part));
}

const sections = [];

function section(name, fn) {
  const c0 = checks, f0 = failures.length;
  fn();
  sections.push([name, String(checks - c0), failures.length === f0 ? "ok" : "FAIL"]);
}

/* ------------------------------------------------------------------ *
 * 1. xss · hostile input must come out escaped in every position
 * ------------------------------------------------------------------ */

section("xss", function () {
  const script = "<script>alert(1)</script>";
  const img = "<img src=x onerror=alert(1)>";

  let out = ObsMd.render(script);
  lacks(out, "<script", "plain script tag must be escaped");
  has(out, "&lt;script&gt;alert(1)&lt;/script&gt;", "plain script tag shows as text");

  out = ObsMd.render(img);
  lacks(out, "<img", "plain img tag must be escaped");
  has(out, "&lt;img src=x onerror=alert(1)&gt;", "plain img tag shows as text");

  out = ObsMd.render("**" + script + "**");
  lacks(out, "<script", "script inside bold must be escaped");
  has(out, "<strong>&lt;script&gt;alert(1)&lt;/script&gt;</strong>",
      "bold still applies around escaped text");

  out = ObsMd.render("```\n" + script + "\n" + img + "\n```");
  lacks(out, "<script", "script inside a fence must be escaped");
  lacks(out, "<img", "img inside a fence must be escaped");
  has(out, "&lt;script&gt;alert(1)&lt;/script&gt;", "fence keeps the escaped script text");

  out = ObsMd.render("[" + img + "](http://example.com)");
  lacks(out, "<img", "img inside link text must be escaped");
  has(out, '<a href="http://example.com" target="_blank" rel="noopener">',
      "link renders around the escaped label");
  has(out, "&lt;img", "link label keeps the escaped text");

  out = ObsMd.render("[click](javascript:alert(1))");
  lacks(out, 'href="javascript', "javascript: href must not become a link");
  lacks(out, "<a", "javascript: link renders no anchor at all");
  has(out, "[click](javascript:alert(1))", "javascript: link stays literal text");

  out = ObsMd.render("[x](data:text/html,<script>alert(1)</script>)");
  lacks(out, 'href="data', "data: href must not become a link");
  lacks(out, "<script", "data: payload must stay escaped");
});

/* ------------------------------------------------------------------ *
 * 2. inline styling · bold, italic, inline code
 * ------------------------------------------------------------------ */

section("inline styling", function () {
  has(ObsMd.render("**bold**"), "<strong>bold</strong>", "double star bold");
  has(ObsMd.render("*it*"), "<em>it</em>", "star italic");
  has(ObsMd.render("_it_"), "<em>it</em>", "underscore italic");
  has(ObsMd.render("mix **b** and *i*"), "<strong>b</strong>", "bold in a sentence");
  has(ObsMd.render("mix **b** and *i*"), "<em>i</em>", "italic in a sentence");
  has(ObsMd.render("`x = 1`"), "<code>x = 1</code>", "inline code");
  has(ObsMd.render("`**not bold**`"), "<code>**not bold**</code>",
      "inline code protects its body");
  lacks(ObsMd.render("a **b\nc** d"), "<strong>", "bold must not cross a newline");
  has(ObsMd.render("snake_case_name stays"), "snake_case_name",
      "mid word underscores stay literal");
});

/* ------------------------------------------------------------------ *
 * 3. headers
 * ------------------------------------------------------------------ */

section("headers", function () {
  has(ObsMd.render("# Title"), '<h1 class="md-h">Title</h1>', "h1");
  has(ObsMd.render("## Two"), '<h2 class="md-h">Two</h2>', "h2");
  has(ObsMd.render("### Three"), '<h3 class="md-h">Three</h3>', "h3");
  has(ObsMd.render("#### Four"), '<h4 class="md-h">Four</h4>', "h4");
  lacks(ObsMd.render("##### Five"), "<h5", "five hashes is not a header level");
  lacks(ObsMd.render("##### Five"), "<h4", "five hashes does not fall back to h4");
  lacks(ObsMd.render("#nospace"), "<h1", "hash without a space is not a header");
});

/* ------------------------------------------------------------------ *
 * 4. fenced code · body is verbatim, protected from every transform
 * ------------------------------------------------------------------ */

section("fenced code", function () {
  const out = ObsMd.render("```python\nvalue = **not bold**\nrow | cell\n```");
  has(out, '<pre class="md-code">', "fence renders a pre block");
  has(out, "value = **not bold**", "a ** inside a fence stays literal");
  lacks(out, "<strong", "no bold inside a fence");
  lacks(out, "python", "language tag is dropped");
  lacks(out, "<table", "no table inside a fence");

  const two = ObsMd.render("before\n```\ncode line\n```\nafter");
  has(two, "<p>before</p>", "text before a fence stays a paragraph");
  has(two, "<p>after</p>", "text after a fence stays a paragraph");
  has(two, '<pre class="md-code">code line</pre>', "fence body is verbatim");
});

/* ------------------------------------------------------------------ *
 * 5. tables
 * ------------------------------------------------------------------ */

section("tables", function () {
  const out = ObsMd.render(
    "| name | qty |\n| --- | --- |\n| apple | 3 |\n| pear | 12 |");
  has(out, '<table class="md-table">', "table tag with house class");
  has(out, "<th>name</th>", "first header cell");
  has(out, "<th>qty</th>", "second header cell");
  has(out, "<td>apple</td>", "body cell text");
  has(out, "<td>12</td>", "numeric body cell");
  has(out, "<thead>", "thead present");
  has(out, "<tbody>", "tbody present");
  ok((out.match(/<tr>/g) || []).length === 3, "one header row plus two body rows");

  const styled = ObsMd.render("| h |\n|---|\n| **b** |");
  has(styled, "<td><strong>b</strong></td>", "inline styling works inside cells");
});

/* ------------------------------------------------------------------ *
 * 6. lists · flat, ordered and unordered
 * ------------------------------------------------------------------ */

section("lists", function () {
  const ul = ObsMd.render("- alpha\n- beta\n* gamma");
  has(ul, "<ul>", "unordered list opens");
  has(ul, "<li>alpha</li>", "dash item");
  has(ul, "<li>gamma</li>", "star item joins the same list");
  ok((ul.match(/<ul>/g) || []).length === 1, "adjacent items form one list");

  const ol = ObsMd.render("1. one\n2. two\n3. three");
  has(ol, "<ol>", "ordered list opens");
  has(ol, "<li>one</li>", "first ordered item");
  has(ol, "<li>three</li>", "last ordered item");
  lacks(ol, "<ul>", "ordered items never render unordered");
});

/* ------------------------------------------------------------------ *
 * 7. paragraphs, rules, quotes
 * ------------------------------------------------------------------ */

section("paragraphs", function () {
  const soft = ObsMd.render("line one\nline two");
  has(soft, "line one<br>line two", "single newline becomes <br>");
  ok((soft.match(/<p>/g) || []).length === 1, "soft break stays one paragraph");

  const hard = ObsMd.render("para one\n\npara two");
  has(hard, "<p>para one</p>", "first paragraph closed");
  has(hard, "<p>para two</p>", "second paragraph opened");
  lacks(hard, "<br>", "blank line is a paragraph break, not <br>");

  has(ObsMd.render("---"), "<hr>", "horizontal rule");
  has(ObsMd.render("> wise words"), "<blockquote>wise words</blockquote>", "blockquote");
});

/* ------------------------------------------------------------------ *
 * 8. links · happy path plus scheme allowlist
 * ------------------------------------------------------------------ */

section("links", function () {
  const out = ObsMd.render("see [the docs](https://example.com/a_b) now");
  has(out, '<a href="https://example.com/a_b" target="_blank" rel="noopener">the docs</a>',
      "https link renders");
  lacks(out, "<em>", "underscores inside an href never italicize");
  has(ObsMd.render("[plain](http://example.org)"), 'href="http://example.org"',
      "http link renders");
  has(ObsMd.render("[ftp](ftp://example.org/x)"), "[ftp](ftp://example.org/x)",
      "non http scheme stays literal");
});

/* ------------------------------------------------------------------ *
 * 9. pretty · plain text de-uglifier for tool payloads
 * ------------------------------------------------------------------ */

section("pretty", function () {
  const obj = '{"tool":"search","args":{"q":"cats"},"n":2}';
  ok(ObsMd.pretty(obj) === JSON.stringify(JSON.parse(obj), null, 2),
     "json object gets 2 space indentation");
  ok(ObsMd.pretty("[1,2]") === "[\n  1,\n  2\n]", "json array gets indented");
  ok(ObsMd.pretty("'line1\\nline2'") === "line1\nline2",
     "python style repr gets real newlines and loses its quotes");
  ok(ObsMd.pretty("a\\tb") === "a\tb", "literal tab sequence becomes a tab");
  ok(ObsMd.pretty("ordinary text passes") === "ordinary text passes",
     "plain text is untouched");
  ok(ObsMd.pretty('"just quoted"') === "just quoted",
     "one layer of wrapping quotes is stripped");
  ok(ObsMd.pretty("") === "", "empty string stays empty");
});

/* ------------------------------------------------------------------ *
 * 10. garbage · never throws, always returns a string, deterministic
 * ------------------------------------------------------------------ */

section("garbage", function () {
  const nasty = [
    "```\nunterminated fence",
    "lone **",
    "***",
    "[[[nested]](((",
    "[a](b](c)",
    "| a | b |\n|---|",
    "||||",
    "> ",
    "-",
    "1.",
    "\u0000\u0000weird control\u0000",
    "",
  ];
  nasty.forEach(function (input) {
    let out = null, threw = false;
    try { out = ObsMd.render(input); } catch (err) { threw = true; }
    ok(!threw, "render threw on " + JSON.stringify(input));
    ok(typeof out === "string", "render returned a non string for " + JSON.stringify(input));
  });

  const fence = ObsMd.render("```\nunterminated <script>alert(1)</script>");
  has(fence, "&lt;script&gt;", "unterminated fence still escapes its body");
  lacks(fence, "<script", "unterminated fence never leaks markup");

  ok(ObsMd.render(null) === "", "null renders as empty string");
  ok(ObsMd.render(undefined) === "", "undefined renders as empty string");
  ok(ObsMd.render(42) === "<p>42</p>", "numbers coerce to text");

  const doc = "# T\n\n**b** *i* `c`\n\n- x\n- y\n\n| a |\n|---|\n| b |\n\n" +
              "> q\n\n[l](https://e.com)\n\n```\nz\n```";
  ok(ObsMd.render(doc) === ObsMd.render(doc), "rendering is deterministic");
});

/* ------------------------------------------------------------------ *
 * run
 * ------------------------------------------------------------------ */

const head = ["section", "checks", ""];
const widths = head.map((h, i) =>
  Math.max(h.length, ...sections.map((r) => r[i].length)));
const line = (cells) => cells.map((c, i) => c.padEnd(widths[i])).join("  ");

console.log(line(head));
console.log(widths.map((w) => "-".repeat(w)).join("  "));
sections.forEach((r) => console.log(line(r)));
console.log("");

if (failures.length) {
  console.log(failures.length + " failure(s) of " + checks + " checks:");
  failures.slice(0, 20).forEach((f) => console.log("  - " + f));
  process.exit(1);
}
console.log("all " + checks + " checks passed across " + sections.length + " sections");
