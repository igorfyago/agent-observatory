/* md.js · agent observatory
 *
 * Dependency-free mini markdown renderer for chat-feed text from LLMs.
 * Model output (gpt-4o-mini style) is markdown-ish and untrusted, so the
 * contract is safety first, fidelity second.
 *
 * Safety model:
 *   1. every character of input is HTML-escaped (& < > ") before any
 *      transform runs, so input text can never reach the page as markup
 *   2. transforms only insert tags this file writes itself, there is no
 *      raw HTML passthrough of any kind
 *   3. link hrefs are allowlisted to http:// and https://, anything else
 *      stays literal text
 *   4. malformed input never throws: on any parse failure the caller gets
 *      escaped text with <br> for newlines
 *
 * Supported syntax, kept deliberately small: fenced code blocks, inline
 * code, # to #### headers, bold, italic, flat unordered and ordered
 * lists, github tables, links, horizontal rules, blockquotes, and
 * blank-line paragraphs where a single newline becomes <br>.
 *
 * Usage (browser):
 *   <script src="/static/md.js"></script>
 *   el.innerHTML = ObsMd.render(text);
 *   pre.textContent = ObsMd.pretty(payload);  // plain text, caller escapes
 *
 * Usage (node, for the test harness):
 *   const ObsMd = require("./md.js");
 *
 * Deterministic and stateless: the same input always yields the same output.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.ObsMd = api;
})(typeof self !== "undefined" ? self : globalThis, function () {
  "use strict";

  /* ------------------------------------------------------------------ *
   * escaping · runs before every other transform, no exceptions
   * ------------------------------------------------------------------ */

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Last-resort output: escaped text with newlines kept visible.
  function fallback(text) {
    return escapeHtml(String(text).replace(/\r\n?/g, "\n")).replace(/\n/g, "<br>");
  }

  /* ------------------------------------------------------------------ *
   * inline transforms · operate on one already-escaped line at a time
   * ------------------------------------------------------------------ */

  function emphasis(tag) {
    return function (match, body) {
      // A body that starts or ends with whitespace stays literal:
      // "3 * 4 * 5" is arithmetic, not emphasis.
      if (/^\s|\s$/.test(body)) return match;
      return "<" + tag + ">" + body + "</" + tag + ">";
    };
  }

  // Emphasis only. Bold runs before italic so a ** pair is never read as
  // two stray * markers. Content classes exclude newlines, and exclude
  // < and > so a match can never straddle a tag inserted a step earlier.
  // The _italic_ form is anchored on word boundaries, which keeps the
  // underscores in snake_case_names literal.
  function styleText(s) {
    s = s.replace(/\*\*([^\n]+?)\*\*/g, emphasis("strong"));
    s = s.replace(/\*([^*\n<>]+?)\*/g, emphasis("em"));
    s = s.replace(/\b_([^_\n<>]+?)_\b/g, emphasis("em"));
    return s;
  }

  // One escaped line in, inline HTML out. Code spans and links are lifted
  // into slots first so nothing inside them is styled, then emphasis runs
  // on what remains, then the slots are put back. The slot delimiter is
  // NUL, which render() strips from the input up front, so a collision
  // with real text is impossible.
  function inline(s) {
    const slots = [];
    function stash(html) {
      slots.push(html);
      return "\u0000" + (slots.length - 1) + "\u0000";
    }

    s = s.replace(/`([^`\n]+)`/g, function (m, body) {
      return stash("<code>" + body + "</code>");
    });

    // Only http and https survive as anchors, anything else stays the
    // literal text it arrived as. The href is already escaped, so a quote
    // inside it cannot break out of the attribute.
    s = s.replace(/\[([^\]\n]*)\]\(([^()\s]+)\)/g, function (m, label, href) {
      if (!/^https?:\/\//i.test(href)) return m;
      return stash('<a href="' + href + '" target="_blank" rel="noopener">' +
                   styleText(label) + "</a>");
    });

    s = styleText(s);

    // A link label may itself hold a code slot, so resolve repeatedly,
    // bounded by the slot count so the loop always halts.
    let guard = slots.length + 1;
    while (guard-- > 0 && s.indexOf("\u0000") !== -1) {
      s = s.replace(/\u0000(\d+)\u0000/g, function (m, idx) {
        const slot = slots[+idx];
        return slot === undefined ? "" : slot;
      });
    }
    return s;
  }

  /* ------------------------------------------------------------------ *
   * block structure · line-oriented single pass
   * ------------------------------------------------------------------ */

  const RE_FENCE_OPEN = /^\s*```/;
  const RE_FENCE_CLOSE = /^\s*```\s*$/;
  const RE_HEADER = /^(#{1,4})(?!#)\s+(.+)$/;
  const RE_HR = /^-{3,}\s*$/;
  const RE_QUOTE = /^&gt;\s?(.*)$/;   // > was escaped before parsing
  const RE_UL = /^\s*[-*]\s+(.+)$/;   // nesting is flattened on purpose
  const RE_OL = /^\s*\d+\.\s+(.+)$/;

  function isTableSep(line) {
    return line.indexOf("|") !== -1 && line.indexOf("-") !== -1 &&
           /^[\s|:-]+$/.test(line);
  }

  function splitRow(line) {
    let s = line.trim();
    if (s.charAt(0) === "|") s = s.slice(1);
    if (s.charAt(s.length - 1) === "|") s = s.slice(0, -1);
    return s.split("|").map(function (c) { return c.trim(); });
  }

  function tableMarkup(head, body) {
    const th = head.map(function (c) {
      return "<th>" + inline(c) + "</th>";
    }).join("");
    const rows = body.map(function (r) {
      const td = r.map(function (c) { return "<td>" + inline(c) + "</td>"; }).join("");
      return "<tr>" + td + "</tr>";
    }).join("");
    return '<table class="md-table"><thead><tr>' + th + "</tr></thead>" +
           "<tbody>" + rows + "</tbody></table>";
  }

  function renderBlocks(src) {
    // NUL is the slot delimiter inside inline() and cannot occur in real
    // text, so strip it up front, then escape everything before parsing.
    const lines = escapeHtml(src.replace(/\r\n?/g, "\n").replace(/\u0000/g, ""))
      .split("\n");

    const out = [];
    const para = [];
    function flushPara() {
      if (!para.length) return;
      out.push("<p>" + para.map(inline).join("<br>") + "</p>");
      para.length = 0;
    }

    let i = 0;
    while (i < lines.length) {
      const line = lines[i];

      // Fenced code first: the body is protected from every other
      // transform. An unterminated fence swallows to end of input.
      if (RE_FENCE_OPEN.test(line)) {
        flushPara();
        const buf = [];
        i++;
        while (i < lines.length && !RE_FENCE_CLOSE.test(lines[i])) {
          buf.push(lines[i]);
          i++;
        }
        i++;   // past the closing fence, or past EOF when there is none
        out.push('<pre class="md-code">' + buf.join("\n") + "</pre>");
        continue;
      }

      // Blank line: paragraph boundary.
      if (!line.trim()) { flushPara(); i++; continue; }

      const h = RE_HEADER.exec(line);
      if (h) {
        flushPara();
        const level = h[1].length;
        out.push("<h" + level + ' class="md-h">' + inline(h[2].trim()) +
                 "</h" + level + ">");
        i++;
        continue;
      }

      if (RE_HR.test(line)) { flushPara(); out.push("<hr>"); i++; continue; }

      // Table: a pipe row is only a table when the next line is a
      // separator row, otherwise it is an ordinary paragraph line.
      if (line.indexOf("|") !== -1 && i + 1 < lines.length &&
          isTableSep(lines[i + 1])) {
        flushPara();
        const head = splitRow(line);
        i += 2;
        const body = [];
        while (i < lines.length && lines[i].trim() &&
               lines[i].indexOf("|") !== -1 && !RE_FENCE_OPEN.test(lines[i])) {
          body.push(splitRow(lines[i]));
          i++;
        }
        out.push(tableMarkup(head, body));
        continue;
      }

      const q = RE_QUOTE.exec(line);
      if (q) {
        flushPara();
        const buf = [inline(q[1])];
        i++;
        let qm;
        while (i < lines.length && (qm = RE_QUOTE.exec(lines[i]))) {
          buf.push(inline(qm[1]));
          i++;
        }
        out.push("<blockquote>" + buf.join("<br>") + "</blockquote>");
        continue;
      }

      if (RE_UL.test(line)) {
        flushPara();
        const items = [];
        let um;
        while (i < lines.length && (um = RE_UL.exec(lines[i]))) {
          items.push("<li>" + inline(um[1].trim()) + "</li>");
          i++;
        }
        out.push("<ul>" + items.join("") + "</ul>");
        continue;
      }

      if (RE_OL.test(line)) {
        flushPara();
        const items = [];
        let om;
        while (i < lines.length && (om = RE_OL.exec(lines[i]))) {
          items.push("<li>" + inline(om[1].trim()) + "</li>");
          i++;
        }
        out.push("<ol>" + items.join("") + "</ol>");
        continue;
      }

      para.push(line);
      i++;
    }
    flushPara();
    return out.join("\n");
  }

  /* ------------------------------------------------------------------ *
   * public: render · markdown-ish text in, safe HTML out
   * ------------------------------------------------------------------ */

  function render(text) {
    if (text === null || text === undefined) return "";
    const src = String(text);
    try {
      return renderBlocks(src);
    } catch (err) {
      return fallback(src);
    }
  }

  /* ------------------------------------------------------------------ *
   * public: pretty · plain text out, never HTML, the caller escapes it
   * ------------------------------------------------------------------ */

  // Best-effort de-uglifier for tool payloads. JSON objects and arrays
  // come back reprinted with 2-space indentation. A bare JSON string
  // comes back decoded, which strips its quotes and turns \n escapes
  // into real newlines. Anything else gets the cheap fixes: one layer of
  // wrapping quotes removed, literal \n and \t turned into real
  // characters.
  function pretty(text) {
    if (text === null || text === undefined) return "";
    const s = String(text);
    try {
      const trimmed = s.trim();
      if (trimmed) {
        try {
          const parsed = JSON.parse(trimmed);
          if (parsed !== null && typeof parsed === "object") {
            return JSON.stringify(parsed, null, 2);
          }
          if (typeof parsed === "string") return parsed;
        } catch (ignored) {
          // not JSON, fall through to the plain-text fixes
        }
      }
      let out = s;
      const first = out.charAt(0);
      const last = out.charAt(out.length - 1);
      if (out.length >= 2 && first === last && (first === '"' || first === "'")) {
        out = out.slice(1, -1);
      }
      return out.replace(/\\n/g, "\n").replace(/\\t/g, "\t");
    } catch (err) {
      return s;
    }
  }

  return {
    render: render,
    pretty: pretty,
  };
});
