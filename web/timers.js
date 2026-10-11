// Floating timer window. The engine reports its timers as [deadline_unix_ms, label, total_ms] on every reply;
// this module only draws them and counts down locally between replies.
const NS = "http://www.w3.org/2000/svg";
const R = 22, C = 2 * Math.PI * R;
const el = (tag, cls, text) => Object.assign(document.createElement(tag), { className: cls || "", textContent: text || "" });
const fmt = (ms) => {
  const s = Math.max(0, Math.ceil(ms / 1000)), h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, p = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${p(m)}:${p(s % 60)}` : `${p(m)}:${p(s % 60)}`;
};

export function initTimers() {
  const dock = el("aside", "dock");
  dock.hidden = true;
  dock.setAttribute("aria-label", "Timers");
  document.body.append(dock);
  let rows = [];

  function build(list) {
    dock.replaceChildren();
    rows = list.map(([due, label, total]) => {
      const row = el("div", "trow"), svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", "0 0 56 56");
      svg.setAttribute("class", "ring");
      const mk = (cls) => {
        const c = document.createElementNS(NS, "circle");
        c.setAttribute("class", cls); c.setAttribute("cx", 28); c.setAttribute("cy", 28); c.setAttribute("r", R);
        svg.append(c);
        return c;
      };
      mk("track");
      const arc = mk("arc");
      arc.setAttribute("stroke-dasharray", C.toFixed(1));
      const body = el("div", "tbody"), time = el("div", "ttime"), name = el("div", "tlabel", label || "Timer");
      body.append(time, name);
      row.append(svg, body);
      dock.append(row);
      return { due, total: Math.max(1000, total), arc, time, row, born: 0 };
    });
    paint();
  }

  function paint() {
    const now = Date.now();
    rows = rows.filter((r) => {
      const left = r.due - now;
      if (left < -6000) { r.row.remove(); return false; }
      const done = left <= 0;
      r.row.classList.toggle("done", done);
      r.time.textContent = done ? "Done" : fmt(left);
      r.arc.setAttribute("stroke-dashoffset", (C * (1 - Math.max(0, Math.min(1, left / r.total)))).toFixed(1));
      return true;
    });
    dock.hidden = rows.length === 0;
  }
  setInterval(paint, 250);

  return { sync: (list) => build(Array.isArray(list) ? list : []) };
}
