// The "what is the network thinking" panel: where the message was routed, the top guesses, the words that mattered.
const prettyTag = (t) => (t.startsWith("skill:") ? t.slice(6) : t).replaceAll("_", " ");

export function renderBrain(r, { route, bars, words, meter }) {
  if (r.kind !== "chat") return;
  const skill = r.tag.startsWith("skill:") || r.tag === "crisis";
  const warn = ["fallback", "clarify", "clarify_no", "crash"].includes(r.tag);

  route.replaceChildren();
  const pill = document.createElement("span");
  pill.className = "pill" + (skill ? " skill" : warn ? " warn" : "");
  pill.textContent = r.tag === "crisis" ? "safety net" : prettyTag(r.tag);
  const small = document.createElement("small");
  small.textContent = r.tag === "crisis" ? "keyword check runs before the network" : skill ? "exact code, not the network" : warn ? "not sure, so it says so" : `${Math.round(r.conf * 100)}% sure`;
  route.append(pill, small);

  bars.replaceChildren();
  for (const [name, p] of r.top || []) {
    const li = document.createElement("li");
    const nm = document.createElement("span"), pc = document.createElement("span"), bar = document.createElement("span"), fill = document.createElement("b");
    nm.className = "nm"; nm.textContent = prettyTag(name);
    pc.className = "pc"; pc.textContent = Math.round(p * 100) + "%";
    bar.className = "bar"; bar.append(fill);
    li.append(nm, pc, bar);
    bars.append(li);
    requestAnimationFrame(() => requestAnimationFrame(() => fill.style.setProperty("transform", `scaleX(${Math.max(0.02, p)})`)));
  }

  words.replaceChildren();
  const ws = r.words || [];
  const max = Math.max(0.001, ...ws.map(([, d]) => Math.abs(d)));
  if (!ws.length) {
    const idle = document.createElement("span");
    idle.className = "idle";
    idle.textContent = "no words to weigh";
    words.append(idle);
  }
  for (const [w, d] of ws) {
    const el = document.createElement("span");
    el.className = "w" + (d < 0 ? " neg" : "");
    el.style.setProperty("--k", Math.min(1, Math.abs(d) / max).toFixed(2));
    el.append(w);
    const i = document.createElement("i");
    i.textContent = (d > 0 ? "+" : "") + d.toFixed(2);
    el.append(i);
    words.append(el);
  }

  meter.replaceChildren("network pass: ");
  const b = document.createElement("b");
  b.textContent = (r.us || 0) + " µs";
  meter.append(b, " in WebAssembly");
}
