import { loadBot } from "./tinybot.js";

const $ = (id) => document.getElementById(id);
const log = $("log"), form = $("form"), input = $("input"), send = $("send"), status = $("status"), chips = $("chips");
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------- theme ---------- */
const root = document.documentElement;
try {
  const t = localStorage.getItem("tinybot.theme");
  if (t === "light" || t === "dark") root.dataset.theme = t;
} catch {}
$("theme").addEventListener("click", () => {
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : !matchMedia("(prefers-color-scheme: light)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("tinybot.theme", root.dataset.theme); } catch {}
});

/* ---------- the face ---------- */
// stroke (.st) and fill (.fl) shapes; the eyes blink via CSS
const dotEyes = (ry = 9, dy = 0) =>
  `<ellipse class="eye fl" cx="68" cy="${70 + dy}" rx="9" ry="${ry}"/><ellipse class="eye r fl" cx="132" cy="${70 + dy}" rx="9" ry="${ry}"/>`;
const arcEyes = `<path class="st" d="M54 76q14-22 28 0"/><path class="st" d="M118 76q14-22 28 0"/>`;
const heart = (x) => `<path class="fl" transform="translate(${x} 66)" d="M0 -4c-4-9-18-4-12 6l12 12 12-12c6-10-8-15-12-6z"/>`;
const EYES = {
  neutral: dotEyes(),
  joy: arcEyes, ecstatic: arcEyes, laugh: `<path class="st" d="M54 70l24 8-24 8"/><path class="st" d="M146 70l-24 8 24 8"/>`, proud: arcEyes,
  love: heart(68) + heart(132),
  anger: dotEyes(8, 4) + `<path class="st" d="M50 50l34 12"/><path class="st" d="M150 50l-34 12"/>`,
  grumpy: dotEyes(6, 4) + `<path class="st" d="M50 52l34 8"/><path class="st" d="M150 52l-34 8"/>`,
  sad: dotEyes(9, 4) + `<path class="st" d="M52 56l30-8"/><path class="st" d="M148 56l-30-8"/>`,
  concern: dotEyes(9, 4) + `<path class="st" d="M52 58l30-10"/><path class="st" d="M148 58l-30-10"/>`,
  wow: `<circle class="st eye" cx="68" cy="68" r="12"/><circle class="st eye r" cx="132" cy="68" r="12"/>`,
  confused: `<ellipse class="eye fl" cx="68" cy="70" rx="6" ry="6"/><ellipse class="eye r fl" cx="132" cy="68" rx="11" ry="11"/>`,
  curious: dotEyes() + `<path class="st" d="M116 50q16-8 32 0"/>`,
  think: `<ellipse class="eye fl" cx="72" cy="62" rx="9" ry="9"/><ellipse class="eye r fl" cx="136" cy="62" rx="9" ry="9"/>`,
  sleepy: `<path class="st" d="M54 74h28"/><path class="st" d="M118 74h28"/>`, asleep: `<path class="st" d="M54 74q14 10 28 0"/><path class="st" d="M118 74q14 10 28 0"/>`,
  wink: dotEyes() .split("<ellipse class=\"eye r")[0] + `<path class="st" d="M118 76q14-22 28 0"/>`,
  sulk: dotEyes(5, 4) + `<path class="st" d="M50 56l34 4"/><path class="st" d="M150 56l-34 4"/>`,
};
const smile = `<path class="st" d="M72 108q28 24 56 0"/>`;
const MOUTH = {
  neutral: `<path class="st" d="M78 114q22 8 44 0"/>`, joy: smile, proud: smile, wink: smile, love: smile,
  ecstatic: `<path class="fl" d="M68 104q32 44 64 0z"/>`, laugh: `<path class="fl" d="M66 102q34 50 68 0z"/>`,
  anger: `<path class="st" d="M76 124q24-18 48 0"/>`, grumpy: `<path class="st" d="M78 120q22-8 44 0"/>`, sulk: `<path class="st" d="M80 122q20-10 40 0"/>`,
  sad: `<path class="st" d="M76 124q24-20 48 0"/>`, concern: `<path class="st" d="M80 120q20-8 40 0"/>`,
  wow: `<ellipse class="st" cx="100" cy="118" rx="10" ry="13"/>`, confused: `<path class="st" d="M72 118q9-9 18 0t18 0t18 0"/>`,
  curious: `<path class="st" d="M82 116h30"/>`, think: `<path class="st" d="M84 116h24"/>`,
  sleepy: `<path class="st" d="M84 116h32"/>`, asleep: `<ellipse class="st" cx="100" cy="116" rx="8" ry="5"/>`,
};
const face = $("face"), eyesEl = $("eyes"), mouthEl = $("mouth"), moodEl = $("mood");
let calm = 0;
function setFace(emo, hold = 5000) {
  const e = EYES[emo] ? emo : "neutral";
  eyesEl.innerHTML = EYES[e];
  mouthEl.innerHTML = MOUTH[e] || MOUTH.neutral;
  face.dataset.emo = e;
  moodEl.textContent = e;
  if (!reduceMotion) { face.classList.remove("swap"); void face.getBoundingClientRect(); face.classList.add("swap"); }
  clearTimeout(calm);
  if (e !== "neutral") calm = setTimeout(() => setFace("neutral"), hold);
}
setFace("neutral");

/* ---------- chat log ---------- */
function atBottom() { log.scrollTop = log.scrollHeight; }
function line(cls, text) {
  const el = document.createElement("div");
  el.className = "msg " + cls;
  el.textContent = text; // never innerHTML: replies can contain anything the user typed
  log.append(el);
  atBottom();
  return el;
}
function note(cls, text) {
  const el = document.createElement("div");
  el.className = cls;
  el.textContent = text;
  log.append(el);
  atBottom();
}
// the bot "types" its reply; the whole text goes to the live region at once so screen readers aren't fed fragments
let typing = Promise.resolve();
function botSay(text, then) {
  $("announce").textContent = text;
  const el = line("bot", reduceMotion ? text : "");
  if (reduceMotion || text.length < 2) { then?.(); return; }
  el.classList.add("caret");
  face.classList.add("talking");
  const per = Math.min(24, 700 / text.length);
  let i = 0, last = performance.now();
  typing = new Promise((done) => {
    (function step(now) {
      i = Math.min(text.length, i + Math.max(1, Math.floor((now - last) / per)));
      last = now;
      el.textContent = text.slice(0, i);
      atBottom();
      if (i < text.length) requestAnimationFrame(step);
      else { el.classList.remove("caret"); face.classList.remove("talking"); then?.(); done(); }
    })(last);
  });
}

/* ---------- the brain panel ---------- */
const prettyTag = (t) => (t.startsWith("skill:") ? t.slice(6) : t).replaceAll("_", " ");
function showBrain(r) {
  const skill = r.tag.startsWith("skill:") || r.tag === "crisis";
  const warn = ["fallback", "clarify", "clarify_no", "crash"].includes(r.tag);
  $("route").replaceChildren();
  const pill = document.createElement("span");
  pill.className = "pill" + (skill ? " skill" : warn ? " warn" : "");
  pill.textContent = r.tag === "crisis" ? "safety net" : prettyTag(r.tag);
  const small = document.createElement("small");
  small.textContent = r.tag === "crisis" ? "keyword check runs before the network" : skill ? "exact code, not the network" : warn ? "not sure, so it says so" : `${Math.round(r.conf * 100)}% sure`;
  $("route").append(pill, small);

  const bars = $("bars");
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

  const words = $("words");
  words.replaceChildren();
  const ws = r.words || [];
  const max = Math.max(0.001, ...ws.map(([, d]) => Math.abs(d)));
  if (!ws.length) words.innerHTML = '<span class="idle">no words to weigh</span>';
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
  const m = $("meter");
  m.replaceChildren("network pass: ");
  const b = document.createElement("b");
  b.textContent = (r.us || 0) + " µs";
  m.append(b, " in WebAssembly");
}

/* ---------- wiring ---------- */
let bot = null, ended = false;
function end(message) {
  ended = true;
  input.disabled = send.disabled = true;
  input.placeholder = message;
}
function show(r) {
  if (r.kind === "chat") {
    botSay(r.text, () => {
      for (const t of r.toasts || []) note("toast", t);
      if (r.debug) note("debug", r.debug);
    });
    setFace(r.emo);
    showBrain(r);
    if (r.done) typing.then(() => end("The chat ended. Reload to start again."));
  } else if (r.kind === "cmd") {
    if (r.clear) log.replaceChildren();
    if (r.lines.length) line("cmd", r.lines.join("\n"));
    if (r.quit) end("Bye! Reload to chat again.");
  }
}
function ask(text) {
  text = text.trim();
  if (!text || !bot || ended) return;
  line("me", text);
  show(bot.reply(text));
}
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const t = input.value;
  input.value = "";
  ask(t);
  input.focus();
});
chips.addEventListener("click", (e) => {
  const q = e.target instanceof HTMLElement && e.target.dataset.q;
  if (q) { ask(q); input.focus(); }
});
$("help").addEventListener("click", () => ask("/help"));
$("reset").addEventListener("click", () => {
  if (!bot || !confirm("Forget your notes, XP and progress on this device?")) return;
  bot.reset();
  log.replaceChildren();
  botSay("Done, I've forgotten everything. Nice to meet you!");
  setFace("joy");
});

(async () => {
  try {
    bot = await loadBot();
    status.textContent = "tinybot · gru · 251k params · running in your browser";
    input.disabled = send.disabled = false;
    chips.hidden = false;
    botSay("Hi! I'm TinyBot, a small chatbot written from scratch in Rust. Ask me something, do some maths, or type /help. Watch the panel on the right to see how I decide.");
    setFace("joy", 2500);
    setInterval(() => {
      for (const t of bot.tick()) {
        note("toast", "⏰ " + t);
        setFace("wow");
      }
    }, 1000);
  } catch (err) {
    console.error(err);
    status.textContent = "couldn't start";
    const el = line("bot err", "I couldn't start: " + (err && err.message ? err.message : err));
    el.setAttribute("role", "alert");
    setFace("sad", 60000);
  }
})();
