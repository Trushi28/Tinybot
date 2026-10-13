// A small window with a live countdown for every running timer.
//
// The wasm module owns the timers (they live in its session and are saved with the rest of the state); this file only
// polls them, draws them, and rings when one runs out. `bot.timers()` gives [{ id, label, left_ms, total_ms }].
import { icon } from "./icons.js";

const MUTE_KEY = "tinybot.mute";
const R = 26, C = 2 * Math.PI * R;
const fmt = (ms) => {
  const s = Math.max(0, Math.ceil(ms / 1000));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
  return (h ? h + ":" + String(m).padStart(2, "0") : String(m)) + ":" + String(x).padStart(2, "0");
};

/**
 * @param {object} bot  from loadBot()
 * @param {HTMLElement} dock  an empty <section> to render into (the page decides where it sits)
 * @param {(msg:string)=>void} [onDue]  called with "Time's up: ..." text when one rings
 */
export function createTimerWindow(bot, dock, onDue) {
  const items = new Map(); // id -> { li, num, ring, total, left, label }
  let muted = false, audio = null, flash = 0;
  try { muted = localStorage.getItem(MUTE_KEY) === "1"; } catch {}

  dock.hidden = true;
  dock.classList.add("timer-dock");
  dock.setAttribute("aria-label", "Timers");
  const head = document.createElement("div");
  head.className = "td-head";
  const title = document.createElement("span");
  title.className = "td-title";
  title.append(icon("timer"), "Timers");
  const mute = document.createElement("button");
  mute.type = "button";
  mute.className = "td-btn";
  const paintMute = () => {
    mute.replaceChildren(icon(muted ? "bellOff" : "bell"));
    mute.setAttribute("aria-label", muted ? "Sound is off, turn it on" : "Sound is on, turn it off");
    mute.title = muted ? "Sound off" : "Sound on";
  };
  paintMute();
  mute.addEventListener("click", () => {
    muted = !muted;
    try { localStorage.setItem(MUTE_KEY, muted ? "1" : "0"); } catch {}
    paintMute();
  });
  head.append(title, mute);
  const list = document.createElement("ul");
  list.className = "td-list";
  dock.append(head, list);

  // browsers only allow sound after a user gesture, so the page calls arm() when the visitor sends a message
  function arm() {
    try {
      audio ||= new (window.AudioContext || window.webkitAudioContext)();
      if (audio.state === "suspended") audio.resume();
    } catch {}
  }
  function beep() {
    if (muted || !audio) return;
    try {
      for (let i = 0; i < 3; i++) {
        const t = audio.currentTime + i * 0.28;
        const o = audio.createOscillator(), g = audio.createGain();
        o.type = "sine"; o.frequency.value = 880;
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(0.18, t + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, t + 0.22);
        o.connect(g).connect(audio.destination);
        o.start(t); o.stop(t + 0.25);
      }
    } catch {}
  }

  function build(t) {
    const li = document.createElement("li");
    li.className = "td-item";
    const wrap = document.createElement("div");
    wrap.className = "td-ring";
    wrap.innerHTML =
      `<svg viewBox="0 0 64 64" aria-hidden="true"><circle class="trk" cx="32" cy="32" r="${R}"/><circle class="prg" cx="32" cy="32" r="${R}" stroke-dasharray="${C.toFixed(2)}" stroke-dashoffset="0" transform="rotate(-90 32 32)"/></svg>`;
    const num = document.createElement("span");
    num.className = "td-num";
    wrap.append(num);
    const meta = document.createElement("div");
    meta.className = "td-meta";
    const lab = document.createElement("span");
    lab.className = "td-label";
    lab.textContent = t.label || "Timer";
    const state = document.createElement("span");
    state.className = "td-state";
    meta.append(lab, state);
    const x = document.createElement("button");
    x.type = "button";
    x.className = "td-btn";
    x.setAttribute("aria-label", "Cancel this timer");
    x.title = "Cancel";
    x.append(icon("x"));
    li.append(wrap, meta);
    if (bot.canCancel) li.append(x);
    list.append(li);
    const rec = { li, num, state, ring: wrap.querySelector(".prg"), total: Math.max(1, t.total_ms || t.left_ms), left: t.left_ms, done: false, x };
    x.addEventListener("click", () => {
      if (rec.done) remove(rec.id);
      else { bot.cancelTimer(rec.id); remove(rec.id); }
    });
    return rec;
  }
  function remove(id) {
    const r = items.get(id);
    if (!r) return;
    r.li.classList.add("out");
    setTimeout(() => { r.li.remove(); paint(); }, 220);
    items.delete(id);
    paint();
  }
  function paint() {
    dock.hidden = items.size === 0 && !list.children.length;
  }
  function ring(id, r) {
    r.done = true;
    r.li.classList.add("done");
    r.num.textContent = "0:00";
    r.state.textContent = "Time's up!";
    r.ring.setAttribute("stroke-dashoffset", "0");
    r.x.setAttribute("aria-label", "Dismiss");
    r.x.title = "Dismiss";
    beep();
    if (document.hidden) {
      const base = document.title;
      clearInterval(flash);
      let on = false;
      flash = setInterval(() => { document.title = (on = !on) ? "⏰ Time's up!" : base; }, 900);
      const stop = () => { clearInterval(flash); document.title = base; document.removeEventListener("visibilitychange", stop); };
      document.addEventListener("visibilitychange", stop);
    }
    setTimeout(() => remove(id), 15000);
  }

  /** Call about 4 times a second. */
  function poll() {
    const due = bot.tick();
    for (const msg of due) onDue?.(msg);
    if (due.length) bot.save(); // a ringing timer is gone from the session: don't let a page change bring it back
    const now = bot.timers();
    const seen = new Set();
    for (const t of now) {
      seen.add(t.id);
      let r = items.get(t.id);
      if (!r) { r = build(t); r.id = t.id; items.set(t.id, r); dock.hidden = false; }
      r.left = t.left_ms;
      r.num.textContent = fmt(t.left_ms);
      r.state.textContent = t.left_ms < 10000 ? "almost…" : "running";
      r.ring.setAttribute("stroke-dashoffset", String((C * (1 - Math.min(1, t.left_ms / r.total))).toFixed(2)));
      r.li.classList.toggle("soon", t.left_ms < 10000);
    }
    for (const [id, r] of items) {
      if (seen.has(id) || r.done) continue;
      // gone from the list: it rang (it was nearly out) or was cancelled in chat (it had time left)
      if (r.left <= 2500 || due.length) ring(id, r);
      else remove(id);
    }
  }
  setInterval(poll, 250);
  poll();
  return { poll, arm };
}
