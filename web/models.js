// Choosing the brain: architecture (bow / cnn / gru) and size (nano ... xl), live, in the browser.
//
// build-web.sh writes models/index.json next to the model files:
//   { "default": "gru-base", "models": [{ "id": "bow-nano", "arch": "bow", "size": "nano", "file": "models/bow-nano.bin", "bytes": 102266 }, ...] }
// Every file was trained on the same intents the .wasm was built from, so switching is just handing different bytes to
// the running engine: the conversation, notes, XP and timers stay. A page without a manifest simply has no switcher.
import { loadBot } from "./tinybot.js";

const KEY = "tinybot.model";
const ARCHS = ["bow", "cnn", "gru"];
const SIZES = ["nano", "small", "base", "large", "xl"];
export const ARCH_HINT = { bow: "fastest: bag of word features", cnn: "keeps local word order", gru: "keeps order and negation" };
const SIZE_LABEL = { nano: "68k", small: "100k", base: "250k", large: "500k", xl: "1M" };
const MAX_MODEL_BYTES = 16 * 1024 * 1024;
const FILE_OK = /^(bot\.bin|models\/[a-z0-9-]{1,40}\.bin)$/; // only files shipped with the site, never a URL from elsewhere

const store = {
  get() { try { return localStorage.getItem(KEY); } catch { return null; } },
  set(v) { try { localStorage.setItem(KEY, v); } catch {} },
  clear() { try { localStorage.removeItem(KEY); } catch {} },
};

/** Read and check models/index.json. Returns null (no switcher) if it is missing or does not make sense. */
export async function loadManifest(url = "models/index.json", fetchImpl = fetch) {
  try {
    const res = await fetchImpl(url);
    if (!res.ok) return null;
    return checkManifest(await res.json());
  } catch {
    return null;
  }
}

export function checkManifest(raw) {
  if (!raw || typeof raw !== "object" || !Array.isArray(raw.models)) return null;
  const models = [];
  for (const m of raw.models) {
    if (!m || !ARCHS.includes(m.arch) || !SIZES.includes(m.size) || m.id !== `${m.arch}-${m.size}`) continue;
    if (typeof m.file !== "string" || !FILE_OK.test(m.file)) continue;
    if (models.some((x) => x.id === m.id)) continue;
    models.push({ id: m.id, arch: m.arch, size: m.size, file: m.file, bytes: Number.isFinite(m.bytes) ? m.bytes : 0 });
  }
  if (!models.length) return null;
  const def = models.find((m) => m.id === raw.default) || models.find((m) => m.id === "gru-base") || models[0];
  return { default: def.id, models };
}

const kb = (n) => (n >= 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB");
export const describe = (m) => `${m.arch} ${m.size} (${SIZE_LABEL[m.size]} params${m.bytes ? ", " + kb(m.bytes) + " download" : ""})`;

/**
 * "gru large", "large gru", "gru-large", "cnn", "small": a part you leave out keeps its current value.
 * Returns { id } on success or { error } with a message for the chat.
 */
export function parseChoice(text, manifest, currentId) {
  const [ca, cs] = currentId.split("-");
  let arch = ca, size = cs, any = false;
  for (const w of text.toLowerCase().split(/[\s\-/,+]+/).filter(Boolean)) {
    any = true;
    if (ARCHS.includes(w)) arch = w;
    else if (SIZES.includes(w)) size = w;
    else return { error: `I don't know a model part called "${w}".` };
  }
  if (!any) return { error: "Say which model, for example /model gru large." };
  const id = `${arch}-${size}`;
  if (!manifest.models.some((m) => m.id === id)) return { error: `${id} isn't included in this build.` };
  return { id };
}

/** The model to start with: the one saved last time if it is still shipped, otherwise the default. */
export function chooseInitial(manifest, saved) {
  return manifest.models.find((m) => m.id === saved) || manifest.models.find((m) => m.id === manifest.default);
}

/**
 * Start the bot with the model the visitor picked last time. A saved choice that fails to load (file gone after a
 * redeploy, say) falls back to the default instead of leaving the page broken.
 */
export async function startBot(manifest, opts = {}) {
  const first = manifest ? chooseInitial(manifest, store.get()) : null;
  if (!first) return { bot: await loadBot(opts), id: null };
  try {
    return { bot: await loadBot({ ...opts, modelUrl: first.file }), id: first.id };
  } catch (err) {
    const def = chooseInitial(manifest, manifest.default);
    if (def.id === first.id) throw err;
    store.clear();
    return { bot: await loadBot({ ...opts, modelUrl: def.file }), id: def.id };
  }
}

/**
 * The switcher. `els` is optional: without it (the home page) there is no picker, only the /model command.
 * @param {object} o
 * @param {object} o.bot  from loadBot()
 * @param {object} o.manifest  from loadManifest()
 * @param {string} o.current  id of the model that is running
 * @param {object} [o.els]  { root: <details>, arch, size, note, now } elements to render the picker into
 * @param {(info:object|null, id:string, initial:boolean)=>void} [o.onSwitched]  after each successful switch, and once at the start
 * @param {(message:string, ok:boolean)=>void} [o.onNote]  the outcome of a switch made with the picker, for the chat log
 * @param {(url:string)=>Promise<Uint8Array>} [o.fetchBytes]  override for tests
 */
export function createSwitcher({ bot, manifest, current, els, onSwitched, onNote, fetchBytes }) {
  const cache = new Map();
  let busy = false, cur = current;
  const entry = (id) => manifest.models.find((m) => m.id === id);
  // what the engine itself reports about the model that is running (exact size), else what the manifest says
  const runningLabel = (id) => {
    const info = bot.modelInfo();
    return info ? `${info.arch} ${info.tier} (${info.label} params, ${info.kb} KB)` : describe(entry(id));
  };

  async function bytesFor(m) {
    if (cache.has(m.id)) return cache.get(m.id);
    if (fetchBytes) { const b = await fetchBytes(m.file); cache.set(m.id, b); return b; }
    const res = await fetch(m.file);
    if (!res.ok) throw new Error(`could not download ${m.file} (HTTP ${res.status})`);
    const buf = await res.arrayBuffer();
    if (buf.byteLength > MAX_MODEL_BYTES) throw new Error("that model file is unexpectedly large, so I did not load it");
    const b = new Uint8Array(buf);
    cache.set(m.id, b);
    return b;
  }

  /** Switch to model `id`. Resolves { ok, message }; never throws. */
  async function switchTo(id) {
    const m = entry(id);
    if (!m) return { ok: false, message: `${id} isn't included in this build.` };
    if (busy) return { ok: false, message: "Still loading the last model. Give it a second." };
    if (id === cur) return { ok: true, message: `Already running ${m.arch} ${m.size}.` };
    if (!bot.canSwitchModel) return { ok: false, message: "This build of tinybot.wasm is too old to switch models (run ./build-web.sh and redeploy)." };
    busy = true;
    paint(id, true);
    try {
      bot.setModel(await bytesFor(m));
      cur = id;
      store.set(id);
      const info = bot.modelInfo();
      onSwitched?.(info, id, false);
      return { ok: true, message: `Now running ${info ? `${info.arch} ${info.tier} (${info.label} params, ${info.kb} KB)` : describe(m)}.` };
    } catch (err) {
      return { ok: false, message: err && err.message ? err.message : String(err) };
    } finally {
      busy = false;
      paint(cur, false);
    }
  }

  /** The /model command: returns lines for the chat, or null when the text is not for this module. */
  function command(text) {
    const m = /^\/model(?:\s+(.*))?$/i.exec(text.trim());
    if (!m) return null;
    const arg = (m[1] || "").trim();
    if (!arg) {
      const lines = [runningLabel(cur)];
      const order = (x) => ARCHS.indexOf(x.arch) * 10 + SIZES.indexOf(x.size);
      lines.push("available: " + [...manifest.models].sort((a, b) => order(a) - order(b)).map((x) => (x.id === cur ? `[${x.id}]` : x.id)).join(" "));
      lines.push("switch with the picker or /model <arch> <size>, e.g. /model cnn small or /model large (one part keeps the other)");
      lines.push("archs: " + ARCHS.map((a) => `${a} (${ARCH_HINT[a]})`).join(", "));
      return lines;
    }
    const pick = parseChoice(arg, manifest, cur);
    if (pick.error) return [pick.error, "try: /model gru large   /model cnn   /model small"];
    return switchTo(pick.id).then((r) => [r.message]);
  }

  // ---- the picker (chat page only)
  const seg = (host, items, onPick) => {
    host.replaceChildren();
    for (const it of items) {
      const b = document.createElement("button");
      b.type = "button";
      b.setAttribute("role", "radio");
      b.dataset.v = it.v;
      b.textContent = it.v;
      if (it.title) b.title = it.title;
      b.addEventListener("click", () => onPick(it.v));
      host.append(b);
    }
  };
  function nearest(arch, size) {
    // the same size in another architecture, or the closest size that architecture has
    const have = manifest.models.filter((m) => m.arch === arch);
    return (have.find((m) => m.size === size) || have.sort((a, b) => Math.abs(SIZES.indexOf(a.size) - SIZES.indexOf(size)) - Math.abs(SIZES.indexOf(b.size) - SIZES.indexOf(size)))[0]).id;
  }
  function paint(id, loading) {
    if (!els) return;
    const [a, s] = id.split("-");
    for (const b of els.arch.children) {
      b.setAttribute("aria-checked", String(b.dataset.v === a));
      b.disabled = loading;
    }
    for (const b of els.size.children) {
      b.setAttribute("aria-checked", String(b.dataset.v === s));
      b.disabled = loading || !manifest.models.some((m) => m.arch === a && m.size === b.dataset.v);
    }
    els.root.setAttribute("aria-busy", String(loading));
    els.now.textContent = `${a} · ${s}`;
    els.note.textContent = loading ? `loading ${describe(entry(id))}…` : runningLabel(id);
  }
  if (els) {
    const present = (list) => ARCHS.filter((a) => list.some((m) => m.arch === a)); // keep a stable order
    const picked = async (id) => {
      const r = await switchTo(id);
      onNote?.(r.message, r.ok);
      if (!r.ok) els.note.textContent = r.message;
    };
    seg(els.arch, present(manifest.models).map((v) => ({ v, title: ARCH_HINT[v] })), (a) => picked(nearest(a, cur.split("-")[1])));
    seg(els.size, SIZES.filter((s) => manifest.models.some((m) => m.size === s)).map((v) => ({ v, title: `${SIZE_LABEL[v]} parameters` })), (s) => picked(nearest(cur.split("-")[0], s)));
    els.root.hidden = false;
    // on a phone the picker starts folded so the chat gets the screen
    if (matchMedia("(max-width: 1020px)").matches) els.root.open = false;
    paint(cur, false);
  }
  onSwitched?.(bot.modelInfo(), cur, true);

  return { command, switchTo, current: () => cur };
}
