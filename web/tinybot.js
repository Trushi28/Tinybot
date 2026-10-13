// Glue between a page and tinybot.wasm. No dependencies, no build step.
//
//   import { loadBot } from "./tinybot.js";
//   const bot = await loadBot();
//   const r = bot.reply("what is 12*(3+4)");   // { kind: "chat", text: "84", ... }
//
// The wasm module has no clock, randomness or storage of its own, so this file supplies the four host functions
// it imports (see src/clock.rs) and keeps the chat state in localStorage. Everything stays in the browser.

const enc = new TextEncoder();
const dec = new TextDecoder();

const INIT_ERRORS = {
  1: "bot.bin is not a valid TinyBot model file.",
  2: "bot.bin was trained on different intents than the ones built into tinybot.wasm. Run ./build-web.sh again so both come from the same source.",
};

async function fetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not load ${url} (HTTP ${res.status})`);
  return new Uint8Array(await res.arrayBuffer());
}

function safeStorage() {
  try {
    const s = window.localStorage;
    s.getItem("tinybot.probe");
    return s;
  } catch {
    return null; // private mode, blocked cookies, ...: the bot still works, it just forgets on reload
  }
}

/**
 * @param {object} [opts]
 * @param {string} [opts.wasmUrl]  where tinybot.wasm is
 * @param {string} [opts.modelUrl] where bot.bin is
 * @param {string} [opts.key]      localStorage key for the saved state
 * @param {Storage|null} [opts.storage] pass null to disable saving
 * @param {(url:string)=>Promise<Uint8Array>} [opts.fetchBytes] override for tests
 */
export async function loadBot(opts = {}) {
  const { wasmUrl = "tinybot.wasm", modelUrl = "bot.bin", key = "tinybot.state.v1" } = opts;
  const storage = "storage" in opts ? opts.storage : safeStorage();
  const get = opts.fetchBytes || fetchBytes;

  let [wasmBytes, model] = await Promise.all([get(wasmUrl), get(modelUrl)]);
  const module = await WebAssembly.compile(wasmBytes);

  const env = {
    host_now_ms: () => Date.now(),
    host_perf_ms: () => performance.now(),
    host_random: () => Math.random(),
    host_tz_offset_min: () => -new Date().getTimezoneOffset(),
  };

  let ex = null;
  let saved = "";
  // replies carry the running timers as [deadline_unix_ms, label, total_ms]; kept for a module that has no tb_timers
  let replyTimers = [];
  try {
    saved = (storage && storage.getItem(key)) || "";
  } catch {}

  // copy bytes into wasm memory; the buffer may be replaced by a growing allocation, so re-read it every time
  function withBytes(bytes, fn) {
    const ptr = ex.tb_alloc(bytes.length);
    new Uint8Array(ex.memory.buffer, ptr, bytes.length).set(bytes);
    try {
      return fn(ptr, bytes.length);
    } finally {
      ex.tb_free(ptr, bytes.length);
    }
  }

  function readResult(n) {
    return dec.decode(new Uint8Array(ex.memory.buffer, ex.tb_result_ptr(), n));
  }

  function start(state) {
    ex = new WebAssembly.Instance(module, { env }).exports;
    replyTimers = [];
    const code = withBytes(model, (mp, ml) => withBytes(enc.encode(state), (sp, sl) => ex.tb_init(mp, ml, sp, sl)));
    if (code !== 0) throw new Error(INIT_ERRORS[code] || `tb_init failed (${code})`);
  }

  function persist() {
    try {
      const n = ex.tb_state();
      saved = readResult(n);
      if (storage) storage.setItem(key, saved);
    } catch {
      // quota exceeded or storage blocked: keep going without saving
    }
  }

  // an invalid saved state must never stop the page from loading
  try {
    start(saved);
  } catch (e) {
    if (!saved) throw e;
    saved = "";
    start("");
  }

  const crashed = {
    kind: "chat",
    text: "Sorry, that one crashed my brain, so I restarted. Your notes and progress are safe.",
    tag: "crash",
    conf: 0,
    emo: "sad",
    toasts: [],
    done: false,
    debug: "",
  };

  // a module built before tb_timers existed still reports its timers in every reply: ask once so the list is current
  if (!ex.tb_timers) {
    try {
      const n = withBytes(enc.encode("/timers"), (p, l) => ex.tb_reply(p, l));
      const out = n ? JSON.parse(readResult(n)) : {};
      if (Array.isArray(out.timers)) replyTimers = out.timers;
    } catch {}
  }

  return {
    /** Send one line. Always returns an object with a `kind` of "chat", "cmd" or "empty". */
    reply(text) {
      const bytes = enc.encode(String(text).slice(0, 4000));
      try {
        const n = withBytes(bytes, (p, l) => ex.tb_reply(p, l));
        if (n === 0) throw new Error("not initialised");
        const out = JSON.parse(readResult(n));
        if (Array.isArray(out.timers)) replyTimers = out.timers;
        persist();
        return out;
      } catch (e) {
        // a wasm trap leaves the module's internal state unusable: start a fresh instance from the last good state
        console.error("tinybot trapped, restarting", e);
        try {
          start(saved);
        } catch {
          start("");
        }
        return crashed;
      }
    },

    /** Timers that have run out (array of strings). Call about once a second. */
    tick() {
      try {
        const out = JSON.parse(readResult(ex.tb_tick()));
        if (out.length) persist(); // a timer that rang must not ring again after a reload
        return out;
      } catch {
        return [];
      }
    },

    /** Save the state now (it is saved after every reply; call this after anything else that changes it). */
    save() {
      persist();
    },

    /** Running timers: [{ id, label, left_ms, total_ms }], soonest first. */
    timers() {
      try {
        if (ex.tb_timers) return JSON.parse(readResult(ex.tb_timers()));
        const now = Date.now();
        return replyTimers
          .filter(([due]) => due > now)
          .map(([due, label, total]) => ({ id: due, label, left_ms: due - now, total_ms: total }))
          .sort((a, b) => a.left_ms - b.left_ms);
      } catch {
        return [];
      }
    },

    /** True when the module can cancel one timer (modules built before tb_cancel_timer cannot). */
    get canCancel() {
      return typeof ex.tb_cancel_timer === "function";
    },

    /** Cancel one timer by the id from timers(). Saves the new state. Returns true if it existed. */
    cancelTimer(id) {
      try {
        const ok = ex.tb_cancel_timer ? ex.tb_cancel_timer(id) === 1 : false;
        persist();
        return ok;
      } catch {
        return false;
      }
    },

    /** True when this module can swap models without restarting (modules built before tb_set_model cannot). */
    get canSwitchModel() {
      return typeof ex.tb_set_model === "function";
    },

    /**
     * Swap in another model (another architecture or size). The conversation, notes, XP and timers all stay.
     * Throws, and keeps the current model, if the bytes are not a model trained on this module's intents.
     */
    setModel(bytes) {
      if (!ex.tb_set_model) throw new Error("This tinybot.wasm is too old to switch models. Run ./build-web.sh and redeploy.");
      let code;
      try {
        code = withBytes(bytes, (p, l) => ex.tb_set_model(p, l));
      } catch (e) {
        // a trap leaves the module unusable: start again from the last good state with the model that was running
        console.error("tinybot trapped while switching models, restarting", e);
        try { start(saved); } catch { start(""); }
        throw new Error("That model crashed the engine, so I went back to the one that was running.");
      }
      if (code !== 0) {
        throw new Error(code === 2 ? INIT_ERRORS[2] : code === 1 ? "That file is not a valid TinyBot model." : `could not switch models (${code})`);
      }
      model = bytes; // a later restart after a trap must use this model, not the first one
      persist();
    },

    /** The running model: { arch, tier, params, label, kb }, or null for a module without tb_model_info. */
    modelInfo() {
      try {
        if (!ex.tb_model_info) return null;
        return JSON.parse(readResult(ex.tb_model_info()));
      } catch {
        return null;
      }
    },

    /** Forget everything: notes, XP, flow. */
    reset() {
      try {
        if (storage) storage.removeItem(key);
      } catch {}
      saved = "";
      start("");
    },
  };
}
