// Tests web/tinybot.js against a JavaScript stand-in for the wasm module (the real one needs the wasm32 target).
// It checks the memory protocol, state persistence, trap recovery and error reporting. Run: node --test tests/
import test from "node:test";
import assert from "node:assert/strict";

const enc = new TextEncoder();
const dec = new TextDecoder();
const RealWA = globalThis.WebAssembly;

function installFakeWasm({ initCode = 0 } = {}) {
  const log = { inits: [], instances: 0 };
  globalThis.WebAssembly = {
    Memory: RealWA.Memory,
    RuntimeError: RealWA.RuntimeError,
    compile: async () => ({}),
    Instance: class {
      constructor(_module, imports) {
        assert.equal(typeof imports.env.host_now_ms(), "number");
        assert.equal(typeof imports.env.host_perf_ms(), "number");
        assert.ok(imports.env.host_random() < 1);
        assert.ok(Number.isFinite(imports.env.host_tz_offset_min()));
        log.instances++;
        const memory = new RealWA.Memory({ initial: 1 });
        let top = 1024;
        let result = new Uint8Array();
        let notes = [];
        const put = (s) => ((result = enc.encode(s)), result.length);
        const slice = (p, l) => new Uint8Array(memory.buffer, p, l);
        this.exports = {
          memory,
          tb_alloc: (n) => {
            const p = top;
            top += n + 8;
            // grow like a real allocator would, which detaches the old ArrayBuffer
            if (top > memory.buffer.byteLength) memory.grow(1);
            return p;
          },
          tb_free: () => {},
          tb_result_ptr: () => {
            // the result lives in memory too
            const p = top;
            top += result.length + 8;
            if (top > memory.buffer.byteLength) memory.grow(1);
            slice(p, result.length).set(result);
            return p;
          },
          tb_init: (mp, ml, sp, sl) => {
            log.inits.push({ model: dec.decode(slice(mp, ml)), state: dec.decode(slice(sp, sl)) });
            notes = sl ? dec.decode(slice(sp, sl)).split("|") : [];
            return initCode;
          },
          tb_reply: (p, l) => {
            const t = dec.decode(slice(p, l));
            if (t === "TRAP") throw new RealWA.RuntimeError("unreachable");
            if (t === "") return put('{"kind":"empty"}');
            notes.push(t.slice(0, 20));
            return put(JSON.stringify({ kind: "chat", text: "echo " + t, tag: "x", conf: 1, emo: "neutral", toasts: [], done: false, debug: "" }));
          },
          tb_state: () => put(notes.join("|")),
          tb_tick: () => put('["Time\'s up!"]'),
        };
      }
    },
  };
  return log;
}

function memStorage(initial = {}) {
  const m = new Map(Object.entries(initial));
  return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => void m.set(k, String(v)), removeItem: (k) => void m.delete(k), m };
}

const files = { "tinybot.wasm": enc.encode("WASM"), "bot.bin": enc.encode("MODEL") };
const fetchBytes = async (u) => files[u];

test("replies, passes unicode through, persists state", async () => {
  const log = installFakeWasm();
  const { loadBot } = await import("../web/tinybot.js");
  const storage = memStorage();
  const bot = await loadBot({ storage, fetchBytes });
  assert.equal(log.inits[0].model, "MODEL");
  const r = bot.reply("héllo 日本語 😀");
  assert.equal(r.text, "echo héllo 日本語 😀");
  assert.equal(storage.getItem("tinybot.state.v1"), "héllo 日本語 😀");
  assert.equal(bot.reply("").kind, "empty");
  assert.deepEqual(bot.tick(), ["Time's up!"]);
});

test("saved state is handed to the next load", async () => {
  const log = installFakeWasm();
  const { loadBot } = await import("../web/tinybot.js");
  await loadBot({ storage: memStorage({ "tinybot.state.v1": "a|b" }), fetchBytes });
  assert.equal(log.inits[0].state, "a|b");
});

test("a wasm trap restarts from the last good state and the next reply works", async () => {
  const log = installFakeWasm();
  const { loadBot } = await import("../web/tinybot.js");
  const storage = memStorage();
  const bot = await loadBot({ storage, fetchBytes });
  bot.reply("one");
  const r = bot.reply("TRAP");
  assert.equal(r.tag, "crash");
  assert.equal(log.instances, 2);
  assert.equal(log.inits[1].state, "one");
  assert.equal(bot.reply("two").text, "echo two");
});

test("works with storage disabled or throwing", async () => {
  installFakeWasm();
  const { loadBot } = await import("../web/tinybot.js");
  const bot = await loadBot({ storage: null, fetchBytes });
  assert.equal(bot.reply("hi").text, "echo hi");
  const angry = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("quota"); }, removeItem() { throw new Error("blocked"); } };
  const bot2 = await loadBot({ storage: angry, fetchBytes });
  assert.equal(bot2.reply("hi").text, "echo hi");
  bot2.reset();
});

test("reset clears storage and starts empty", async () => {
  const log = installFakeWasm();
  const { loadBot } = await import("../web/tinybot.js");
  const storage = memStorage();
  const bot = await loadBot({ storage, fetchBytes });
  bot.reply("keep me");
  bot.reset();
  assert.equal(storage.getItem("tinybot.state.v1"), null);
  assert.equal(log.inits.at(-1).state, "");
});

test("a model/intents mismatch is reported in plain words", async () => {
  installFakeWasm({ initCode: 2 });
  const { loadBot } = await import("../web/tinybot.js");
  await assert.rejects(loadBot({ storage: null, fetchBytes }), /different intents/);
});

test("a failed download is reported", async () => {
  installFakeWasm();
  const { loadBot } = await import("../web/tinybot.js");
  await assert.rejects(loadBot({ storage: null, fetchBytes: async () => { throw new Error("could not load bot.bin (HTTP 404)"); } }), /404/);
});
