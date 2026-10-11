// End to end: the real tinybot.wasm and the shipped bot.bin, driven through web/tinybot.js exactly as a page would.
// Needs the wasm built first (./build-web.sh, or the cargo rustc line in .github/workflows/web.yml).
// Run: node tests/e2e.mjs [path/to/tinybot.wasm]
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { loadBot } from "../web/tinybot.js";

const wasmPath = process.argv[2] || "target/wasm32-unknown-unknown/release/tinybot.wasm";
const files = { "tinybot.wasm": wasmPath, "bot.bin": "web/bot.bin" };
const fetchBytes = async (u) => new Uint8Array(await readFile(files[u]));
const store = new Map();
const storage = { getItem: (k) => store.get(k) ?? null, setItem: (k, v) => void store.set(k, String(v)), removeItem: (k) => void store.delete(k) };

let bot = await loadBot({ storage, fetchBytes });

const say = (t) => {
  const r = bot.reply(t);
  assert.ok(["chat", "cmd", "empty"].includes(r.kind), JSON.stringify(r));
  return r;
};

// maths and units go through the skills, not the network
assert.match(say("what is 12*(3+4)").text, /84/);
assert.match(say("10 km to miles").text, /6\.21/);

// ordinary requests are understood by the model that ships with the page
for (const q of ["hello", "tell me a joke", "thanks a lot"]) {
  const r = say(q);
  assert.notEqual(r.tag, "fallback", `${q}: ${JSON.stringify(r)}`);
}

// state survives a "page reload" (a fresh instance built from localStorage)
say("my name is Zara");
say("note: buy stamps");
bot = await loadBot({ storage, fetchBytes });
assert.match(say("show my notes").text, /buy stamps/);
assert.match(say("what is my name").text, /Zara/);

// commands: allowed ones work, file/training ones are refused politely
assert.ok(say("/help").lines.length > 5);
assert.match(say("/teach foo => bar").lines.join(" "), /native app/);
assert.equal(say("/clear").clear, true);

// timers are announced by tick() (the host clock really drives them)
assert.doesNotMatch(say("set a timer for 1 second").text, /plain mode/);
await new Promise((r) => setTimeout(r, 1300));
assert.match(bot.tick().join(" "), /Time's up/);

// hostile and odd input never traps
for (const t of ["x".repeat(20000), "((((((((((((((((((((", "/", "日本語 😀 \u0000 ‮", "[#> @", "a\nb\tc"]) say(t);
// ... and if it ever did, the glue restarts the module, so the bot keeps answering
assert.match(say("what is 2+2").text, /4/);

bot.reset();
assert.equal(storage.getItem("tinybot.state.v1"), null);
console.log("e2e ok");
