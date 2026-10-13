// Tests web/models.js: manifest checking, the /model command grammar, saved choice, and the switcher against a fake bot.
// Run: node --test tests/
import test from "node:test";
import assert from "node:assert/strict";
import { checkManifest, parseChoice, chooseInitial, createSwitcher, describe } from "../web/models.js";

const entry = (arch, size, bytes = 100_000) => ({ id: `${arch}-${size}`, arch, size, file: arch === "gru" && size === "base" ? "bot.bin" : `models/${arch}-${size}.bin`, bytes });
const RAW = {
  default: "gru-base",
  models: [entry("gru", "base"), entry("bow", "nano"), entry("bow", "small"), entry("cnn", "small"), entry("cnn", "large"), entry("gru", "large", 1_500_000)],
};

test("a manifest is checked, and anything odd in it is dropped", () => {
  const m = checkManifest(RAW);
  assert.equal(m.default, "gru-base");
  assert.equal(m.models.length, 6);
  const dirty = checkManifest({
    default: "nope",
    models: [
      entry("bow", "nano"),
      { ...entry("bow", "nano") }, // duplicate
      { ...entry("cnn", "small"), file: "https://evil.example/m.bin" }, // not shipped with the site
      { ...entry("cnn", "base"), file: "../secret.bin" },
      { ...entry("gru", "xl"), id: "gru-nano" }, // id does not match arch and size
      { id: "lstm-base", arch: "lstm", size: "base", file: "models/lstm-base.bin" },
      null,
      "x",
    ],
  });
  assert.deepEqual(dirty.models.map((x) => x.id), ["bow-nano"]);
  assert.equal(dirty.default, "bow-nano"); // an unknown default falls back to something that exists
  assert.equal(checkManifest(null), null);
  assert.equal(checkManifest({ models: [] }), null);
  assert.equal(checkManifest({ models: "x" }), null);
});

test("/model grammar: parts in any order, a missing part keeps the current one", () => {
  const m = checkManifest(RAW);
  assert.deepEqual(parseChoice("cnn large", m, "gru-base"), { id: "cnn-large" });
  assert.deepEqual(parseChoice("large cnn", m, "gru-base"), { id: "cnn-large" });
  assert.deepEqual(parseChoice("GRU-LARGE", m, "bow-nano"), { id: "gru-large" });
  assert.deepEqual(parseChoice("bow", m, "gru-base").error?.includes("isn't included"), true); // bow-base is not shipped
  assert.deepEqual(parseChoice("small", m, "cnn-large"), { id: "cnn-small" });
  for (const bad of ["", "   ", "-", "huge", "gru huge", "lstm", "../../etc/passwd"]) assert.ok(parseChoice(bad, m, "gru-base").error, bad);
  assert.match(parseChoice("xl", m, "gru-base").error, /isn't included/);
});

test("the saved choice is used only if it is still shipped", () => {
  const m = checkManifest(RAW);
  assert.equal(chooseInitial(m, "cnn-small").id, "cnn-small");
  assert.equal(chooseInitial(m, "cnn-xl").id, "gru-base");
  assert.equal(chooseInitial(m, null).id, "gru-base");
});

function fakeBot({ canSwitch = true } = {}) {
  const calls = [];
  let info = { arch: "gru", tier: "base", params: 251000, label: "251k", kb: 316 };
  return {
    calls,
    canSwitchModel: canSwitch,
    modelInfo: () => info,
    setModel(bytes) {
      const t = new TextDecoder().decode(bytes);
      if (t === "BROKEN") throw new Error("That file is not a valid TinyBot model.");
      calls.push(t);
      const [arch, tier] = t.split("-");
      info = { arch, tier, params: 1, label: "1k", kb: 1 };
    },
  };
}
const fetchBytes = async (url) => {
  if (url.includes("broken")) return new TextEncoder().encode("BROKEN");
  return new TextEncoder().encode(url.replace(/^models\//, "").replace(/\.bin$/, "").replace("bot", "gru-base"));
};

test("the switcher swaps models, reports each result, and ignores nonsense", async () => {
  const bot = fakeBot();
  const seen = [];
  const sw = createSwitcher({ bot, manifest: checkManifest(RAW), current: "gru-base", fetchBytes, onSwitched: (info, id, initial) => seen.push([id, initial]) });
  assert.deepEqual(seen, [["gru-base", true]]);
  assert.equal(sw.command("hello"), null); // not a /model line
  assert.equal(sw.command("/models"), null);

  const lines = sw.command("/model");
  assert.match(lines[0], /gru base/);
  assert.match(lines[1], /bow-nano.*cnn-large.*\[gru-base\].*gru-large/); // ordered by architecture, then size

  assert.deepEqual(await sw.command("/model cnn large"), ["Now running cnn large (1k params, 1 KB)."]);
  assert.equal(sw.current(), "cnn-large");
  assert.deepEqual(bot.calls, ["cnn-large"]);
  assert.deepEqual(seen.at(-1), ["cnn-large", false]);
  assert.match((await sw.command("/model cnn large"))[0], /Already running/);
  assert.equal(bot.calls.length, 1, "no reload for the model that is already running");

  // errors in the command do not switch anything
  assert.match(sw.command("/model nope")[0], /don't know/);
  assert.match(sw.command("/model xl")[0], /isn't included/);
  assert.equal(sw.current(), "cnn-large");

  // going back to a model uses the cached bytes
  await sw.command("/model small");
  assert.equal(sw.current(), "cnn-small");
  assert.equal((await sw.switchTo("gru-large")).ok, true);
});

test("a failed switch keeps the old model and says why", async () => {
  const bot = fakeBot();
  const raw = { ...RAW, models: [...RAW.models, { ...entry("cnn", "nano"), file: "models/broken-cnn.bin" }] };
  const sw = createSwitcher({ bot, manifest: checkManifest(raw), current: "gru-base", fetchBytes });
  const r = await sw.switchTo("cnn-nano");
  assert.equal(r.ok, false);
  assert.match(r.message, /not a valid/);
  assert.equal(sw.current(), "gru-base");
  // and a download error does the same
  const down = createSwitcher({ bot, manifest: checkManifest(RAW), current: "gru-base", fetchBytes: async () => { throw new Error("could not download models/bow-nano.bin (HTTP 404)"); } });
  assert.match((await down.switchTo("bow-nano")).message, /404/);
  assert.equal(down.current(), "gru-base");
});

test("two switches at once: the second is refused instead of racing", async () => {
  const bot = fakeBot();
  let release;
  const slow = (url) => new Promise((r) => { release = () => r(fetchBytes(url)); });
  const sw = createSwitcher({ bot, manifest: checkManifest(RAW), current: "gru-base", fetchBytes: slow });
  const first = sw.switchTo("bow-nano");
  const second = await sw.switchTo("bow-small");
  assert.equal(second.ok, false);
  assert.match(second.message, /Still loading/);
  release();
  assert.equal((await first).ok, true);
  assert.equal(sw.current(), "bow-nano");
});

test("a module that cannot switch says so instead of pretending", async () => {
  const bot = fakeBot({ canSwitch: false });
  const sw = createSwitcher({ bot, manifest: checkManifest(RAW), current: "gru-base", fetchBytes });
  const r = await sw.switchTo("bow-nano");
  assert.equal(r.ok, false);
  assert.match(r.message, /too old/);
});

test("describe shows the size and download", () => {
  assert.match(describe(entry("cnn", "large", 384298)), /cnn large \(500k params, 375 KB download\)/);
  assert.match(describe(entry("bow", "xl", 2_000_000)), /1\.9 MB/);
});
