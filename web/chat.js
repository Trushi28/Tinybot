// The dedicated chat page: full-height console, the network's view of every message, and a timer window.
import { wireTheme } from "./theme.js";
import { createFace } from "./face.js";
import { createChat } from "./chatcore.js";
import { renderBrain } from "./brain.js";
import { createTimerWindow } from "./timers.js";
import { loadManifest, startBot, createSwitcher } from "./models.js";

const $ = (id) => document.getElementById(id);
wireTheme($("theme"));

const face = createFace($("face"), $("mood"));
const brainEls = { route: $("route"), bars: $("bars"), words: $("words"), meter: $("meter") };
let timers = null, bot = null, switcher = null;
const chat = createChat({
  log: $("log"), form: $("form"), input: $("input"), announce: $("announce"), chips: $("chips"), face,
  command: (t) => switcher?.command(t) ?? null,
  onSend: () => timers?.arm(),
  onReply: (r) => {
    renderBrain(r, brainEls);
    timers?.poll(); // a "set a timer" reply should show its window at once, not at the next poll
  },
});

$("help").addEventListener("click", () => chat.ask("/help"));
$("reset").addEventListener("click", () => {
  if (!bot || !confirm("Forget your notes, XP and progress on this device?")) return;
  bot.reset();
  chat.forget("Done, I've forgotten everything. Nice to meet you!");
});

(async () => {
  try {
    const manifest = await loadManifest();
    const started = await startBot(manifest);
    bot = started.bot;
    const status = (info) => ($("status").textContent = info ? `tinybot · ${info.arch} ${info.tier} · ${info.label} params · running in your browser` : "tinybot · running in your browser");
    status(bot.modelInfo());
    chat.ready(bot);
    if (manifest && started.id && bot.canSwitchModel) {
      switcher = createSwitcher({
        bot, manifest, current: started.id,
        els: { root: $("picker"), arch: $("pk-arch"), size: $("pk-size"), note: $("pk-note"), now: $("pk-now") },
        onSwitched: (info, id, initial) => { status(info); if (!initial) face.set("wow", 1200); },
        onNote: (msg, ok) => chat.note(ok ? "toast" : "debug", msg),
      });
    }
    timers = createTimerWindow(bot, $("timers"), (msg) => {
      chat.note("toast", "⏰ " + msg);
      face.set("wow");
    });
  } catch (err) {
    console.error(err);
    $("status").textContent = "couldn't start";
    chat.fail(err);
  }
})();
