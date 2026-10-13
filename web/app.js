// Home page: a short quick-chat card. The dedicated chat page (chat.html) has the full console and brain panel.
import { wireTheme } from "./theme.js";
import { createFace } from "./face.js";
import { createChat } from "./chatcore.js";
import { createTimerWindow } from "./timers.js";
import { loadManifest, startBot, createSwitcher } from "./models.js";

const $ = (id) => document.getElementById(id);
wireTheme($("theme"));

const face = createFace($("face"));
const chat = createChat({
  log: $("log"), form: $("form"), input: $("input"), announce: $("announce"), chips: $("chips"), face,
  onSend: () => timers?.arm(),
  command: (t) => switcher?.command(t) ?? null,
});
let timers = null, switcher = null;

(async () => {
  try {
    const manifest = await loadManifest();
    const { bot, id } = await startBot(manifest); // the model picked last time on the chat page
    $("status").textContent = "running in your browser";
    chat.ready(bot);
    if (manifest && id && bot.canSwitchModel) switcher = createSwitcher({ bot, manifest, current: id }); // /model works here too
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
