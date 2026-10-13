// The chat controller shared by the quick-chat card on the home page and the full chat page.
// It owns the log, the typewriter effect and the input; the pages decide what else to show (brain panel, timers).
import { icon } from "./icons.js";

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const LOG_KEY = "tinybot.log";
const GREETING = "Hi! I'm TinyBot, a small chatbot written from scratch in Rust. Ask me something, do some maths, or type /help.";

/**
 * @param {object} o
 * @param {HTMLElement} o.log  @param {HTMLFormElement} o.form  @param {HTMLInputElement} o.input
 * @param {HTMLElement} o.announce  live region for screen readers
 * @param {HTMLElement} [o.chips]  @param {object} [o.face] from createFace
 * @param {(r:object)=>void} [o.onReply]  @param {()=>void} [o.onSend]
 * @param {(text:string)=>(string[]|Promise<string[]>|null)} [o.command]  handles a line the page answers itself (/model);
 *   returns null for lines it does not own
 * @param {string} [o.greeting]
 */
export function createChat(o) {
  const { log, form, input, announce, chips, face } = o;
  const send = form.querySelector("button[type=submit]");
  let bot = null, ended = false, typing = Promise.resolve();

  // the last few lines are carried between pages, so opening the full chat continues the conversation
  let history = [];
  try {
    history = JSON.parse(sessionStorage.getItem(LOG_KEY) || "[]").filter((x) => Array.isArray(x) && typeof x[1] === "string").slice(-60);
  } catch {}
  const remember = (cls, text) => {
    history.push([cls, text]);
    history = history.slice(-60);
    try { sessionStorage.setItem(LOG_KEY, JSON.stringify(history)); } catch {}
  };
  const forgetLog = () => {
    history = [];
    try { sessionStorage.removeItem(LOG_KEY); } catch {}
  };

  const bottom = () => { log.scrollTop = log.scrollHeight; };
  function line(cls, text, keep = true) {
    const el = document.createElement("div");
    el.className = "msg " + cls;
    el.textContent = text; // never innerHTML: replies can contain anything the user typed
    log.append(el);
    bottom();
    if (keep) remember(cls, text);
    return el;
  }
  function note(cls, text) {
    const el = document.createElement("div");
    el.className = cls;
    el.textContent = text;
    log.append(el);
    bottom();
  }

  // the bot "types" its reply; the full text goes to the live region at once so screen readers aren't fed fragments
  function botSay(text, then) {
    announce.textContent = text;
    remember("bot", text);
    const el = line("bot", reduceMotion ? text : "", false);
    if (reduceMotion || text.length < 2) { then?.(); return; }
    el.classList.add("caret");
    face?.talking(true);
    const per = Math.min(24, 700 / text.length);
    let i = 0, last = performance.now();
    typing = new Promise((done) => {
      (function step(now) {
        i = Math.min(text.length, i + Math.max(1, Math.floor((now - last) / per)));
        last = now;
        el.textContent = text.slice(0, i);
        bottom();
        if (i < text.length) requestAnimationFrame(step);
        else { el.classList.remove("caret"); face?.talking(false); then?.(); done(); }
      })(last);
    });
  }

  function end(message) {
    ended = true;
    input.disabled = true;
    if (send) send.disabled = true;
    input.placeholder = message;
  }

  function show(r) {
    if (r.kind === "chat") {
      botSay(r.text, () => {
        for (const t of r.toasts || []) note("toast", t);
        if (r.debug) note("debug", r.debug);
      });
      face?.set(r.emo);
      if (r.done) typing.then(() => end("The chat ended. Reload to start again."));
    } else if (r.kind === "cmd") {
      if (r.clear) { log.replaceChildren(); forgetLog(); }
      if (r.lines.length) line("cmd", r.lines.join("\n"));
      if (r.quit) end("Bye! Reload to chat again.");
    }
    o.onReply?.(r);
  }

  function ask(text) {
    text = text.trim();
    if (!text || !bot || ended) return;
    o.onSend?.();
    line("me", text);
    const own = o.command?.(text);
    if (own) {
      Promise.resolve(own).then((lines) => line("cmd", lines.join("\n")));
      return;
    }
    show(bot.reply(text));
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const t = input.value;
    input.value = "";
    ask(t);
    input.focus();
  });
  chips?.addEventListener("click", (e) => {
    const b = e.target instanceof Element && e.target.closest("button[data-q]");
    if (b) { ask(b.dataset.q); input.focus(); }
  });

  if (send && !send.querySelector("svg")) {
    send.append(icon("send"));
  }

  return {
    ask,
    note,
    /** hand over the loaded bot: enable the input, replay the carried-over conversation or greet */
    ready(b) {
      bot = b;
      input.disabled = false;
      if (send) send.disabled = false;
      if (chips) chips.hidden = false;
      if (history.length) {
        for (const [cls, text] of history) line(cls, text, false);
        note("debug", "— conversation carried over —");
      } else {
        botSay(o.greeting || GREETING);
        face?.set("joy", 2500);
      }
    },
    forget(message) {
      forgetLog();
      log.replaceChildren();
      botSay(message);
      face?.set("joy");
    },
    fail(err) {
      const el = line("bot err", "I couldn't start: " + (err && err.message ? err.message : err), false);
      el.setAttribute("role", "alert");
      face?.set("sad", 60000);
    },
  };
}
