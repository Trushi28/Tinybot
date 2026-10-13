//! The browser build: TinyBot behind a tiny C ABI, so any page can drive it with a few lines of JavaScript.
//!
//! Nothing here is specific to wasm except the thin `exports` module at the bottom; `Web` itself is plain Rust and
//! is tested natively. The page does the I/O the sandbox cannot:
//!
//! * it fetches `bot.bin` (the trained model, see `build-web.sh`) and passes the bytes to `tb_init`
//! * it keeps the chat state (memory, XP, flow) in `localStorage` and hands it back on the next visit
//! * it provides the clock, randomness and time zone (`env.host_*`, see `clock.rs`)
//!
//! Commands that would write files or retrain the network (`/teach`, `/fix`, `/know`, ...) are switched off here.
//! Learning needs `intents.txt` to be rewritten and a training run, which belong on your own machine.

use crate::bot::{Brain, Session};
use crate::clock;
use crate::cmds::{self, App};
use crate::intents;
use crate::model::Ensemble;
use crate::skills::Memory;

/// Separates the memory text from the flow text in an exported state. Control characters are stripped from
/// everything stored in memory, so this byte can never appear inside either half.
const SEP: char = '\u{1}';
const MAX_INPUT_CHARS: usize = 1000;
const MAX_STATE_BYTES: usize = 1 << 20;

/// Commands that need the file system or a training run.
const NATIVE_ONLY: [&str; 8] = ["teach", "fix", "learn", "unlearn", "retrain", "know", "models", "train"];

pub struct Web {
    app: App,
}

/// Why `Web::new` failed; the number is what `tb_init` returns to the page.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum InitError {
    /// the bytes are not a model file
    BadModel = 1,
    /// the model was trained on different intents than the ones built into this binary
    IntentsMismatch = 2,
}

/// Parse model bytes and check they were trained on the intents built into this binary.
fn load_brain(model: &[u8]) -> Result<Brain, InitError> {
    let model = Ensemble::from_bytes(model).ok_or(InitError::BadModel)?;
    let intents = intents::load_embedded();
    if model.tags.len() != intents.len() || model.tags.iter().zip(&intents).any(|(t, i)| *t != i.tag) {
        return Err(InitError::IntentsMismatch);
    }
    Ok(Brain::new(model, intents))
}

impl Web {
    pub fn new(model: &[u8], state: &str) -> Result<Web, InitError> {
        let brain = load_brain(model)?;
        let mut parts = state.split(SEP);
        let (mem_text, flow_text, timer_text) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""), parts.next().unwrap_or(""));
        let mut s = Session::new(Memory::from_text(mem_text), false);
        s.flow_from_text(flow_text);
        restore_timers(&mut s, timer_text);
        s.realtime = true;
        let mut app = App::new(brain, s);
        app.width = 60;
        Ok(Web { app })
    }

    /// Swap in another model (another architecture or size) without touching the conversation: notes, XP, mood, flow
    /// and timers all stay. If the bytes are not a usable model the old one keeps running.
    pub fn set_model(&mut self, model: &[u8]) -> Result<(), InitError> {
        self.app.brain = load_brain(model)?;
        Ok(())
    }

    /// The running model as JSON: `{"arch":"gru","tier":"base","params":251000,"label":"251k","kb":316}`.
    pub fn model_info(&self) -> String {
        let m = &self.app.brain.model;
        format!(
            "{{\"arch\":{},\"tier\":{},\"params\":{},\"label\":{},\"kb\":{:.0}}}",
            json_str(m.cfg().arch.name()),
            json_str(crate::train::model_tier(m).name),
            m.nominal_params(),
            json_str(&crate::train::params_label(m)),
            crate::train::model_kb(m)
        )
    }

    /// Everything worth keeping between visits, as text: memory, intent flow and running timers.
    pub fn state(&self) -> String {
        let s = &self.app.s;
        let s_text = format!("{}{SEP}{}{SEP}{}", s.mem.to_text(), s.flow_to_text(), self.timers_text(false));
        if s_text.len() > MAX_STATE_BYTES { String::new() } else { s_text }
    }

    /// The running timers: JSON `[[deadline_unix_ms,label,total_ms],..]` for the page's countdown window, or one
    /// `deadline_unix_ms total_ms label` line each for saving, so a timer survives a page change.
    fn timers_text(&self, json: bool) -> String {
        let (now_ms, now_us) = (clock::unix_ms(), clock::mono_us());
        let items = self.app.s.timers.iter().map(|(due, label, total)| {
            let deadline = now_ms + (due.saturating_sub(now_us) / 1000) as i64;
            if json { format!("[{deadline},{},{}]", json_str(label), total / 1000) } else { format!("{deadline} {} {label}", total / 1000) }
        });
        let (sep, open, close) = if json { (",", "[", "]") } else { ("\n", "", "") };
        format!("{open}{}{close}", items.collect::<Vec<_>>().join(sep))
    }

    /// Running timers as JSON `[{"id":..,"label":..,"left_ms":..,"total_ms":..}]`, soonest first. `id` is stable for the
    /// life of the timer on this page, and is what `cancel_timer` takes.
    pub fn timers(&self) -> String {
        let now = clock::mono_us();
        let mut v: Vec<&(u64, String, u64)> = self.app.s.timers.iter().collect();
        v.sort_by_key(|(t, _, _)| *t);
        let items: Vec<String> = v
            .iter()
            .map(|(t, l, total)| format!("{{\"id\":{t},\"label\":{},\"left_ms\":{},\"total_ms\":{}}}", json_str(l), t.saturating_sub(now) / 1000, total / 1000))
            .collect();
        format!("[{}]", items.join(","))
    }

    /// Cancel one timer by the `id` from `timers()`. Returns whether it existed.
    pub fn cancel_timer(&mut self, id: u64) -> bool {
        let n = self.app.s.timers.len();
        self.app.s.timers.retain(|(t, _, _)| *t != id);
        self.app.s.timers.len() != n
    }

    /// Timers that have run out since the last call, as a JSON array of strings.
    pub fn tick(&mut self) -> String {
        json_list(&self.app.s.take_due())
    }

    /// One user line in, one JSON object out.
    ///
    /// `{"kind":"chat","text":..,"tag":..,"conf":..,"emo":..,"toasts":[..],"done":bool,"debug":..,"top":[[tag,p]..],
    /// "words":[[word,influence]..],"us":..}` for talk (`top`, `words` and `us` describe the network's own view of
    /// the line, which the skills may override),
    /// `{"kind":"cmd","lines":[..],"quit":bool,"clear":bool}` for `/commands`, `{"kind":"empty"}` for nothing.
    pub fn reply(&mut self, input: &str) -> String {
        let line: String = input.chars().filter(|c| !c.is_control()).take(MAX_INPUT_CHARS).collect();
        let line = line.trim();
        if line.is_empty() {
            return "{\"kind\":\"empty\"}".to_string();
        }
        if line.starts_with('/') {
            return self.command(line);
        }
        // what the network itself thinks, taken before the reply so the intent-flow prior is the one that routed it
        let a = self.app.brain.analyze(&self.app.s, line);
        let r = self.app.brain.reply(&mut self.app.s, line);
        // `r.learn` is ignored on purpose: the browser build does not retrain
        self.app.s.persona.store(&mut self.app.s.mem);
        let conf = if r.conf.is_finite() { r.conf } else { 0.0 };
        let debug = if self.app.s.debug { format!("{} {}", r.tag, r.debug) } else { String::new() };
        let tags = &self.app.brain.model.tags;
        let mut ranked: Vec<(usize, f32)> = a.probs.iter().copied().enumerate().filter(|(_, p)| p.is_finite()).collect();
        ranked.sort_by(|x, y| y.1.total_cmp(&x.1));
        let top = ranked.iter().take(5).map(|(k, p)| format!("[{},{p:.3}]", json_str(&tags[*k]))).collect::<Vec<_>>().join(",");
        let words = a.saliency.iter().take(24).filter(|(_, d)| d.is_finite()).map(|(w, d)| format!("[{},{d:.3}]", json_str(w))).collect::<Vec<_>>().join(",");
        let us = if a.micros.is_finite() { a.micros } else { 0.0 };
        format!(
            "{{\"kind\":\"chat\",\"text\":{},\"tag\":{},\"conf\":{conf:.3},\"emo\":{},\"toasts\":{},\"done\":{},\"debug\":{},\"top\":[{top}],\"words\":[{words}],\"us\":{us:.0},\"timers\":{}}}",
            json_str(&r.text),
            json_str(&r.tag),
            json_str(r.emo),
            json_list(&r.toasts),
            r.done,
            json_str(&debug),
            self.timers_text(true)
        )
    }

    fn command(&mut self, line: &str) -> String {
        let cmd = line.trim_start_matches('/');
        let name = cmd.split_whitespace().next().unwrap_or("");
        let (lines, quit, clear) = if NATIVE_ONLY.contains(&name) {
            (vec![format!("/{name} needs files and training, so it only works in the native app (cargo run --release). The web demo is read-only.")], false, false)
        } else if name == "model" {
            let m = &self.app.brain.model;
            (
                vec![
                    format!("{} ({})", crate::train::model_name(m), m.cfg().tag()),
                    format!("{} params, {:.0} KB download", crate::train::params_label(m), crate::train::model_kb(m)),
                    "switch with the model picker, or /model gru large (archs: bow cnn gru; sizes: nano small base large)".to_string(),
                    "running in your browser as WebAssembly; nothing you type leaves this page".to_string(),
                ],
                false,
                false,
            )
        } else {
            let out = cmds::run(&mut self.app, line);
            if name != "forget" {
                self.app.s.persona.store(&mut self.app.s.mem);
            }
            (out.lines, out.quit, name == "clear")
        };
        format!("{{\"kind\":\"cmd\",\"lines\":{},\"quit\":{quit},\"clear\":{clear},\"timers\":{}}}", json_list(&lines), self.timers_text(true))
    }
}

/// Put back the timers saved by `Web::state` ("deadline_unix_ms total_ms label" per line). One that ran out while the
/// page was away rings on the first tick; one that ran out over an hour ago, or is not a sane timer, is dropped.
fn restore_timers(s: &mut Session, text: &str) {
    let (now_ms, now_us) = (clock::unix_ms(), clock::mono_us());
    for line in text.lines().take(20) {
        let mut it = line.splitn(3, ' ');
        let (Some(Ok(due)), Some(Ok(total))) = (it.next().map(str::parse::<i64>), it.next().map(str::parse::<u64>)) else { continue };
        let left = due.saturating_sub(now_ms);
        if !(-3_600_000..=86_400_000).contains(&left) || total == 0 || total > 86_400_000 {
            continue;
        }
        let label: String = it.next().unwrap_or("").chars().filter(|c| !c.is_control()).take(60).collect();
        s.timers.push((now_us + left.max(0) as u64 * 1000, label, total * 1000));
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_list(v: &[String]) -> String {
    format!("[{}]", v.iter().map(|s| json_str(s)).collect::<Vec<_>>().join(","))
}

/// The C ABI the page calls. Compiled for wasm (and for the native tests, which drive it the way JS does).
///
/// Memory convention: the page asks `tb_alloc` for a buffer, copies bytes in, passes pointer and length, and frees
/// it with `tb_free`. Results come back through one shared buffer: the call returns its length and the page reads
/// that many bytes at `tb_result_ptr()` straight away (before the next call).
#[cfg(any(target_arch = "wasm32", test))]
mod exports {
    use super::Web;
    use std::cell::RefCell;

    thread_local! {
        static WEB: RefCell<Option<Web>> = const { RefCell::new(None) };
        static RESULT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }

    fn put(s: String) -> usize {
        RESULT.with(|r| {
            let mut r = r.borrow_mut();
            *r = s.into_bytes();
            r.len()
        })
    }

    /// # Safety
    /// `ptr` must come from `tb_alloc(len)` and not have been freed, or be null with `len == 0`.
    unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
        if ptr.is_null() || len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(ptr, len) } }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn tb_alloc(len: usize) -> *mut u8 {
        Box::into_raw(vec![0u8; len.max(1)].into_boxed_slice()) as *mut u8
    }

    /// # Safety
    /// `ptr`/`len` must be exactly what `tb_alloc(len)` returned/was given.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn tb_free(ptr: *mut u8, len: usize) {
        if !ptr.is_null() {
            drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len.max(1))) });
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn tb_result_ptr() -> *const u8 {
        RESULT.with(|r| r.borrow().as_ptr())
    }

    /// Load the model and the saved state (state may be empty). Returns 0 on success, 1 for a bad model file,
    /// 2 when the model does not match the built-in intents.
    ///
    /// # Safety
    /// Both pointers must satisfy the `tb_alloc` contract.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn tb_init(model: *const u8, model_len: usize, state: *const u8, state_len: usize) -> i32 {
        let (model, state) = unsafe { (bytes(model, model_len), bytes(state, state_len)) };
        let state = String::from_utf8_lossy(state);
        match Web::new(model, &state) {
            Ok(w) => {
                WEB.with(|c| *c.borrow_mut() = Some(w));
                0
            }
            Err(e) => e as i32,
        }
    }

    /// Reply to one line; returns the length of the JSON in the result buffer, or 0 if `tb_init` has not succeeded.
    ///
    /// # Safety
    /// Pointer must satisfy the `tb_alloc` contract.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn tb_reply(text: *const u8, len: usize) -> usize {
        let text = String::from_utf8_lossy(unsafe { bytes(text, len) }).into_owned();
        let out = WEB.with(|c| c.borrow_mut().as_mut().map(|w| w.reply(&text)));
        out.map(put).unwrap_or(0)
    }

    /// The state to persist (call after each reply); returns its length in the result buffer.
    #[unsafe(no_mangle)]
    pub extern "C" fn tb_state() -> usize {
        let out = WEB.with(|c| c.borrow().as_ref().map(|w| w.state()));
        out.map(put).unwrap_or(0)
    }

    /// Swap the model, keeping the conversation. Returns 0 on success, 1 if the bytes are not a model, 2 if the model
    /// was trained on other intents, -1 if `tb_init` has not succeeded yet. On any failure the old model keeps running.
    ///
    /// # Safety
    /// `model..model+len` must be memory returned by `tb_alloc`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn tb_set_model(model: *const u8, len: usize) -> i32 {
        let bytes = unsafe { bytes(model, len) };
        WEB.with(|c| match c.borrow_mut().as_mut() {
            None => -1,
            Some(w) => match w.set_model(bytes) {
                Ok(()) => 0,
                Err(e) => e as i32,
            },
        })
    }

    /// The running model as JSON (see `Web::model_info`); returns its length in the result buffer.
    #[unsafe(no_mangle)]
    pub extern "C" fn tb_model_info() -> usize {
        let out = WEB.with(|c| c.borrow().as_ref().map(|w| w.model_info()));
        out.map(put).unwrap_or(0)
    }

    /// Running timers as JSON (see `Web::timers`); returns its length in the result buffer.
    #[unsafe(no_mangle)]
    pub extern "C" fn tb_timers() -> usize {
        let out = WEB.with(|c| c.borrow().as_ref().map(|w| w.timers()));
        out.map(put).unwrap_or(0)
    }

    /// Cancel the timer with this id (ids stay below 2^53, so a JS number carries them exactly). Returns 1 or 0.
    #[unsafe(no_mangle)]
    pub extern "C" fn tb_cancel_timer(id: f64) -> i32 {
        if !(id.is_finite() && id >= 0.0) {
            return 0;
        }
        WEB.with(|c| c.borrow_mut().as_mut().map(|w| w.cancel_timer(id as u64)).unwrap_or(false)) as i32
    }

    /// Due timers as a JSON array; returns its length in the result buffer.
    #[unsafe(no_mangle)]
    pub extern "C" fn tb_tick() -> usize {
        let out = WEB.with(|c| c.borrow_mut().as_mut().map(|w| w.tick()));
        out.map(put).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Arch, Cfg};

    const TINY: Cfg = Cfg { arch: Arch::Bow, dim: 8, hid: 16, bits: 10, ch: 0, epochs: 6 };

    /// A model trained on the built-in intents (tiny, so the test stays fast; answers are not the point).
    fn model_bytes() -> Vec<u8> {
        let intents = intents::load_embedded();
        let (tags, ex) = intents::flatten(&intents);
        let t = Ensemble::train(&ex, tags, &[TINY, TINY], 1, 1.0, 0);
        Ensemble::distill(&t, &ex, TINY, 1, 1.0, 0).to_bytes()
    }

    fn bytes_cache() -> &'static Vec<u8> {
        static B: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
        B.get_or_init(model_bytes)
    }

    #[test]
    fn json_is_escaped() {
        assert_eq!(json_str("a\"b\\c\nd\u{1}é"), "\"a\\\"b\\\\c\\nd\\u0001é\"");
        assert_eq!(json_list(&["x".into(), "y\"".into()]), "[\"x\",\"y\\\"\"]");
        assert_eq!(json_list(&[]), "[]");
    }

    #[test]
    fn rejects_bad_model_and_mismatched_intents() {
        assert_eq!(Web::new(b"not a model", "").err(), Some(InitError::BadModel));
        assert_eq!(Web::new(&[], "").err(), Some(InitError::BadModel));
        let mut cut = bytes_cache().clone();
        cut.truncate(cut.len() / 2);
        assert_eq!(Web::new(&cut, "").err(), Some(InitError::BadModel));
        // a model for other tags
        let ex = vec![(0usize, "hello".to_string()), (1, "bye".to_string())];
        let t = Ensemble::train(&ex, vec!["a".into(), "b".into()], &[TINY], 1, 1.0, 0);
        assert_eq!(Web::new(&t.to_bytes(), "").err(), Some(InitError::IntentsMismatch));
    }

    #[test]
    fn chat_commands_and_blocked_commands() {
        // a developer's checkout may already hold these files from native use, so compare before/after instead of
        // asserting they do not exist
        let snapshot = || (std::fs::read("learned.txt").ok(), std::fs::read("kb_user.txt").ok());
        let before = snapshot();
        let mut w = Web::new(bytes_cache(), "").unwrap();
        assert_eq!(w.reply("   \u{7} "), "{\"kind\":\"empty\"}");
        let r = w.reply("what is 12*(3+4)");
        assert!(r.starts_with("{\"kind\":\"chat\""), "{r}");
        assert!(r.contains("84"), "{r}");
        assert!(r.contains("\"top\":[[\"") && r.contains("\"words\":[") && r.contains("\"us\":"), "{r}");
        let r = w.reply("/help");
        assert!(r.starts_with("{\"kind\":\"cmd\"") && r.contains("/teach"), "{r}");
        for c in ["/teach a => b", "/fix greeting", "/know x => y", "/retrain", "/unlearn"] {
            let r = w.reply(c);
            assert!(r.contains("only works in the native app"), "{c}: {r}");
        }
        assert!(w.reply("/clear").contains("\"clear\":true"));
        assert!(w.reply("/quit").contains("\"quit\":true"));
        assert!(w.reply("/model").contains("WebAssembly"));
        assert!(w.reply("/stats").contains("level"));
        // nothing was written to disk by any of that
        assert!(snapshot() == before, "the web build must not touch learned.txt or kb_user.txt");
    }

    #[test]
    fn state_round_trips() {
        let mut w = Web::new(bytes_cache(), "").unwrap();
        w.reply("my name is Zara");
        w.reply("note: buy stamps");
        w.reply("hello");
        let st = w.state();
        assert!(st.contains('\u{1}') && st.contains("fact:name=Zara") && st.contains("note:buy stamps"), "{st:?}");
        let mut w2 = Web::new(bytes_cache(), &st).unwrap();
        assert!(w2.reply("show my notes").contains("buy stamps"));
        // garbage state never panics and just starts fresh-ish
        for junk in ["", "\u{1}", "fact:=\nnote:\ntodo:\u{1}a b c\nx y", "\u{1}\u{1}\u{1}", "fact:name=\u{0}"] {
            let mut w = Web::new(bytes_cache(), junk).unwrap();
            assert!(w.reply("hello").starts_with("{\"kind\":\"chat\""));
        }
    }

    #[test]
    fn timers_are_announced_by_tick() {
        let mut w = Web::new(bytes_cache(), "").unwrap();
        let r = w.reply("set a timer for 1 second");
        assert!(r.contains("Timer set") && !r.contains("plain mode"), "{r}");
        assert_eq!(w.tick(), "[]");
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let t = w.tick();
        assert!(t.contains("Time's up"), "{t}");
        assert_eq!(w.tick(), "[]");
    }

    #[test]
    fn the_model_can_be_swapped_without_losing_the_conversation() {
        // a different architecture and size, trained on the same intents
        let other_cfg = Cfg { arch: Arch::Bow, dim: 12, hid: 24, bits: 11, ch: 0, epochs: 4 };
        let intents = intents::load_embedded();
        let (tags, ex) = intents::flatten(&intents);
        let t = Ensemble::train(&ex, tags, &[TINY], 1, 1.0, 0);
        let other = Ensemble::distill(&t, &ex, other_cfg, 1, 1.0, 0).to_bytes();

        let mut w = Web::new(bytes_cache(), "").unwrap();
        let before = w.model_info();
        assert!(before.contains("\"arch\":\"bow\"") && before.contains("\"params\":"), "{before}");
        w.reply("my name is Zara");
        w.reply("remind me in 10 minutes to stretch");
        assert!(w.set_model(&other).is_ok());
        let after = w.model_info();
        assert_ne!(before, after, "the new model is the one reporting");
        // notes, name and timers survived, and it still answers
        assert!(w.reply("what is my name").contains("Zara"));
        assert!(w.timers().contains("stretch"));
        assert!(w.reply("what is 12*(3+4)").contains("84"));

        // anything that is not a model for these intents is refused and the current model keeps going
        assert_eq!(w.set_model(b"nope"), Err(InitError::BadModel));
        assert_eq!(w.set_model(&[]), Err(InitError::BadModel));
        let ex2 = vec![(0usize, "hello".to_string()), (1, "bye".to_string())];
        let t2 = Ensemble::train(&ex2, vec!["a".into(), "b".into()], &[TINY], 1, 1.0, 0);
        assert_eq!(w.set_model(&t2.to_bytes()), Err(InitError::IntentsMismatch));
        assert_eq!(w.model_info(), after);
        assert!(w.reply("what is my name").contains("Zara"));
    }

    #[test]
    fn timers_list_and_cancel() {
        let mut w = Web::new(bytes_cache(), "").unwrap();
        assert_eq!(w.timers(), "[]");
        w.reply("remind me in 10 minutes to stretch");
        w.reply("set a timer for 2 minutes");
        let t = w.timers();
        assert!(t.starts_with("[{\"id\":") && t.contains("\"left_ms\":") && t.contains("\"total_ms\":600000"), "{t}");
        // soonest first: 2 minutes before 10 minutes
        let first = t.find("\"label\":\"\"").expect(&t);
        let second = t.find("stretch").expect(&t);
        assert!(first < second, "{t}");
        let id: u64 = t.split("\"id\":").nth(1).unwrap().split(',').next().unwrap().parse().unwrap();
        assert!(w.cancel_timer(id));
        assert!(!w.cancel_timer(id));
        assert!(!w.timers().contains("\"label\":\"\""));
        assert!(w.timers().contains("stretch"));
        w.reply("cancel my timer");
        assert_eq!(w.timers(), "[]");
    }

    #[test]
    fn timers_reach_the_page_and_survive_a_reload() {
        let mut w = Web::new(bytes_cache(), "").unwrap();
        let r = w.reply("set a timer for 90 seconds");
        assert!(r.contains("\"timers\":[[") && r.contains(",90000]"), "{r}");
        // a new page (a new instance) built from the saved state still has it
        let mut w2 = Web::new(bytes_cache(), &w.state()).unwrap();
        let r = w2.reply("/timers");
        assert!(r.contains("\"timers\":[[") && r.contains(",90000]"), "{r}");
        assert!(w2.reply("cancel my timer").contains("\"timers\":[]"));
        // junk in the timer section is ignored
        assert!(Web::new(bytes_cache(), "\u{1}\u{1}x y z\n9 9\n99999999999999999999 5 a").is_ok());
    }

    #[test]
    fn hostile_input_does_not_panic() {
        let mut w = Web::new(bytes_cache(), "").unwrap();
        let long = "a".repeat(50_000);
        for s in [long.as_str(), "/", "//", "/ /", "/\u{202e}x", "[#> @", "9".repeat(400).as_str(), "((((((((((", "ünïcödé 😀 日本語", "\u{0}\u{1}\u{2}"] {
            let r = w.reply(s);
            assert!(r.starts_with('{') && r.ends_with('}'), "{r}");
        }
    }

    /// A wasm module gets a 1 MiB stack by default and overflowing it traps. Everything the page does (loading the
    /// model, building the index, answering hostile input) has to fit in half of that natively.
    #[test]
    fn fits_in_a_small_stack() {
        let bytes = bytes_cache().clone();
        let h = std::thread::Builder::new()
            .stack_size(512 * 1024)
            .spawn(move || {
                let mut w = Web::new(&bytes, "").unwrap();
                let deep = format!("what is 1+{}1{}", "(".repeat(140), ")".repeat(140));
                let minus = format!("what is {}1", "-".repeat(280));
                for s in [deep.as_str(), minus.as_str(), "hello", "/stats", "/why", "tell me a joke", "roll 3d6+2", "days until 2026-12-25"] {
                    assert!(w.reply(s).starts_with('{'));
                }
            })
            .unwrap();
        h.join().expect("overflowed a 512 KiB stack");
    }

    /// The model shipped in web/ (if present) must load against the built-in intents; this catches a stale bot.bin.
    #[test]
    fn shipped_model_matches_the_intents() {
        if let Ok(b) = std::fs::read("web/bot.bin") {
            let mut w = Web::new(&b, "").expect("web/bot.bin is stale: run ./build-web.sh");
            // and it should actually understand ordinary requests, not just load
            for q in ["hello", "tell me a joke", "thanks a lot", "what can you do"] {
                let r = w.reply(q);
                assert!(!r.contains("\"tag\":\"fallback\""), "{q}: {r}");
            }
        }
    }

    /// Every model the page can switch to (web/models/index.json) must be listed truthfully, load against the built-in
    /// intents, and run on a stack as small as a wasm module's, including on hostile input. A stale or half-copied
    /// model fails here rather than in a visitor's browser.
    #[test]
    fn shipped_models_match_the_manifest_and_the_intents() {
        let Ok(manifest) = std::fs::read_to_string("web/models/index.json") else { return };
        // one entry per line, written by build-web.sh: {"id":"bow-nano","arch":"bow","size":"nano","file":"models/bow-nano.bin","bytes":102266}
        let field = |line: &str, key: &str| -> Option<String> {
            let at = line.find(&format!("\"{key}\":"))? + key.len() + 3;
            let rest = line[at..].trim_start_matches('"');
            Some(rest.split(['"', ',', '}']).next()?.to_string())
        };
        let mut entries = Vec::new();
        for line in manifest.lines().filter(|l| l.contains("\"id\":")) {
            let line = line.trim_start_matches([',', ' ']);
            entries.push((field(line, "id").unwrap(), field(line, "arch").unwrap(), field(line, "size").unwrap(), field(line, "file").unwrap(), field(line, "bytes").unwrap().parse::<usize>().unwrap()));
        }
        assert!(entries.len() >= 3, "the model picker needs a few models to choose from: run ./build-web.sh");
        let mut archs = std::collections::BTreeSet::new();
        for (id, arch, size, file, bytes) in entries {
            assert_eq!(id, format!("{arch}-{size}"));
            assert!(file == "bot.bin" || file == format!("models/{id}.bin"), "{file}");
            let data = std::fs::read(format!("web/{file}")).unwrap_or_else(|_| panic!("{file} is listed in the manifest but missing"));
            assert_eq!(data.len(), bytes, "{file}: the manifest is out of date, run ./build-web.sh");
            let (arch2, size2, id2) = (arch.clone(), size.clone(), id.clone());
            let h = std::thread::Builder::new()
                .stack_size(512 * 1024)
                .spawn(move || {
                    let mut w = Web::new(&data, "").unwrap_or_else(|e| panic!("{id2} is stale ({e:?}): run ./build-web.sh"));
                    let info = w.model_info();
                    assert!(info.contains(&format!("\"arch\":\"{arch2}\"")) && info.contains(&format!("\"tier\":\"{size2}\"")), "{id2}: {info}");
                    assert!(w.reply("what is 12*(3+4)").contains("84"), "{id2}");
                    for q in ["hello", "tell me a joke", "thanks a lot"] {
                        let r = w.reply(q);
                        assert!(!r.contains("\"tag\":\"fallback\""), "{id2} {q}: {r}");
                    }
                    assert!(w.reply(&"x".repeat(4000)).starts_with('{'));
                })
                .unwrap();
            h.join().unwrap_or_else(|_| panic!("{id} failed or overflowed a 512 KiB stack"));
            archs.insert(arch);
        }
        assert_eq!(archs.len(), 3, "bow, cnn and gru should all be shipped: {archs:?}");
    }

    /// The same conversation tests/e2e.mjs runs against the real wasm; here natively so a wrong expectation
    /// shows up in `cargo test` rather than in CI.
    #[test]
    fn shipped_model_conversation() {
        let Ok(b) = std::fs::read("web/bot.bin") else { return };
        let mut w = Web::new(&b, "").unwrap();
        assert!(w.reply("what is 12*(3+4)").contains("84"));
        assert!(w.reply("10 km to miles").contains("6.21"));
        w.reply("my name is Zara");
        w.reply("note: buy stamps");
        let mut w = Web::new(&b, &w.state()).unwrap();
        assert!(w.reply("show my notes").contains("buy stamps"));
        assert!(w.reply("what is my name").contains("Zara"));
        assert!(w.reply("/help").contains("\"lines\":["));
        assert!(w.reply("/teach foo => bar").contains("native app"));
        assert!(w.reply("set a timer for 1 second").contains("Timer set"));
    }

    /// Drive the C ABI exactly as the page does.
    #[test]
    fn c_abi_round_trip() {
        use super::exports::*;
        fn put(b: &[u8]) -> (*mut u8, usize) {
            let p = tb_alloc(b.len());
            unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), p, b.len()) };
            (p, b.len())
        }
        fn result(n: usize) -> String {
            String::from_utf8(unsafe { std::slice::from_raw_parts(tb_result_ptr(), n) }.to_vec()).unwrap()
        }
        unsafe {
            // before init everything answers 0
            let (t, tl) = put(b"hi");
            assert_eq!(tb_reply(t, tl), 0);
            assert_eq!(tb_state(), 0);
            assert_eq!(tb_tick(), 0);
            assert_eq!(tb_timers(), 0);
            assert_eq!(tb_cancel_timer(1.0), 0);
            assert_eq!(tb_model_info(), 0);
            assert_eq!(tb_set_model(t, tl), -1);
            let (bad, bl) = put(b"nope");
            assert_eq!(tb_init(bad, bl, std::ptr::null(), 0), 1);
            tb_free(bad, bl);
            let m = bytes_cache();
            let (mp, ml) = put(m);
            assert_eq!(tb_init(mp, ml, std::ptr::null(), 0), 0);
            tb_free(mp, ml);
            let (q, ql) = put("what is 2+2".as_bytes());
            let n = tb_reply(q, ql);
            tb_free(q, ql);
            let r = result(n);
            assert!(r.contains("\"kind\":\"chat\"") && r.contains('4'), "{r}");
            let st = result(tb_state());
            assert!(st.contains('\u{1}'));
            let (e, el) = put("hello \u{1f600} \u{65e5}\u{672c}".as_bytes());
            assert!(result(tb_reply(e, el)).starts_with("{\"kind\":\"chat\""));
            tb_free(e, el);
            // invalid UTF-8 in is tolerated
            let (j, jl) = put(&[0xff, 0xfe, b'h', b'i']);
            assert!(tb_reply(j, jl) > 0);
            tb_free(j, jl);
            assert_eq!(result(tb_tick()), "[]");
            assert!(result(tb_model_info()).contains("\"arch\":\"bow\""));
            let (mp2, ml2) = put(bytes_cache());
            assert_eq!(tb_set_model(mp2, ml2), 0);
            tb_free(mp2, ml2);
            let (junk, jl) = put(b"not a model");
            assert_eq!(tb_set_model(junk, jl), 1);
            tb_free(junk, jl);
            let (tq, tql) = put(b"set a timer for 5 minutes");
            tb_reply(tq, tql);
            tb_free(tq, tql);
            let list = result(tb_timers());
            assert!(list.contains("\"left_ms\":"), "{list}");
            let id: f64 = list.split("\"id\":").nth(1).unwrap().split(',').next().unwrap().parse().unwrap();
            assert_eq!(tb_cancel_timer(id), 1);
            assert_eq!(tb_cancel_timer(-1.0), 0);
            assert_eq!(tb_cancel_timer(f64::NAN), 0);
            assert_eq!(result(tb_timers()), "[]");
            tb_free(t, tl);
        }
    }
}
