//! Brain = immutable, shareable (model + replies + phrase index).
//! Session = per-conversation mutable state. One Brain can serve many Sessions.

use crate::calc::{convert, convert_followup, fmt_num, try_calc, Conv};
use crate::art;
use crate::kb::{Hit, Kb};
use crate::facts::{self, Facts};
use crate::persona::{emotion_for, flavour, Persona};
use crate::intents::Intent;
use crate::model::{argmax, decide, Decision, Ensemble, Pred};
use crate::rng::{time_seed, Rng};
use crate::skills::*;
use crate::text::words;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

const MAX_INPUT_CHARS: usize = 400;
const GENERIC_FALLBACKS: [&str; 3] = [
    "Hmm, I don't know that one yet. Try rephrasing?",
    "I'm a tiny model, that went over my head.",
    "Not sure I follow. Ask me something else?",
];
const TEACH_FALLBACK: &str = "I don't have an answer for that. Use /teach to add one!";

pub enum Pending {
    None,
    Confirm { tag: usize, phrase: String },
    Riddle { answers: Vec<String>, shown: String, tries: u8 },
    Rps,
    Guess { target: i64, tries: u32 },
}

pub struct Turn {
    pub user: String,
    pub bot: String,
    pub tag: String,
}

const HISTORY: usize = 24;
const REPEATABLE: [&str; 11] = ["joke", "tell_fact", "riddle", "coin", "dice", "random_number", "quote", "favorite", "art", "skill:dice", "skill:random"];
const FLAVOURED: [&str; 16] = [
    "greeting", "goodbye", "thanks", "how_are_you", "mood_good", "mood_bad", "compliment", "insult", "joke", "bored", "sorry", "laugh", "and_you", "age", "ask_bond", "ask_energy",
];
const SULK_LINES: [&str; 5] = [
    "Hmph. I'm not talking to you until you apologise.",
    "...",
    "Still sulking. A sorry might help.",
    "I heard what you said earlier. Rude.",
    "Silent treatment. Say sorry or pay me a compliment.",
];

pub struct Session {
    pub mem: Memory,
    pub persona: Persona,
    pub debug: bool,
    pub can_learn: bool,
    pub last_input: Option<String>,
    pub last_pred: Option<Vec<f32>>,
    /// (topic, next line) of the last knowledge-base answer, so "more" can continue it
    pub kb_ctx: Option<(usize, usize)>,
    pub history: VecDeque<Turn>,
    /// previous turn's intent, drives `>@prev` replies and the transition prior
    pub prev: Option<String>,
    /// observed intent->intent transitions, learned from this and past sessions
    pub flow: HashMap<(String, String), f32>,
    /// true when a UI can announce timers while idle (the TUI); plain mode only reports them on your next message
    pub realtime: bool,
    pub timers: Vec<(Instant, String)>,
    extra_toasts: Vec<String>,
    ans: Option<f64>,
    last_conv: Option<Conv>,
    repeatable: Option<String>,
    last_pick: HashMap<String, usize>,
    rng: Rng,
    pending: Pending,
    rps_score: (u32, u32),
    rps_trans: [[u32; 3]; 3],
    rps_prev: Option<usize>,
}

impl Session {
    pub fn new(mut mem: Memory, can_learn: bool) -> Session {
        if let Some(m) = mem.facts.get("_tz").and_then(|v| v.parse::<i32>().ok()) {
            set_tz_override(Some(m));
        }
        if let Some(t) = mem.facts.get("_conf").and_then(|v| v.parse::<f32>().ok()) {
            crate::model::set_answer_threshold(t);
        }
        let persona = Persona::load(&mut mem);
        Session {
            mem,
            persona,
            debug: false,
            can_learn,
            last_input: None,
            last_pred: None,
            kb_ctx: None,
            history: VecDeque::new(),
            prev: None,
            flow: HashMap::new(),
            realtime: false,
            timers: Vec::new(),
            extra_toasts: Vec::new(),
            ans: None,
            last_conv: None,
            repeatable: None,
            last_pick: HashMap::new(),
            rng: Rng::new(time_seed()),
            pending: Pending::None,
            rps_score: (0, 0),
            rps_trans: [[0; 3]; 3],
            rps_prev: None,
        }
    }

    pub fn load_flow(&mut self, path: &str) {
        for line in std::fs::read_to_string(path).unwrap_or_default().lines().take(2000) {
            let w: Vec<&str> = line.split(' ').collect();
            if let [a, b, n] = w.as_slice() {
                if let Ok(n) = n.parse::<f32>() {
                    self.flow.insert((a.to_string(), b.to_string()), n.min(1000.0));
                }
            }
        }
    }

    pub fn save_flow(&self, path: &str) {
        let mut v: Vec<_> = self.flow.iter().collect();
        v.sort_by(|a, b| b.1.total_cmp(a.1));
        let out: String = v.iter().take(500).map(|((a, b), n)| format!("{a} {b} {n:.1}\n")).collect();
        let _ = std::fs::write(path, out);
    }

    /// Timers whose time has come (removes them).
    pub fn take_due(&mut self) -> Vec<String> {
        let now = Instant::now();
        let mut due = Vec::new();
        self.timers.retain(|(t, l)| {
            if *t <= now {
                due.push(if l.is_empty() { "Time's up!".to_string() } else { format!("Time's up: {l}") });
                false
            } else {
                true
            }
        });
        due
    }

    pub fn next_timer(&self) -> Option<(Duration, &str)> {
        let now = Instant::now();
        self.timers.iter().map(|(t, l)| (t.saturating_duration_since(now), l.as_str())).min_by_key(|x| x.0)
    }

    pub fn push_turn(&mut self, user: &str, bot: &str, tag: &str) {
        if self.history.len() >= HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(Turn { user: user.into(), bot: bot.into(), tag: tag.into() });
    }
}

pub struct Reply {
    pub text: String,
    pub tag: String,
    pub conf: f32,
    pub done: bool,
    pub learn: Option<(usize, String)>,
    pub debug: String,
    pub won: bool,
    pub toasts: Vec<String>,
    pub emo: &'static str,
    /// true when this reply re-ran an earlier request ("again")
    pub again: bool,
}

impl Reply {
    fn skill(text: String, tag: &str) -> Reply {
        Reply { text, tag: tag.into(), conf: 1.0, done: false, learn: None, debug: String::new(), won: false, toasts: Vec::new(), emo: "neutral", again: false }
    }
}

pub struct Brain {
    pub model: Ensemble,
    pub intents: Vec<Intent>,
    index: Vec<(usize, String, Vec<f32>)>,
    /// prior knowledge of which intent tends to follow which (from `>@prev` replies + a few built-ins)
    seed: HashMap<(String, String), f32>,
    pub facts: Facts,
    pub kb: Kb,
}

pub struct Analysis {
    pub probs: Vec<f32>,
    pub cov: f32,
    pub acts: Vec<f32>,
    pub saliency: Vec<(String, f32)>,
    pub micros: f32,
}

const SEED_FLOW: [(&str, &str); 22] = [
    ("how_are_you", "mood_good"), ("how_are_you", "mood_bad"), ("how_are_you", "and_you"), ("greeting", "how_are_you"),
    ("greeting", "greeting"), ("joke", "laugh"), ("joke", "again"), ("joke", "yes"), ("joke", "no"), ("joke", "insult"),
    ("tell_fact", "elaborate"), ("tell_fact", "again"), ("tell_fact", "compliment"), ("mood_bad", "yes"), ("mood_bad", "no"),
    ("mood_good", "and_you"), ("capabilities", "calc_help"), ("riddle", "again"), ("quote", "thanks"), ("compliment", "thanks"),
    ("insult", "sorry"), ("bored", "yes"),
];

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn yes_no(text: &str) -> Option<bool> {
    let w: Vec<String> = words(text).iter().map(|w| w.to_lowercase()).collect();
    if w.is_empty() || w.len() > 4 {
        return None;
    }
    const YES: [&str; 12] = ["yes", "yeah", "yep", "yup", "sure", "ok", "okay", "correct", "right", "y", "exactly", "indeed"];
    const NO: [&str; 6] = ["no", "nope", "nah", "n", "wrong", "negative"];
    if YES.contains(&w[0].as_str()) {
        Some(true)
    } else if NO.contains(&w[0].as_str()) {
        Some(false)
    } else {
        None
    }
}

/// lowercase words joined by single spaces
fn norm(s: &str) -> String {
    words(s).iter().map(|w| w.to_lowercase()).collect::<Vec<_>>().join(" ")
}

fn human(tag: &str) -> String {
    match tag {
        "skill:calc" => "math".into(),
        "skill:convert" => "unit conversions".into(),
        "skill:todo" => "your list".into(),
        "skill:date" => "dates".into(),
        "skill:dice" | "dice" => "dice".into(),
        "skill:memory" => "things about you".into(),
        "skill:notes" => "your notes".into(),
        "skill:kb" => "general knowledge".into(),
        "skill:timer" => "timers".into(),
        t => t.trim_start_matches("skill:").replace('_', " "),
    }
}

impl Brain {
    pub fn new(model: Ensemble, intents: Vec<Intent>) -> Brain {
        let mut index = Vec::new();
        for (k, it) in intents.iter().enumerate() {
            for e in &it.examples {
                index.push((k, e.clone(), model.embed(e)));
            }
        }
        let mut seed: HashMap<(String, String), f32> = HashMap::new();
        for (a, b) in SEED_FLOW {
            // "again" has no `>@prev` replies of its own to seed from, so weight it by hand
            *seed.entry((a.into(), b.into())).or_default() += if b == "again" { 4.0 } else { 1.0 };
        }
        // anything you can ask for again makes "again" / "another" a likely next message
        for t in REPEATABLE {
            *seed.entry((t.to_string(), "again".to_string())).or_default() += 4.0;
        }
        for it in &intents {
            for (prev, _) in &it.ctx {
                *seed.entry((prev.clone(), it.tag.clone())).or_default() += 2.0;
            }
        }
        let facts = facts::load();
        let pairs: Vec<(String, String)> = facts.items.iter().map(|f| (f.cat.clone(), f.text.clone())).collect();
        let kb = crate::kb::load(&pairs);
        Brain { model, intents, index, seed, facts, kb }
    }

    pub fn reload_kb(&mut self) {
        let pairs: Vec<(String, String)> = self.facts.items.iter().map(|f| (f.cat.clone(), f.text.clone())).collect();
        self.kb = crate::kb::load(&pairs);
    }

    fn kb_reply(&self, s: &mut Session, h: &Hit) -> Reply {
        let (mut t, next) = self.kb.answer(h);
        s.kb_ctx = next.map(|n| (h.entry, n));
        if h.coverage < 0.85 && !h.keyword_hit {
            t = format!("Closest thing I know: {t}");
        }
        if next.is_some() {
            t.push_str(" (say \"more\" to continue)");
        }
        let mut r = Reply::skill(t, "skill:kb");
        r.conf = h.coverage;
        r
    }

    pub fn tag_idx(&self, tag: &str) -> Option<usize> {
        self.model.tags.iter().position(|t| t == tag)
    }

    /// Re-weight intent probabilities by what usually follows the previous intent. Short, ambiguous
    /// utterances ("fine", "more", "yes") lean on it hard; long, specific ones barely at all.
    pub fn apply_flow(&self, s: &Session, probs: &mut [f32], n_words: usize) {
        let Some(prev) = s.prev.as_deref() else { return };
        // context breaks ties; it never overrides a read the network is sure about
        if probs.iter().cloned().fold(0.0, f32::max) >= 0.85 {
            return;
        }
        let k = probs.len() as f32;
        let w: Vec<f32> = self
            .model
            .tags
            .iter()
            .map(|t| {
                let key = (prev.to_string(), t.clone());
                0.05 + 0.5 * self.seed.get(&key).copied().unwrap_or(0.0) + 0.25 * s.flow.get(&key).copied().unwrap_or(0.0)
            })
            .collect();
        let mean = w.iter().sum::<f32>() / k;
        let alpha = if n_words <= 2 { 0.7 } else if n_words <= 4 { 0.45 } else { 0.15 };
        let mut sum = 0.0;
        for (p, w) in probs.iter_mut().zip(&w) {
            *p *= (w / mean).powf(alpha);
            sum += *p;
        }
        probs.iter_mut().for_each(|p| *p /= sum);
    }

    /// Closest known phrase of a given intent, by cosine over pooled embeddings.
    pub fn nearest(&self, text: &str, tag: usize) -> Option<(&str, f32)> {
        let q = self.model.embed(text);
        let n = self.model.nets.len() as f32;
        self.index
            .iter()
            .filter(|(k, _, _)| *k == tag)
            .max_by(|a, b| dot(&q, &a.2).total_cmp(&dot(&q, &b.2)))
            .map(|(_, p, e)| (p.as_str(), dot(&q, e) / n))
    }

    pub fn top3(&self, p: &[f32]) -> String {
        let mut idx: Vec<usize> = (0..p.len()).collect();
        idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
        idx.iter().take(3).map(|&i| format!("{} {:.2}", self.model.tags[i], p[i])).collect::<Vec<_>>().join(", ")
    }

    /// Leave-one-word-out saliency: how much does each word support the top intent?
    pub fn explain(&self, text: &str) -> Vec<(String, f32)> {
        let ws: Vec<&str> = text.split_whitespace().collect();
        let full = self.model.predict(text);
        let k = argmax(&full.probs);
        (0..ws.len())
            .map(|i| {
                let rest: Vec<&str> = ws.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, w)| *w).collect();
                let p = self.model.predict(&rest.join(" "));
                (ws[i].to_string(), full.probs[k] - p.probs[k])
            })
            .collect()
    }

    /// Everything the TUI's brain panel draws, in one pass.
    pub fn analyze(&self, s: &Session, text: &str) -> Analysis {
        let t = std::time::Instant::now();
        let mut pred = self.model.predict(text);
        self.apply_flow(s, &mut pred.probs, words(text).len());
        let micros = t.elapsed().as_secs_f32() * 1e6;
        Analysis { probs: pred.probs, cov: pred.cov, acts: self.model.activations(text), saliency: self.explain(text), micros }
    }

    fn pick(&self, s: &mut Session, key: &str, options: &[String]) -> String {
        if options.is_empty() {
            return "...".into();
        }
        let mut i = s.rng.below(options.len());
        if options.len() > 1 && s.last_pick.get(key) == Some(&i) {
            i = (i + 1) % options.len();
        }
        s.last_pick.insert(key.to_string(), i);
        options[i].clone()
    }

    fn fill(&self, s: &mut Session, tpl: &str) -> String {
        let mut out = tpl.to_string();
        if !out.contains('{') {
            return out;
        }
        let p = &s.persona;
        let reps: [(&str, String); 23] = [
            ("{name}", s.mem.name().unwrap_or("friend").to_string()),
            ("{time}", clock()),
            ("{greet}", greet_word().to_string()),
            ("{daypart}", daypart().to_string()),
            ("{params}", { let n = self.model.nominal_params(); if n >= 1_000_000 { format!("{:.2}M", n as f32 / 1e6) } else { format!("{}k", n / 1000) } }),
            ("{tz}", tz_label()),
            ("{date}", fmt_date(today())),
            ("{weekday}", weekday(today()).to_string()),
            ("{mood}", p.mood_label().to_string()),
            ("{level}", p.level().to_string()),
            ("{title}", p.title().to_string()),
            ("{xp}", p.xp.to_string()),
            ("{streak}", p.streak.to_string()),
            ("{msgs}", p.msgs.to_string()),
            ("{dex}", p.dex.len().to_string()),
            ("{dex_total}", self.intents.len().to_string()),
            ("{known_days}", p.known_days().to_string()),
            ("{bond_label}", p.bond_label().to_string()),
            ("{bond}", p.bond.to_string()),
            ("{hearts}", format!("{}{}", "♥".repeat(p.hearts()), "♡".repeat(5 - p.hearts()))),
            ("{energy}", format!("{:.0}", p.energy * 100.0)),
            ("{coin}", String::new()),
            ("{dice}", String::new()),
        ];
        for (k, v) in reps {
            if out.contains(k) {
                let v = match k {
                    "{coin}" => (if s.rng.below(2) == 0 { "Heads" } else { "Tails" }).to_string(),
                    "{dice}" => s.rng.range(1, 6).to_string(),
                    _ => v,
                };
                out = out.replace(k, &v);
            }
        }
        if out.contains("{rand}") {
            out = out.replace("{rand}", &s.rng.range(1, 100).to_string());
        }
        out
    }

    fn fallback(&self, s: &mut Session) -> String {
        let mut f: Vec<String> = GENERIC_FALLBACKS.iter().map(|x| x.to_string()).collect();
        if s.can_learn {
            f.push(TEACH_FALLBACK.to_string());
        }
        self.pick(s, "fallback", &f)
    }

    /// `>@prev` replies win over generic ones; `=other` borrows another intent's replies.
    fn answer(&self, s: &mut Session, k: usize, text: &str) -> (String, String) {
        let tag = self.model.tags[k].clone();
        let mut ctx_tag = tag.clone();
        match tag.as_str() {
            "tell_name" => match extract_name(text) {
                Some(n) => {
                    s.mem.set("name", &n);
                }
                None => return ("Nice to meet you! What should I call you?".into(), tag),
            },
            "ask_my_name" if s.mem.name().is_none() => return ("You haven't told me yet. What's your name?".into(), tag),
            "tell_fact" => {
                if let Some(t) = self.fact_reply(s, text) {
                    return (t, tag);
                }
            }
            "art" => return (art::draw(text, &mut s.rng), tag),
            "recall_user" => {
                let t = match s.history.back() {
                    Some(h) => format!("Your last message was: \"{}\"", h.user),
                    None => "Nothing yet, we've only just started.".into(),
                };
                return (t, tag);
            }
            "recall_bot" => {
                let t = match s.history.back() {
                    Some(h) => format!("I said: \"{}\"", h.bot),
                    None => "I haven't said anything yet.".into(),
                };
                return (t, tag);
            }
            "recap" => {
                let mut seen: Vec<String> = Vec::new();
                for h in &s.history {
                    let t = human(&h.tag);
                    if !["fallback", "clarify", "recap", "recall user", "recall bot", "again"].contains(&t.as_str()) && !seen.contains(&t) {
                        seen.push(t);
                    }
                }
                let t = if seen.is_empty() {
                    "We haven't covered anything yet.".to_string()
                } else {
                    format!("So far we've covered: {} ({} messages in my short-term memory).", seen.join(", "), s.history.len())
                };
                return (t, tag);
            }
            "riddle" => {
                let opts = self.intents[k].responses.clone();
                let raw = self.pick(s, "riddle", &opts);
                if let Some((q, a)) = raw.split_once('|') {
                    let answers: Vec<String> = a.split('/').map(|x| norm(x)).filter(|x| !x.is_empty()).collect();
                    let shown = a.split('/').next().unwrap_or("").trim().to_string();
                    s.pending = Pending::Riddle { answers, shown, tries: 0 };
                    return (format!("{} (say \"give up\" to skip)", q.trim()), tag);
                }
            }
            "play_rps" => {
                s.pending = Pending::Rps;
                s.rps_score = (0, 0);
            }
            "play_guess" => {
                s.pending = Pending::Guess { target: s.rng.range(1, 100), tries: 0 };
            }
            _ => {}
        }
        let it = &self.intents[k];
        let ctx_opts: Vec<String> = it.ctx.iter().filter(|(p, _)| s.prev.as_deref() == Some(p.as_str())).map(|(_, r)| r.clone()).collect();
        let opts = if ctx_opts.is_empty() { it.responses.clone() } else { ctx_opts };
        let mut tpl = self.pick(s, &tag, &opts);
        if let Some(other) = tpl.strip_prefix('=').map(|x| x.trim().to_string()) {
            if let Some(ok) = self.tag_idx(&other) {
                let o = self.intents[ok].responses.clone();
                tpl = self.pick(s, &other, &o);
                ctx_tag = other;
            }
        }
        let mut out = self.fill(s, &tpl);
        if FLAVOURED.contains(&ctx_tag.as_str()) && s.rng.f32() < 0.35 {
            let pick = s.rng.below(3);
            out = format!("{}{}", flavour(s.persona.mood_label(), pick), out);
        } else if FLAVOURED.contains(&ctx_tag.as_str()) && s.persona.tired() && s.rng.f32() < 0.4 {
            out = format!("*yawn* {out}");
        }
        (out, ctx_tag)
    }

    fn fact_reply(&self, s: &mut Session, text: &str) -> Option<String> {
        let low = text.to_lowercase();
        let cat = facts::detect_category(text);
        let f = if low.contains("of the day") { self.facts.daily(today()) } else { self.facts.pick(cat, &s.persona.factbook, &mut s.rng) }?;
        let (id, line) = (f.id.clone(), format!("{} fact: {}", crate::skills::capitalize(if f.cat == "animals" { "animal" } else { &f.cat }), f.text));
        if s.persona.factbook.insert(id) {
            s.persona.add_xp(2, &mut s.extra_toasts);
            s.extra_toasts.push(format!("Fact book: {}/{} collected", s.persona.factbook.len(), self.facts.items.len()));
        }
        Some(line)
    }

    fn timer_cmd(&self, s: &mut Session, c: TimerCmd) -> Reply {
        let text = match c {
            TimerCmd::Set(0, _) => "Pick a time between 1 second and 24 hours.".to_string(),
            TimerCmd::Set(secs, label) => {
                s.timers.push((Instant::now() + Duration::from_secs(secs), label.clone()));
                let tail = if s.realtime { "" } else { " (plain mode: I'll announce it on your next message)" };
                let lab = if label.is_empty() { String::new() } else { format!(" for \"{label}\"") };
                format!("Timer set for {}{lab}.{tail}", fmt_dur(secs))
            }
            TimerCmd::Cancel => {
                let n = s.timers.len();
                s.timers.clear();
                if n == 0 { "No timers running.".to_string() } else { format!("Cancelled {n} timer(s).") }
            }
            TimerCmd::Status => match s.next_timer() {
                Some((d, l)) => format!("{} left{}.", fmt_dur(d.as_secs()), if l.is_empty() { String::new() } else { format!(" on \"{l}\"") }),
                None => "No timers running.".to_string(),
            },
        };
        Reply::skill(text, "skill:timer")
    }

    fn pending_turn(&self, s: &mut Session, text: &str) -> Option<Reply> {
        let low = norm(text);
        let n_words = low.split(' ').filter(|w| !w.is_empty()).count();
        match std::mem::replace(&mut s.pending, Pending::None) {
            Pending::None => None,
            Pending::Confirm { tag, phrase } => match yes_no(text) {
                Some(true) => {
                    let (t, ct) = self.answer(s, tag, &phrase);
                    let mut r = Reply::skill(t, &ct);
                    if s.can_learn {
                        r.learn = Some((tag, phrase));
                    }
                    Some(r)
                }
                Some(false) => Some(Reply::skill("Okay, my mistake. Could you rephrase it?".into(), "clarify_no")),
                None => None,
            },
            Pending::Riddle { answers, shown, tries } => {
                if n_words > 8 {
                    // off-topic: answer it normally, but keep the riddle alive
                    s.pending = Pending::Riddle { answers, shown, tries };
                    return None;
                }
                let padded = format!(" {low} ");
                if answers.iter().any(|a| padded.contains(&format!(" {a} "))) {
                    let mut r = Reply::skill("Correct! Nice one.".into(), "riddle_win");
                    r.won = true;
                    return Some(r);
                }
                if ["give up", "idk", "i dont know", "skip", "dont know", "no idea", "tell me"].iter().any(|g| low.contains(g)) || tries >= 2 {
                    return Some(Reply::skill(format!("It was: {shown}."), "riddle_reveal"));
                }
                s.pending = Pending::Riddle { answers, shown, tries: tries + 1 };
                Some(Reply::skill("Not quite, try again!".into(), "riddle_retry"))
            }
            Pending::Rps => {
                if ["stop", "quit", "enough", "exit", "done", "end"].iter().any(|w| low.split(' ').any(|x| x == *w)) {
                    let (a, b) = s.rps_score;
                    let verdict = if a > b { "You win the match!" } else if a < b { "I win the match!" } else { "A draw." };
                    let mut r = Reply::skill(format!("Final score, you {a} - bot {b}. {verdict}"), "rps_end");
                    r.won = a > b;
                    return Some(r);
                }
                let mv = ["rock", "paper", "scissors"].iter().position(|m| low.split(' ').any(|w| w == *m || (w.len() > 3 && m.starts_with(w))));
                let Some(you) = mv else {
                    // not a move: answer it normally, but keep the match going
                    s.pending = Pending::Rps;
                    return None;
                };
                // adaptive opponent: predict your next move from your move-to-move habits, then counter it
                let predicted = match s.rps_prev {
                    Some(p) if s.rng.f32() < 0.6 && s.rps_trans[p].iter().sum::<u32>() > 1 => {
                        s.rps_trans[p].iter().enumerate().max_by_key(|(_, c)| **c).map(|(i, _)| i)
                    }
                    _ => None,
                };
                let bot = match predicted {
                    Some(pm) => (pm + 1) % 3,
                    None => s.rng.below(3),
                };
                if let Some(p) = s.rps_prev {
                    s.rps_trans[p][you] += 1;
                }
                s.rps_prev = Some(you);
                let names = ["rock", "paper", "scissors"];
                let outcome = match (you + 3 - bot) % 3 {
                    0 => "Tie.",
                    1 => {
                        s.rps_score.0 += 1;
                        "You win this round."
                    }
                    _ => {
                        s.rps_score.1 += 1;
                        "I win this round."
                    }
                };
                s.pending = Pending::Rps;
                Some(Reply::skill(
                    format!("I picked {}. {outcome} (you {} - bot {}) Go again or say stop.", names[bot], s.rps_score.0, s.rps_score.1),
                    "rps_round",
                ))
            }
            Pending::Guess { target, tries } => {
                if low.contains("give up") {
                    return Some(Reply::skill(format!("It was {target}."), "guess_reveal"));
                }
                // a guess is a short message with exactly one number and no operators ("50", "is it 50?");
                // "what is 2+2" or "add 3 to my list" are other requests and must not eat a guess
                let guess = crate::calc::lex(text).and_then(|t| {
                    let nums: Vec<f64> = t.iter().filter_map(|x| if let crate::calc::Tok::Num(n) = x { Some(*n) } else { None }).collect();
                    let ops = t.iter().any(|x| matches!(x, crate::calc::Tok::Sym(c) if "+*/^%!".contains(*c)));
                    (nums.len() == 1 && !ops && n_words <= 5).then(|| nums[0])
                });
                let Some(g) = guess else {
                    s.pending = Pending::Guess { target, tries };
                    return None;
                };
                let tries = tries + 1;
                if g == target as f64 {
                    let note = if tries <= 7 { "That's within the 7 guesses binary search needs." } else { "Binary search would have taken 7 at most." };
                    let mut r = Reply::skill(format!("Correct, it was {target}! {tries} guesses. {note}"), "guess_win");
                    r.won = true;
                    return Some(r);
                }
                if tries >= 10 {
                    return Some(Reply::skill(format!("Out of guesses, it was {target}."), "guess_lose"));
                }
                s.pending = Pending::Guess { target, tries };
                Some(Reply::skill((if g < target as f64 { "Higher." } else { "Lower." }).to_string(), "guess_hint"))
            }
        }
    }

    /// "double that", "half of that", "square it": arithmetic on the previous result.
    fn on_answer(&self, s: &mut Session, text: &str) -> Option<Reply> {
        let a = s.ans?;
        let low = norm(text);
        let v = match low.as_str() {
            "double that" | "double it" | "twice that" => a * 2.0,
            "triple that" | "triple it" => a * 3.0,
            "half of that" | "half that" | "halve that" | "half of it" | "halve it" => a / 2.0,
            "square that" | "square it" | "squared" => a * a,
            "cube that" | "cube it" => a * a * a,
            "negate that" | "flip the sign" => -a,
            "root of that" | "square root of that" | "sqrt that" if a >= 0.0 => a.sqrt(),
            _ => return None,
        };
        s.ans = Some(v);
        Some(Reply::skill(format!("= {}", fmt_num(v)), "skill:calc"))
    }

    pub fn reply(&self, s: &mut Session, input: &str) -> Reply {
        let text: String = input.trim().chars().take(MAX_INPUT_CHARS).collect();
        let mut r = self.reply_inner(s, &text, 0);
        let prev = s.prev.clone();
        // no XP, toasts or flavour on a safety message: it isn't a game
        let mut toasts = if r.tag == "crisis" { Vec::new() } else { s.persona.on_turn(&r.tag, self.intents.len(), local_hour()) };
        if r.won {
            s.persona.win(&mut toasts);
        }
        if r.tag == "insult" {
            s.persona.insult(&mut toasts);
            if s.persona.sulking > 0 {
                r.text.push_str(" That's the last straw. I'm not talking to you until you apologise.");
                s.pending = Pending::None;
            }
        }
        if r.tag == "forgive" {
            s.persona.forgive(&mut toasts);
        }
        // learn the conversational flow from confident intent answers
        if let Some(p) = prev {
            if !r.tag.starts_with("skill:") && !["fallback", "clarify", "clarify_no", "sulk"].contains(&r.tag.as_str()) && r.conf > 0.5 {
                *s.flow.entry((p, r.tag.clone())).or_default() += 1.0;
            }
        }
        if r.tag != "skill:kb" {
            s.kb_ctx = None;
        }
        if REPEATABLE.contains(&r.tag.as_str()) && !r.again {
            s.repeatable = Some(text.clone());
        }
        s.prev = if r.tag == "fallback" || r.tag == "clarify" { None } else { Some(r.tag.clone()) };
        s.push_turn(&text, &r.text, &r.tag);
        toasts.append(&mut s.extra_toasts);
        if !s.realtime {
            for t in s.take_due() {
                toasts.push(t);
            }
        }
        if toasts.iter().any(|t| t.starts_with("LEVEL") || t.starts_with("Achievement")) {
            r.emo = "proud";
        } else {
            r.emo = emotion_for(&r.tag, s.persona.mood_label());
        }
        r.toasts = toasts;
        r
    }

    fn reply_inner(&self, s: &mut Session, text: &str, depth: u8) -> Reply {
        s.last_input = Some(text.to_string());

        // deterministic safety net: runs before the network, games, sulking and every skill
        if crisis_text(text) {
            if let Some(k) = self.tag_idx("crisis") {
                s.pending = Pending::None;
                let (t, _) = self.answer(s, k, text);
                let mut r = Reply::skill(t, "crisis");
                r.debug = "[keyword safety net]".into();
                return r;
            }
        }

        // classify first: crisis messages and the sulk gate both need the neural read of this input
        let n_words = words(text).len();
        let mut pred: Pred = self.model.predict(text);
        self.apply_flow(s, &mut pred.probs, n_words);
        let conf = pred.probs.iter().cloned().fold(0.0, f32::max);
        let decision = decide(&pred.probs, pred.cov);
        let top_tag = match decision {
            Decision::Answer(k) | Decision::Clarify(k) => Some(self.model.tags[k].as_str()),
            Decision::Reject => None,
        };
        let debug = format!("[{} | cov {:.2}]", self.top3(&pred.probs), pred.cov);
        s.last_pred = Some(pred.probs.clone());

        // safety first: never gated by sulking, clarification or games. This is the network's second opinion
        // after the keyword net above, so it must not fire on skill-style input: numbers ("10 km to miles")
        // pushed a small model into the crisis class once, so digits and very short inputs are excluded
        // and it needs more than a vague hunch.
        if let Decision::Answer(k) | Decision::Clarify(k) = decision {
            if self.model.tags[k] == "crisis" && pred.probs[k] >= 0.35 && n_words >= 3 && !text.chars().any(|c| c.is_ascii_digit()) {
                s.pending = Pending::None;
                let (t, _) = self.answer(s, k, text);
                let mut r = Reply::skill(t, "crisis");
                r.conf = conf;
                r.debug = debug;
                return r;
            }
        }

        if s.persona.sulking > 0 {
            if matches!(top_tag, Some("sorry" | "compliment" | "thanks")) {
                let mut r = Reply::skill("...Fine. Apology accepted. Just don't do it again.".into(), "forgive");
                r.conf = conf;
                return r;
            }
            let i = s.rng.below(SULK_LINES.len());
            let mut r = Reply::skill(SULK_LINES[i].into(), "sulk");
            r.conf = conf;
            return r;
        }

        if let Some(r) = self.pending_turn(s, text) {
            return r;
        }
        // note: pending_turn keeps a game alive on off-topic input and drops a stale confirmation itself

        // deterministic skills first: high-precision, no training data needed
        if let Some(r) = self.on_answer(s, text) {
            return r;
        }
        if let Some(t) = try_todo(text, &mut s.mem) {
            return Reply::skill(t, "skill:todo");
        }
        if let Some(c) = try_timer(text) {
            return self.timer_cmd(s, c);
        }
        if let Some(t) = try_birthday(text, &mut s.mem) {
            return Reply::skill(t, "skill:memory");
        }
        if let Some(t) = try_about_me(text, &s.mem) {
            return Reply::skill(t, "skill:memory");
        }
        if let Some(t) = try_date(text) {
            return Reply::skill(t, "skill:date");
        }
        if let Some(last) = s.last_conv.clone() {
            match convert_followup(text, &last) {
                Some(Ok((c, line))) => {
                    s.ans = Some(c.out);
                    s.last_conv = Some(c);
                    return Reply::skill(line, "skill:convert");
                }
                Some(Err(e)) => return Reply::skill(e, "skill:convert"),
                None => {}
            }
        }
        match convert(text) {
            Some(Ok((c, line))) => {
                s.ans = Some(c.out);
                s.last_conv = Some(c);
                return Reply::skill(line, "skill:convert");
            }
            Some(Err(e)) => return Reply::skill(e, "skill:convert"),
            None => {}
        }
        if let Some(t) = try_dice(text, &mut s.rng) {
            return Reply::skill(t, "skill:dice");
        }
        if let Some(t) = try_random(text, &mut s.rng) {
            return Reply::skill(t, "skill:random");
        }
        if !has_iso_date(text) {
            match try_calc(text, s.ans) {
                Some(Ok(v)) => {
                    s.ans = Some(v);
                    return Reply::skill(format!("= {}", fmt_num(v)), "skill:calc");
                }
                Some(Err(e)) => return Reply::skill(e, "skill:calc"),
                None => {}
            }
        }
        if let Some(t) = try_facts(text, &mut s.mem) {
            return Reply::skill(t, "skill:memory");
        }
        if let Some(t) = try_notes(text, &mut s.mem) {
            return Reply::skill(t, "skill:notes");
        }
        // "fact of the day" is explicit enough not to leave to the network
        if text.to_lowercase().contains("fact of the day") {
            if let Some(t) = self.fact_reply(s, text) {
                return Reply::skill(t, "tell_fact");
            }
        }

        // knowledge base: "more" continues the last answer; open questions the router isn't sure about land here
        if let Some((e, i)) = s.kb_ctx {
            let wants_more = matches!(norm(text).as_str(), "more" | "tell me more" | "go on" | "continue" | "what else" | "keep going" | "more please" | "elaborate" | "go ahead")
                || matches!(decision, Decision::Answer(k) if self.model.tags[k] == "elaborate");
            if wants_more {
                if let Some((mut t, next)) = self.kb.more(e, i) {
                    s.kb_ctx = next.map(|n| (e, n));
                    if next.is_some() {
                        t.push_str(" (say \"more\" to continue)");
                    }
                    return Reply::skill(t, "skill:kb");
                }
            }
        }
        let router_sure = matches!(decision, Decision::Answer(_)) && conf >= 0.85 && pred.cov >= 0.85;
        if !router_sure {
            if let Some(h) = self.kb.lookup(text) {
                let need = if matches!(decision, Decision::Answer(_)) { 0.85 } else { crate::kb::MIN_COVERAGE };
                if h.coverage >= need {
                    let mut r = self.kb_reply(s, &h);
                    r.debug = debug;
                    return r;
                }
            }
        }

        // neural router
        let mut r = match decision {
            Decision::Answer(k) if self.model.tags[k] == "again" => match s.repeatable.clone() {
                Some(t) if depth == 0 => {
                    let mut inner = self.reply_inner(s, &t, depth + 1);
                    inner.conf = conf;
                    inner.again = true;
                    return inner;
                }
                _ => Reply::skill("Again what? We haven't done anything repeatable yet.".into(), "again"),
            },
            Decision::Answer(k) => {
                let (t, ctx_tag) = self.answer(s, k, text);
                let mut r = Reply::skill(t, &ctx_tag);
                r.done = self.model.tags[k] == "goodbye";
                r
            }
            Decision::Clarify(k) => match self.nearest(text, k) {
                // only ask when the closest known phrase is genuinely close: a nonsense
                // "did you mean...?" is worse than admitting it doesn't know
                Some((ph, sim)) if sim >= self.model.clarify_sim => {
                    let msg = format!("Did you mean something like \"{ph}\"? (yes/no)");
                    s.pending = Pending::Confirm { tag: k, phrase: text.to_string() };
                    Reply::skill(msg, "clarify")
                }
                _ => Reply::skill(self.fallback(s), "fallback"),
            },
            Decision::Reject => Reply::skill(self.fallback(s), "fallback"),
        };
        r.conf = conf;
        r.debug = debug;
        r
    }
}
