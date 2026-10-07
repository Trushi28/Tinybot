//! Slash commands shared by the plain REPL and the TUI. Output is returned as lines so each
//! front-end decides how to show it.

use crate::bot::{Brain, Session};
use crate::intents;
use crate::train::{self, MODEL_PATH};
use std::sync::mpsc;
use std::time::Instant;

pub struct App {
    pub brain: Brain,
    pub s: Session,
    /// columns available for multi-column output like /dex
    pub width: usize,
    retrain_rx: Option<mpsc::Receiver<(Brain, String)>>,
    retrain_again: bool,
    /// where the model lives on disk, so relearning overwrites the right file
    pub model_path: String,
}

impl App {
    pub fn new(brain: Brain, s: Session) -> App {
        App { brain, s, width: 80, retrain_rx: None, retrain_again: false, model_path: MODEL_PATH.to_string() }
    }

    pub fn retraining(&self) -> bool {
        self.retrain_rx.is_some()
    }

    /// Relearn from intents.txt + learned.txt on a background thread (~10-30 s on one core),
    /// so the UI keeps animating. A second request while one is running queues exactly one more.
    pub fn start_retrain(&mut self) {
        if self.retrain_rx.is_some() {
            self.retrain_again = true;
            return;
        }
        let (temp, scfg, clarify) = (self.brain.model.temp, self.brain.model.cfg(), self.brain.model.clarify_sim);
        let save_to = self.model_path.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let t0 = Instant::now();
            let intents = intents::load();
            let m = train::quick_student(&intents, temp, scfg, clarify);
            let _ = m.save(&save_to);
            let _ = tx.send((Brain::new(m, intents), format!("relearned in {:.1?}", t0.elapsed())));
        });
        self.retrain_rx = Some(rx);
    }

    /// Non-blocking: Some(message) when a retrain just finished and the new brain is live.
    pub fn poll_retrain(&mut self) -> Option<String> {
        let res = match self.retrain_rx.as_ref()?.try_recv() {
            Ok(r) => Some(r),
            Err(mpsc::TryRecvError::Empty) => return None,
            Err(mpsc::TryRecvError::Disconnected) => None,
        };
        self.retrain_rx = None;
        let msg = match res {
            Some((b, msg)) => {
                self.brain = b;
                msg
            }
            None => "relearning failed".to_string(),
        };
        if std::mem::take(&mut self.retrain_again) {
            self.start_retrain();
        }
        Some(msg)
    }

    /// Blocking variant for the plain REPL.
    pub fn wait_retrain(&mut self) -> Option<String> {
        self.retrain_rx.as_ref()?;
        loop {
            if let Some(m) = self.poll_retrain() {
                return Some(m);
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
}

pub struct Out {
    pub lines: Vec<String>,
    pub quit: bool,
}

fn out(lines: Vec<String>) -> Out {
    Out { lines, quit: false }
}

pub const HELP: &[&str] = &[
    "/dex        which of my intents you've discovered",
    "/facts      your fact book (what you've heard)",
    "/memory     everything I remember about you",
    "/notes      your notes",
    "/timers     running timers",
    "/clear      clear this console",
    "/tz <+5:30> set your time zone (or /tz auto)",
    "/confidence <0.3-0.95>  how sure I must be to answer (higher = fewer mistakes, more questions)",
    "/unlearn    undo the last thing I learned",
    "/emote <x>  make a face (try: love anger wow sleepy proud)",
    "/stats      level, XP, badges, model size and speed",
    "/model      which brain I'm running (architecture and size)",
    "/history    my short-term memory of this chat",
    "/why        which words drove my last decision",
    "/fix <tag>  my last answer was wrong, it meant <tag>",
    "/teach a => b   brand-new intent: when you say a, I reply b",
    "/intents    all intents with phrase counts",
    "/bench      measure my inference speed",
    "/anim       toggle animations (TUI)",
    "/debug      toggle raw scores (plain mode)",
    "/forget     wipe remembered facts and list",
    "/quit",
    "try: 12*(3+4)  20% of 80  10 km to miles  and in meters?  double that",
    "     roll 3d6+2  days until 2026-12-25  my favorite color is teal",
    "     add milk to my list  riddle me this  play rock paper scissors",
];

fn bar(frac: f32, w: usize) -> String {
    let n = (frac.clamp(0.0, 1.0) * w as f32).round() as usize;
    format!("{}{}", "█".repeat(n), "░".repeat(w - n))
}

pub fn run(app: &mut App, line: &str) -> Out {
    let cmd = line.trim_start_matches('/');
    let (name, arg) = cmd.split_once(' ').map(|(a, b)| (a, b.trim())).unwrap_or((cmd, ""));
    let (brain, s) = (&mut app.brain, &mut app.s);
    match name {
        "quit" | "exit" | "q" => Out { lines: vec!["bye!".into()], quit: true },
        "help" | "?" => out(HELP.iter().map(|x| x.to_string()).collect()),
        "debug" => {
            s.debug = !s.debug;
            out(vec![format!("debug {}", if s.debug { "on" } else { "off" })])
        }
        "forget" => {
            s.mem.clear();
            out(vec!["memory wiped (facts, list, XP and dex)".into()])
        }
        "clear" => out(vec![]),
        "memory" => out(match crate::skills::try_about_me("what do you know about me", &s.mem) {
            Some(t) => t.lines().map(String::from).collect(),
            None => vec![],
        }),
        "notes" => out(if s.mem.notes.is_empty() {
            vec!["no notes yet, try: note: buy stamps".into()]
        } else {
            s.mem.notes.iter().enumerate().map(|(i, n)| format!("{}. {n}", i + 1)).collect()
        }),
        "timers" => out(match s.next_timer() {
            Some((d, l)) => vec![format!("next: {} {l}", crate::skills::fmt_dur(d.as_secs())), format!("{} running", s.timers.len())],
            None => vec!["no timers running".into()],
        }),
        "facts" => {
            let mut v = vec![format!("fact book: {}/{} collected", s.persona.factbook.len(), brain.facts.items.len())];
            for c in brain.facts.categories() {
                let all: Vec<_> = brain.facts.items.iter().filter(|f| f.cat == c).collect();
                let got = all.iter().filter(|f| s.persona.factbook.contains(&f.id)).count();
                v.push(format!("{c:<9} [{}{}] {got}/{}", "#".repeat(got), ".".repeat(all.len() - got), all.len()));
            }
            out(v)
        }
        "intents" => out(brain.intents.iter().map(|it| format!("{:<18} {} phrases", it.tag, it.examples.len())).collect()),
        "dex" => {
            let per_row = (app.width / 17).max(1);
            let mut v = vec![format!("intent dex: {}/{} discovered", s.persona.dex.len(), brain.intents.len())];
            for chunk in brain.intents.chunks(per_row) {
                v.push(chunk.iter().map(|it| format!("{:<17}", if s.persona.dex.contains(&it.tag) { it.tag.as_str() } else { "???" })).collect::<String>().trim_end().to_string());
            }
            out(v)
        }
        "stats" => {
            let p = &s.persona;
            let (a, b) = p.progress();
            let (us, _) = train::bench(&brain.model);
            let badges = if p.badges.is_empty() { "none yet".to_string() } else { p.badges.iter().cloned().collect::<Vec<_>>().join(", ") };
            out(vec![
                format!("level {} {}  [{}] {a}/{b} XP", p.level(), p.title(), bar(a as f32 / b as f32, 12)),
                format!("{} messages, {}-day streak, {} game wins, mood {}", p.msgs, p.streak, p.wins, p.mood_label()),
                format!("dex {}/{}, badges: {badges}", p.dex.len(), brain.intents.len()),
                format!("model: {} params, {:.0} KB on disk, {us:.0} us/query", train::params_label(&brain.model), train::model_kb(&brain.model)),
            ])
        }
        "history" => {
            if s.history.is_empty() {
                return out(vec!["nothing yet".into()]);
            }
            let skip = s.history.len().saturating_sub(10);
            out(s.history.iter().skip(skip).map(|t| format!("you: {}  |  bot[{}]: {}", t.user, t.tag, t.bot.chars().take(60).collect::<String>())).collect())
        }
        "why" => match s.last_input.clone() {
            Some(t) => {
                let mut ex = brain.explain(&t);
                ex.sort_by(|a, b| b.1.total_cmp(&a.1));
                out(vec![format!("word influence: {}", ex.iter().map(|(w, d)| format!("{w} ({d:+.2})")).collect::<Vec<_>>().join("  "))])
            }
            None => out(vec!["nothing to explain yet".into()]),
        },
        "model" => {
            let c = brain.model.cfg();
            out(vec![
                format!("{} ({})", train::model_name(&brain.model), c.tag()),
                format!("{} params allocated, {} trained, {:.0} KB on disk", train::params_label(&brain.model), train::fmt_params(brain.model.param_report().2), train::model_kb(&brain.model)),
                format!("available: {}", {
                    let mut v: Vec<String> = std::fs::read_dir("models").map(|d| d.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().trim_end_matches(".bin").to_string()).collect()).unwrap_or_default();
                    v.sort();
                    if v.is_empty() { "none built yet (tinybot train --all)".to_string() } else { v.join(" ") }
                }),
                "switch with: tinybot --size base --arch gru   (sizes: nano small base large xl max, archs: bow cnn gru)".into(),
            ])
        }
        "bench" => {
            let (us, qps) = train::bench(&brain.model);
            out(vec![format!("{us:.1} us/query ({qps:.0} queries/s) with {} params", train::params_label(&brain.model))])
        }
        "fix" | "teach" if arg.contains("=>") => {
            let (p, r) = arg.split_once("=>").unwrap_or(("", ""));
            if intents::clean_line(p).is_empty() || intents::clean_line(r).is_empty() {
                return out(vec!["usage: /teach how is the weather on mars => Cold and dusty.".into()]);
            }
            let n = (1..).find(|n| brain.tag_idx(&format!("taught_{n}")).is_none()).unwrap_or(1);
            let tag = format!("taught_{n}");
            let _ = intents::append_learned(&tag, p, Some(r));
            let mut toasts = Vec::new();
            s.persona.add_xp(15, &mut toasts);
            let mut lines = vec![format!("ok, new intent {tag}: learning in the background, +15 XP")];
            lines.extend(toasts);
            app.start_retrain();
            out(lines)
        }
        "fix" | "teach" => match (brain.tag_idx(arg), s.last_input.clone()) {
            (Some(_), Some(t)) => {
                let _ = intents::append_learned(arg, &t, None);
                let mut lines = vec![format!("learned: {t:?} -> {arg} (learning in the background, +10 XP; /unlearn undoes it)")];
                s.persona.add_xp(10, &mut lines);
                app.start_retrain();
                out(lines)
            }
            (None, _) => out(vec![format!("unknown intent {arg:?}, see /intents")]),
            _ => out(vec!["say something first".into()]),
        },
        "retrain" => {
            app.start_retrain();
            out(vec!["relearning in the background".into()])
        }
        "unlearn" => match intents::unlearn_last() {
            Some(removed) => {
                app.start_retrain();
                out(vec![format!("removed: {removed}"), "relearning in the background".into()])
            }
            None => out(vec!["nothing learned yet".into()]),
        },
        "confidence" | "strict" => {
            if arg.is_empty() {
                let t = crate::model::answer_threshold();
                return out(vec![
                    format!("answer threshold {t:.2}: below it I ask \"did you mean...?\" or admit I don't know"),
                    "try /confidence 0.7 (about 93% precision) or 0.8 (about 95%); default 0.55. See the table in `tinybot eval`.".into(),
                ]);
            }
            match arg.parse::<f32>() {
                Ok(t) if (0.3..=0.95).contains(&t) => {
                    crate::model::set_answer_threshold(t);
                    s.mem.set("_conf", &format!("{t:.2}"));
                    out(vec![format!("answer threshold set to {t:.2}")])
                }
                _ => out(vec!["give a number between 0.3 and 0.95".into()]),
            }
        }
        "tz" => {
            if arg.is_empty() {
                return out(vec![format!("time zone: {} (set one with /tz +5:30, or /tz auto)", crate::skills::tz_label())]);
            }
            if arg == "auto" {
                crate::skills::set_tz_override(None);
                s.mem.facts.remove("_tz");
                s.mem.save();
                return out(vec![format!("time zone: auto ({})", crate::skills::tz_label())]);
            }
            match crate::skills::parse_tz(arg) {
                Some(m) => {
                    crate::skills::set_tz_override(Some(m));
                    s.mem.set("_tz", &m.to_string());
                    out(vec![format!("time zone set to {} ({})", crate::skills::tz_label(), crate::skills::clock())])
                }
                None => out(vec!["couldn't read that, try /tz +5:30 or /tz -8".into()]),
            }
        }
        _ => out(vec!["unknown command, try /help".into()]),
    }
}
