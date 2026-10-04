//! Training pipeline: teacher ensemble -> distilled student, evaluation, and the
//! parameters-vs-accuracy sweep.

use crate::intents::{self, Intent};
use crate::model::{argmax, decide, Cfg, Decision, Ensemble, STUDENT, TEACHER, TEACHER_NETS};
use crate::rng::Rng;
use crate::text::noise;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

pub const MODEL_PATH: &str = "bot.bin";
pub const SEED: u64 = 7;
pub type Examples = Vec<(usize, String)>;

/// held-out split: every 5th phrase of each intent that has at least 6 phrases
pub fn split(intents: &[Intent], fold: usize) -> (Examples, Examples) {
    let (mut tr, mut val) = (Vec::new(), Vec::new());
    for (k, it) in intents.iter().enumerate() {
        for (i, ex) in it.examples.iter().enumerate() {
            let dest = if it.examples.len() >= 6 && i % 5 == fold { &mut val } else { &mut tr };
            dest.push((k, ex.clone()));
        }
    }
    (tr, val)
}

pub fn accuracy(m: &Ensemble, val: &Examples, rng: &mut Rng, typos: bool) -> f32 {
    let ok = val
        .iter()
        .filter(|(k, t)| {
            let t = if typos { noise(t, rng) } else { t.clone() };
            argmax(&m.predict(&t).probs) == *k
        })
        .count();
    100.0 * ok as f32 / val.len().max(1) as f32
}

fn file_kb(m: &Ensemble) -> f32 {
    let p = std::env::temp_dir().join("tinybot_size_probe.bin");
    let ps = p.to_string_lossy().to_string();
    let _ = m.save(&ps);
    let n = fs::metadata(&p).map(|x| x.len()).unwrap_or(0);
    let _ = fs::remove_file(&p);
    n as f32 / 1024.0
}

fn fmt_params(n: usize) -> String {
    if n >= 1_000_000 { format!("{:.2}M", n as f32 / 1e6) } else { format!("{:.0}k", n as f32 / 1e3) }
}

/// Full build: returns (teacher, student) trained on everything, temperature fitted on a held-out split.
pub fn build(intents: &[Intent], cfg: Cfg, verbose: bool, show_misses: bool) -> (Ensemble, Ensemble) {
    let (tags, all) = intents::flatten(intents);
    let (tr, val) = split(intents, 4);
    let t0 = Instant::now();
    let mut t = Ensemble::train(&tr, tags.clone(), TEACHER_NETS, TEACHER, SEED, 1.0, 0);
    t.fit_temp(&val);
    let mut s = Ensemble::distill(&t, &tr, cfg, SEED, 1.0, 0);
    s.fit_temp(&val);
    if verbose {
        let mut rng = Rng::new(99);
        for (k, txt) in &val {
            let p = s.predict(txt);
            let g = argmax(&p.probs);
            if show_misses && g != *k {
                println!("  miss: {txt:?} -> {} ({:.2}), wanted {}", s.tags[g], p.probs[g], s.tags[*k]);
            }
        }
        println!("{} intents, {} train / {} held-out phrases", tags.len(), tr.len(), val.len());
        for (name, m) in [("teacher", &t), ("student", &s)] {
            println!(
                "  {name}: {:>6} params, {:>6.0} KB | held-out {:.1}% clean, {:.1}% typos (T={:.1})",
                fmt_params(m.param_report().2),
                file_kb(m),
                accuracy(m, &val, &mut rng, false),
                accuracy(m, &val, &mut rng, true),
                m.temp
            );
        }
    }
    let teacher = Ensemble::train(&all, tags, TEACHER_NETS, TEACHER, SEED, t.temp, 0);
    let student = Ensemble::distill(&teacher, &all, cfg, SEED, s.temp, 0);
    if verbose {
        println!("built in {:.2?}", t0.elapsed());
    }
    (teacher, student)
}

pub fn cmd_train() -> Ensemble {
    train_impl(true)
}

/// Used for the automatic first-run / out-of-date retrain: same work, without the list of misses.
pub fn train_quiet() -> Ensemble {
    train_impl(false)
}

fn train_impl(show_misses: bool) -> Ensemble {
    let intents = intents::load();
    let (_, student) = build(&intents, STUDENT, true, show_misses);
    student.save(MODEL_PATH).expect("could not write model");
    println!("saved student -> {MODEL_PATH}");
    student
}

const OOD: [&str; 24] = [
    "explain quantum entanglement", "order me a pizza", "what is the capital of france", "translate this into spanish",
    "who won the football match", "play some music", "book a flight to paris", "what is the meaning of life",
    "how do i cook pasta", "write me a poem about the sea", "turn off the lights", "call my mom",
    "send an email to john", "what is the stock price of apple", "show me pictures of cats", "how tall is mount everest",
    "recommend a movie", "define photosynthesis", "who is the president", "how do i reset my router password",
    "what is a black hole", "sing me a song", "open the pod bay doors", "install linux on my laptop",
];

#[derive(Default)]
struct Score {
    n: u32,
    clean: u32,
    typo: u32,
    answered: u32,
    answered_ok: u32,
    clarified: u32,
    ood_ok: u32,
    ood_n: u32,
    conf: HashMap<(String, String), u32>,
}

impl Score {
    fn add(&mut self, m: &Ensemble, val: &Examples, rng: &mut Rng) {
        for (k, t) in val {
            self.n += 1;
            let p = m.predict(t);
            let g = argmax(&p.probs);
            self.clean += (g == *k) as u32;
            self.typo += (argmax(&m.predict(&noise(t, rng)).probs) == *k) as u32;
            match decide(&p.probs, p.cov) {
                Decision::Answer(a) => {
                    self.answered += 1;
                    self.answered_ok += (a == *k) as u32;
                }
                Decision::Clarify(_) => self.clarified += 1,
                Decision::Reject => {}
            }
            if g != *k {
                *self.conf.entry((m.tags[*k].clone(), m.tags[g].clone())).or_default() += 1;
            }
        }
        for t in OOD {
            let p = m.predict(t);
            self.ood_n += 1;
            self.ood_ok += !matches!(decide(&p.probs, p.cov), Decision::Answer(_)) as u32;
        }
    }
    fn pct(a: u32, b: u32) -> f32 {
        100.0 * a as f32 / b.max(1) as f32
    }
}

/// 5-fold CV of teacher and student, plus optional student-size sweep.
pub fn cmd_eval(sweep: bool) {
    let intents = intents::load();
    let tags: Vec<String> = intents.iter().map(|i| i.tag.clone()).collect();
    let cfgs: Vec<Cfg> = if sweep {
        [(16, 32, 10), (16, 32, 12), (24, 32, 11), (24, 48, 12), (32, 48, 12), (32, 48, 13), (48, 64, 12), (48, 64, 13)]
            .iter()
            .map(|&(dim, hid, bits)| Cfg { dim, hid, bits, epochs: STUDENT.epochs })
            .collect::<Vec<_>>()
    } else {
        vec![STUDENT]
    };
    let t0 = Instant::now();
    let mut rng = Rng::new(5);
    let mut teacher = Score::default();
    let mut students: Vec<Score> = cfgs.iter().map(|_| Score::default()).collect();
    let mut params = vec![0usize; cfgs.len()];
    let mut kb = vec![0f32; cfgs.len()];
    let (mut tp, mut tkb) = (0, 0.0);
    for fold in 0..5 {
        let (tr, val) = split(&intents, fold);
        let mut t = Ensemble::train(&tr, tags.clone(), TEACHER_NETS, TEACHER, SEED + fold as u64, 1.0, 0);
        t.fit_temp(&val);
        teacher.add(&t, &val, &mut rng);
        tp = t.param_report().2;
        tkb = file_kb(&t);
        for (i, cfg) in cfgs.iter().enumerate() {
            let mut s = Ensemble::distill(&t, &tr, *cfg, SEED + fold as u64, 1.0, 0);
            s.fit_temp(&val);
            students[i].add(&s, &val, &mut rng);
            params[i] = s.param_report().2;
            kb[i] = file_kb(&s);
        }
        eprint!(".");
    }
    eprintln!();
    println!("5-fold CV over {} phrases ({:.1?})", teacher.n, t0.elapsed());
    println!("{:<22} {:>7} {:>7} {:>7} {:>7} {:>8} {:>6}", "model", "params", "KB", "clean", "typos", "auto-ok", "ood");
    let row = |name: String, p: usize, k: f32, s: &Score| {
        println!(
            "{name:<22} {:>7} {:>7.0} {:>6.1}% {:>6.1}% {:>7.1}% {:>5.0}%",
            fmt_params(p),
            k,
            Score::pct(s.clean, s.n),
            Score::pct(s.typo, s.n),
            Score::pct(s.answered_ok, s.answered),
            Score::pct(s.ood_ok, s.ood_n)
        );
    };
    row(format!("teacher {TEACHER_NETS}x{}", TEACHER.dim), tp, tkb, &teacher);
    for (i, cfg) in cfgs.iter().enumerate() {
        row(format!("student d{} h{} b{}", cfg.dim, cfg.hid, cfg.bits), params[i], kb[i], &students[i]);
    }
    let s = if sweep { &teacher } else { &students[0] };
    let mut c: Vec<_> = s.conf.iter().collect();
    c.sort_by(|a, b| b.1.cmp(a.1));
    println!("top confusions ({}):", if sweep { "teacher" } else { "student" });
    for ((want, got), n) in c.iter().take(6) {
        println!("  {n}x  {want} -> {got}");
    }
}

/// Rebuild after /fix or /teach: teacher on everything, distill, keep the fitted temperature.
pub fn quick_student(intents: &[Intent], temp: f32) -> Ensemble {
    let (tags, all) = intents::flatten(intents);
    let teacher = Ensemble::train(&all, tags, TEACHER_NETS, TEACHER, SEED, temp, 0);
    Ensemble::distill(&teacher, &all, STUDENT, SEED, temp, 0)
}

/// Raw inference speed of the shipped model.
pub fn bench(m: &Ensemble) -> (f32, f32) {
    let probes = ["hello there", "whats the weather like", "tell me a joke please", "i feel really down today", "thanks a lot", "who made you"];
    let n = 4000;
    let t = Instant::now();
    let mut sink = 0.0;
    for i in 0..n {
        sink += m.predict(probes[i % probes.len()]).probs[0];
    }
    let us = t.elapsed().as_secs_f32() * 1e6 / n as f32;
    std::hint::black_box(sink);
    (us, 1e6 / us)
}

pub fn model_kb(m: &Ensemble) -> f32 {
    file_kb(m)
}

pub fn params_label(m: &Ensemble) -> String {
    fmt_params(m.param_report().2)
}

/// Does knowing the previous intent help? Held-out short replies, with and without the flow prior.
pub fn cmd_ctx_eval() {
    use crate::bot::{Brain, Session};
    use crate::skills::Memory;
    // (previous intent, what the user says, intent we want). None of these phrases are in intents.txt.
    const CASES: [(&str, &str, &str); 36] = [
        ("how_are_you", "fine thanks", "mood_good"), ("how_are_you", "pretty good", "mood_good"), ("how_are_you", "not great", "mood_bad"),
        ("how_are_you", "meh", "mood_bad"), ("how_are_you", "and you", "and_you"), ("how_are_you", "great", "mood_good"),
        ("how_are_you", "terrible", "mood_bad"), ("how_are_you", "tired", "mood_bad"),
        ("joke", "hehe", "laugh"), ("joke", "another", "again"), ("joke", "one more", "again"), ("joke", "yes", "yes"),
        ("joke", "nope", "no"), ("joke", "ha", "laugh"), ("joke", "not funny", "insult"), ("joke", "do it again", "again"),
        ("tell_fact", "really", "elaborate"), ("tell_fact", "why", "elaborate"), ("tell_fact", "another", "again"), ("tell_fact", "more", "again"),
        ("tell_fact", "interesting", "compliment"), ("tell_fact", "how so", "elaborate"),
        ("mood_bad", "yes", "yes"), ("mood_bad", "yeah", "yes"), ("mood_bad", "no", "no"), ("mood_bad", "not really", "no"),
        ("mood_bad", "sure", "yes"), ("mood_bad", "nah", "no"),
        ("greeting", "good thanks", "mood_good"), ("greeting", "hello again", "greeting"), ("greeting", "hey", "greeting"),
        ("capabilities", "can you do math", "calc_help"), ("quote", "thanks", "thanks"), ("compliment", "thanks", "thanks"),
        ("insult", "sorry", "sorry"), ("bored", "ok", "yes"),
    ];
    let intents = intents::load();
    let (tags, all) = intents::flatten(&intents);
    let (mut base_tot, mut flow_tot, mut n_tot) = (0, 0, 0);
    // several independently trained students: a 36-case probe is too small to trust a single seed
    for seed in 0..4u64 {
        let t = Ensemble::train(&all, tags.clone(), TEACHER_NETS, TEACHER, SEED + 100 * seed, 1.0, 0);
        let m = Ensemble::distill(&t, &all, STUDENT, SEED + 100 * seed, 1.0, 0);
        let brain = Brain::new(m, intents.clone());
        let mut s = Session::new(Memory::ephemeral(), false);
        for (prev, text, want) in CASES {
            let Some(w) = brain.tag_idx(want) else { continue };
            n_tot += 1;
            let mut p = brain.model.predict(text).probs;
            base_tot += (argmax(&p) == w) as u32;
            s.prev = Some(prev.to_string());
            brain.apply_flow(&s, &mut p, text.split_whitespace().count());
            flow_tot += (argmax(&p) == w) as u32;
        }
    }
    let (base, flow, n) = (base_tot, flow_tot, n_tot / 4);
    let d = 4.0 * n as f32;
    println!("context probe, {n} short follow-ups x 4 seeds: {:.1}% without context, {:.1}% with the flow prior", 100.0 * base as f32 / d, 100.0 * flow as f32 / d);
}
