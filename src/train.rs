//! Training pipeline: teacher ensemble -> distilled student of any architecture and size, data
//! augmentation, calibration, cross-validation, and the architecture x size sweep.

use crate::intents::{self, Intent};
use crate::model::{self, argmax, decide, Arch, Cfg, Decision, Ensemble, Tier, TIERS};
use crate::rng::Rng;
use crate::text::{class_for_intent, class_members, noise};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::time::Instant;

pub const MODEL_PATH: &str = "bot.bin";
pub const SEED: u64 = 7;
pub type Examples = Vec<(usize, String)>;

/// What ships by default. Chosen from the sweep, see README.
pub const DEFAULT_ARCH: Arch = Arch::Gru;
pub const DEFAULT_TIER: &str = "base";

#[derive(Clone, Copy, PartialEq)]
pub enum TeacherKind {
    Bow,
    Het,
}

#[derive(Clone)]
pub struct Opts {
    pub arch: Arch,
    pub tier: &'static Tier,
    pub teacher: TeacherKind,
    pub aug: bool,
    pub folds: Vec<usize>,
    pub archs: Vec<Arch>,
    pub tiers: Vec<&'static Tier>,
}

impl Opts {
    pub fn default_opts() -> Opts {
        Opts {
            arch: DEFAULT_ARCH,
            tier: model::tier(DEFAULT_TIER).expect("default tier exists"),
            teacher: TeacherKind::Het,
            aug: true,
            folds: vec![0, 3],
            archs: vec![Arch::Bow, Arch::Cnn, Arch::Gru],
            tiers: TIERS.iter().collect(),
        }
    }

    /// --size base --arch gru --teacher bow|het --no-aug --folds 0,3 --archs bow,cnn --tiers nano,base
    pub fn parse(args: &[String]) -> Opts {
        let mut o = Opts::default_opts();
        let mut i = 0;
        while i < args.len() {
            let val = args.get(i + 1).map(String::as_str).unwrap_or("");
            match args[i].as_str() {
                "--size" => {
                    if let Some(t) = model::tier(val) {
                        o.tier = t;
                    }
                    i += 1;
                }
                "--arch" => {
                    if let Some(a) = Arch::parse(val) {
                        o.arch = a;
                    }
                    i += 1;
                }
                "--teacher" => {
                    o.teacher = if val == "bow" { TeacherKind::Bow } else { TeacherKind::Het };
                    i += 1;
                }
                "--no-aug" => o.aug = false,
                "--folds" => {
                    o.folds = val.split(',').filter_map(|x| x.parse().ok()).filter(|f| *f < 5).collect();
                    i += 1;
                }
                "--archs" => {
                    o.archs = val.split(',').filter_map(Arch::parse).collect();
                    i += 1;
                }
                "--tiers" => {
                    o.tiers = val.split(',').filter_map(model::tier).collect();
                    i += 1;
                }
                _ => {}
            }
            i += 1;
        }
        if o.folds.is_empty() {
            o.folds = vec![0, 3];
        }
        o
    }

    pub fn is_default(&self) -> bool {
        self.arch == DEFAULT_ARCH && self.tier.name == DEFAULT_TIER
    }

    /// bot.bin for the default model, models/<arch>-<size>.bin for every other size or architecture
    pub fn path(&self) -> String {
        if self.is_default() {
            MODEL_PATH.to_string()
        } else {
            format!("models/{}-{}.bin", self.arch.name(), self.tier.name)
        }
    }
}

pub fn student_cfg(arch: Arch, tier: &Tier, classes: usize) -> Cfg {
    model::fit(arch, tier, classes, model::student_epochs(arch))
}

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

/// Label-aware augmentation: inside an intent like mood_good, swap a word for another member of its
/// semantic class ("im feeling good" -> "im feeling great"), up to 2 variants per phrase. Only ever
/// applied to training data, never to held-out phrases.
pub fn augment(ex: &Examples, tags: &[String], seed: u64) -> Examples {
    let mut rng = Rng::new(seed ^ 0xA46);
    let mut out = ex.clone();
    let mut seen: std::collections::HashSet<(usize, String)> = ex.iter().cloned().collect();
    for (k, t) in ex {
        let Some(class) = tags.get(*k).and_then(|tag| class_for_intent(tag)) else { continue };
        let members = class_members(class);
        let ws: Vec<&str> = t.split_whitespace().collect();
        let cands: Vec<usize> = (0..ws.len()).filter(|&i| members.contains(&ws[i].to_lowercase().as_str())).collect();
        if cands.is_empty() {
            continue;
        }
        for _ in 0..2 {
            let i = cands[rng.below(cands.len())];
            let cur = ws[i].to_lowercase();
            let others: Vec<&str> = members.iter().copied().filter(|m| *m != cur).collect();
            let r: &str = others[rng.below(others.len())];
            let new: Vec<&str> = ws.iter().enumerate().map(|(j, w)| if j == i { r } else { *w }).collect();
            let new = new.join(" ");
            if seen.insert((*k, new.clone())) {
                out.push((*k, new));
            }
        }
    }
    out
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

pub fn model_kb(m: &Ensemble) -> f32 {
    let p = std::env::temp_dir().join(format!("tinybot_size_probe_{}.bin", std::process::id()));
    let ps = p.to_string_lossy().to_string();
    let _ = m.save(&ps);
    let n = fs::metadata(&p).map(|x| x.len()).unwrap_or(0);
    let _ = fs::remove_file(&p);
    n as f32 / 1024.0
}

pub fn fmt_params(n: usize) -> String {
    if n >= 1_000_000 { format!("{:.2}M", n as f32 / 1e6) } else { format!("{:.0}k", n as f32 / 1e3) }
}

/// "gru base, 253k params" style label for UIs
pub fn params_label(m: &Ensemble) -> String {
    fmt_params(m.nominal_params())
}

pub fn model_name(m: &Ensemble) -> String {
    let c = m.cfg();
    let t = TIERS.iter().min_by_key(|t| (t.target as i64 - m.nominal_params() as i64).abs()).map(|t| t.name).unwrap_or("custom");
    format!("{} {}", c.arch.name(), t)
}

fn teacher_list(kind: TeacherKind) -> Vec<Cfg> {
    match kind {
        TeacherKind::Bow => model::teacher_cfgs_bow(),
        TeacherKind::Het => model::teacher_cfgs(),
    }
}

const OOD: [&str; 24] = [
    "explain quantum entanglement", "order me a pizza", "what is the capital of france", "translate this into spanish",
    "who won the football match", "play some music", "book a flight to paris", "what is the meaning of life",
    "how do i cook pasta", "write me a poem about the sea", "turn off the lights", "call my mom",
    "send an email to john", "what is the stock price of apple", "show me pictures of cats", "how tall is mount everest",
    "recommend a movie", "define photosynthesis", "who is the president", "how do i reset my router password",
    "what is a black hole", "sing me a song", "open the pod bay doors", "install linux on my laptop",
];

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn quantile(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f32 * q).round() as usize]
}

/// Choose the "did you mean...?" similarity threshold from data: above what out-of-scope phrases reach,
/// below what genuinely related phrases reach. Each architecture's embeddings live on their own scale.
pub fn calibrate_clarify(m: &mut Ensemble, train: &Examples, val: &Examples) {
    let n = m.nets.len() as f32;
    let idx: Vec<(usize, Vec<f32>)> = train.iter().map(|(k, t)| (*k, m.embed(t))).collect();
    let nearest = |m: &Ensemble, text: &str, k: usize| -> f32 {
        let q = m.embed(text);
        idx.iter().filter(|(c, _)| *c == k).map(|(_, e)| dot(&q, e) / n).fold(-1.0, f32::max)
    };
    let (mut s_in, mut s_out) = (Vec::new(), Vec::new());
    for (y, t) in val {
        let p = m.predict(t);
        let k = argmax(&p.probs);
        if k == *y && !matches!(decide(&p.probs, p.cov), Decision::Reject) {
            s_in.push(nearest(m, t, k));
        }
    }
    for t in OOD {
        let p = m.predict(t);
        s_out.push(nearest(m, t, argmax(&p.probs)));
    }
    let (lo_in, hi_out) = (quantile(&mut s_in, 0.25), quantile(&mut s_out, 0.85));
    m.clarify_sim = ((lo_in + hi_out) / 2.0).max(hi_out + 0.01).clamp(0.5, 0.97);
}

/// Full build on all data: returns (teacher, student). Temperature and clarify threshold are fitted on
/// a held-out split first, then everything is retrained on all phrases.
pub fn build(intents: &[Intent], scfg: Cfg, kind: TeacherKind, aug: bool, verbose: bool, show_misses: bool) -> (Ensemble, Ensemble) {
    let (tags, all) = intents::flatten(intents);
    let (tr, val) = split(intents, 4);
    let t0 = Instant::now();
    let salt = model::cfg_salt(&scfg);
    let tr_aug = if aug { augment(&tr, &tags, SEED) } else { tr.clone() };
    let mut t = Ensemble::train(&tr_aug, tags.clone(), &teacher_list(kind), SEED, 1.0, 0);
    t.fit_temp(&val);
    let mut s = Ensemble::distill(&t, &tr_aug, scfg, SEED, 1.0, salt);
    s.fit_temp(&val);
    calibrate_clarify(&mut s, &tr, &val);
    if verbose {
        let mut rng = Rng::new(99);
        if show_misses {
            for (k, txt) in &val {
                let p = s.predict(txt);
                let g = argmax(&p.probs);
                if g != *k {
                    println!("  miss: {txt:?} -> {} ({:.2}), wanted {}", s.tags[g], p.probs[g], s.tags[*k]);
                }
            }
        }
        println!("{} intents, {} train / {} held-out phrases{}", tags.len(), tr.len(), val.len(), if aug { format!(" (+{} augmented)", tr_aug.len() - tr.len()) } else { String::new() });
        for (name, m) in [("teacher", &t), ("student", &s)] {
            println!(
                "  {name}: {:>6} params, {:>6.0} KB | held-out {:.1}% clean, {:.1}% typos (T={:.1})",
                fmt_params(m.nominal_params()),
                model_kb(m),
                accuracy(m, &val, &mut rng, false),
                accuracy(m, &val, &mut rng, true),
                m.temp
            );
        }
    }
    let all_aug = if aug { augment(&all, &tags, SEED) } else { all.clone() };
    let teacher = Ensemble::train(&all_aug, tags, &teacher_list(kind), SEED, t.temp, 0);
    let mut student = Ensemble::distill(&teacher, &all_aug, scfg, SEED, s.temp, salt);
    student.clarify_sim = s.clarify_sim;
    if verbose {
        println!("built in {:.2?}", t0.elapsed());
    }
    (teacher, student)
}

/// Builds every requested (architecture, size) from ONE shared teacher and writes each to its own file.
pub fn cmd_train_all(o: &Opts) {
    let intents = intents::load();
    let (tags, all) = intents::flatten(&intents);
    let (tr, val) = split(&intents, 4);
    let t0 = Instant::now();
    let (tr_aug, all_aug) = if o.aug { (augment(&tr, &tags, SEED), augment(&all, &tags, SEED)) } else { (tr.clone(), all.clone()) };
    let mut t = Ensemble::train(&tr_aug, tags.clone(), &teacher_list(o.teacher), SEED, 1.0, 0);
    t.fit_temp(&val);
    let teacher_all = Ensemble::train(&all_aug, tags, &teacher_list(o.teacher), SEED, t.temp, 0);
    println!("teacher ready ({:.0?}). held-out numbers below are from students trained without the held-out phrases.", t0.elapsed());
    header();
    let mut rng = Rng::new(99);
    for &arch in &o.archs {
        for tier in &o.tiers {
            let cfg = student_cfg(arch, tier, intents.len());
            let salt = model::cfg_salt(&cfg);
            let mut s = Ensemble::distill(&t, &tr_aug, cfg, SEED, 1.0, salt);
            s.fit_temp(&val);
            calibrate_clarify(&mut s, &tr, &val);
            let mut sc = Score::default();
            sc.add(&s, &val, &mut rng);
            let mut f = Ensemble::distill(&teacher_all, &all_aug, cfg, SEED, s.temp, salt);
            f.clarify_sim = s.clarify_sim;
            let one = Opts { arch, tier, ..o.clone() };
            let path = one.path();
            if let Some(dir) = std::path::Path::new(&path).parent() {
                let _ = fs::create_dir_all(dir);
            }
            f.save(&path).expect("could not write model");
            row(&format!("{}-{}", arch.name(), tier.name), f.nominal_params(), model_kb(&f), &sc, 0.0);
        }
    }
    println!("wrote {} models in {:.0?}; run one with: tinybot --size <tier> --arch <arch>", o.archs.len() * o.tiers.len(), t0.elapsed());
}

pub fn cmd_train(o: &Opts) -> Ensemble {
    train_impl(o, true)
}

/// Used for the automatic first-run / out-of-date retrain: same work, without the list of misses.
pub fn train_quiet(o: &Opts) -> Ensemble {
    train_impl(o, false)
}

fn train_impl(o: &Opts, show_misses: bool) -> Ensemble {
    let intents = intents::load();
    let cfg = student_cfg(o.arch, o.tier, intents.len());
    let (_, student) = build(&intents, cfg, o.teacher, o.aug, true, show_misses);
    let path = o.path();
    if let Some(dir) = std::path::Path::new(&path).parent() {
        let _ = fs::create_dir_all(dir);
    }
    student.save(&path).expect("could not write model");
    println!("saved {} ({}) -> {path}", model_name(&student), cfg.tag());
    student
}

/// Rebuild after /fix, /teach or a confirmed clarification. Always the quick bag-of-features teacher so
/// it finishes in seconds even on one core; the student keeps whatever architecture and size it has.
pub fn quick_student(intents: &[Intent], temp: f32, scfg: Cfg, clarify: f32) -> Ensemble {
    let (tags, all) = intents::flatten(intents);
    let all = augment(&all, &tags, SEED);
    let teacher = Ensemble::train(&all, tags, &model::teacher_cfgs_bow(), SEED, temp, 0);
    let mut s = Ensemble::distill(&teacher, &all, scfg, SEED, temp, model::cfg_salt(&scfg));
    s.clarify_sim = clarify;
    s
}

/// Was the model on disk built from exactly these intents? Launch uses this to decide whether to retrain.
///
/// Two kinds of model are current: one a fresh `train` run would produce, and one `quick_student` wrote after
/// /teach, /fix or a clarification. The second keeps the size the model already had (a fresh run would size it
/// for the new class count) and always trains on augmented data, so it is checked against its own config.
/// Comparing only against a fresh run's config made every /teach look out of date and forced a full retrain.
pub fn model_is_current(m: &Ensemble, intents: &[Intent], o: &Opts) -> bool {
    let (tags, ex) = intents::flatten(intents);
    if m.tags != tags {
        return false;
    }
    let augmented = augment(&ex, &tags, SEED);
    let fresh = student_cfg(o.arch, o.tier, intents.len());
    let used = if o.aug { &augmented } else { &ex };
    if m.data_hash == model::data_hash(&tags, used, model::cfg_salt(&fresh)) {
        return true;
    }
    let own = m.cfg();
    own.arch == o.arch && m.data_hash == model::data_hash(&tags, &augmented, model::cfg_salt(&own))
}

/// Raw inference speed of the shipped model.
pub fn bench(m: &Ensemble) -> (f32, f32) {
    let probes = ["hello there", "whats the weather like", "tell me a joke please", "i feel really down today", "thanks a lot", "who made you"];
    let n = 3000;
    let t = Instant::now();
    let mut sink = 0.0;
    for i in 0..n {
        sink += m.predict(probes[i % probes.len()]).probs[0];
    }
    let us = t.elapsed().as_secs_f32() * 1e6 / n as f32;
    std::hint::black_box(sink);
    (us, 1e6 / us)
}

// ---------------------------------------------------------------- scoring

#[derive(Default, Clone)]
pub struct Score {
    pub n: u32,
    pub clean: u32,
    pub typo: u32,
    pub answered: u32,
    pub answered_ok: u32,
    pub clarified: u32,
    pub ood_ok: u32,
    pub ood_n: u32,
    /// (top calibrated probability, was the top intent right) per held-out phrase
    pub conf_pairs: Vec<(f32, bool)>,
    pub conf: HashMap<(String, String), u32>,
}

impl Score {
    pub fn add(&mut self, m: &Ensemble, val: &Examples, rng: &mut Rng) {
        for (k, t) in val {
            self.n += 1;
            let p = m.predict(t);
            let g = argmax(&p.probs);
            self.clean += (g == *k) as u32;
            self.conf_pairs.push((p.probs[g], g == *k));
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
    pub fn pct(a: u32, b: u32) -> f32 {
        100.0 * a as f32 / b.max(1) as f32
    }
    pub fn clean_pct(&self) -> f32 {
        Self::pct(self.clean, self.n)
    }
    pub fn typo_pct(&self) -> f32 {
        Self::pct(self.typo, self.n)
    }
    pub fn prec_pct(&self) -> f32 {
        Self::pct(self.answered_ok, self.answered)
    }
    pub fn ood_pct(&self) -> f32 {
        Self::pct(self.ood_ok, self.ood_n)
    }
}

fn header() {
    println!("{:<18} {:>8} {:>7} {:>7} {:>7} {:>8} {:>7} {:>6} {:>7}", "model", "params", "KB", "clean", "typos", "auto-ok", "answd", "ood", "secs");
}

fn row(name: &str, params: usize, kb: f32, s: &Score, secs: f32) {
    println!(
        "{name:<18} {:>8} {:>7.0} {:>6.1}% {:>6.1}% {:>7.1}% {:>6.1}% {:>5.0}% {:>7.0}",
        fmt_params(params),
        kb,
        s.clean_pct(),
        s.typo_pct(),
        s.prec_pct(),
        Score::pct(s.answered, s.n),
        s.ood_pct(),
        secs
    );
    let _ = std::io::stdout().flush();
}

/// 5-fold cross-validation of the teacher and the configured student.
pub fn cmd_eval(o: &Opts) {
    let intents = intents::load();
    let tags: Vec<String> = intents.iter().map(|i| i.tag.clone()).collect();
    let scfg = student_cfg(o.arch, o.tier, intents.len());
    let t0 = Instant::now();
    let mut rng = Rng::new(5);
    let (mut teacher, mut student) = (Score::default(), Score::default());
    let (mut tp, mut tkb, mut sp, mut skb) = (0, 0.0, 0, 0.0);
    for fold in 0..5 {
        let (tr, val) = split(&intents, fold);
        let tr = if o.aug { augment(&tr, &tags, SEED) } else { tr };
        let mut t = Ensemble::train(&tr, tags.clone(), &teacher_list(o.teacher), SEED + fold as u64, 1.0, 0);
        t.fit_temp(&val);
        teacher.add(&t, &val, &mut rng);
        tp = t.nominal_params();
        tkb = model_kb(&t);
        let mut s = Ensemble::distill(&t, &tr, scfg, SEED + fold as u64, 1.0, 0);
        s.fit_temp(&val);
        student.add(&s, &val, &mut rng);
        sp = s.nominal_params();
        skb = model_kb(&s);
        eprint!(".");
    }
    eprintln!();
    println!("5-fold CV over {} phrases ({:.1?}), teacher = {}", teacher.n, t0.elapsed(), if o.teacher == TeacherKind::Het { "3x bow + cnn + gru" } else { "5x bow" });
    header();
    row("teacher (ensemble)", tp, tkb, &teacher, 0.0);
    row(&format!("student {}-{}", o.arch.name(), o.tier.name), sp, skb, &student, 0.0);
    println!("precision vs coverage as the answer threshold moves (now {:.2}); below it the bot asks or declines:", model::P_ANSWER);
    println!("  threshold  answered  precision  wrong answers");
    for th in [0.40f32, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90] {
        let ans: Vec<&(f32, bool)> = student.conf_pairs.iter().filter(|(p, _)| *p >= th).collect();
        let ok = ans.iter().filter(|(_, c)| *c).count();
        println!("  {th:>8.2}  {:>7.1}%  {:>8.1}%  {:>5.1}% of inputs", Score::pct(ans.len() as u32, student.n), Score::pct(ok as u32, ans.len() as u32), Score::pct((ans.len() - ok) as u32, student.n));
    }
    let mut c: Vec<_> = student.conf.iter().collect();
    c.sort_by(|a, b| b.1.cmp(a.1));
    println!("top confusions (student):");
    for ((want, got), n) in c.iter().take(6) {
        println!("  {n}x  {want} -> {got}");
    }
}

/// Every architecture at every size on the same folds, all distilled from the same teacher.
pub fn cmd_sweep(o: &Opts) {
    let intents = intents::load();
    let tags: Vec<String> = intents.iter().map(|i| i.tag.clone()).collect();
    let c = intents.len();
    let t0 = Instant::now();
    eprintln!("training teachers on {} fold(s)...", o.folds.len());
    let mut folds = Vec::new();
    let (mut t_bow, mut t_het, mut sc_bow, mut sc_het) = (Vec::new(), Vec::new(), Score::default(), Score::default());
    let mut rng = Rng::new(5);
    for &f in &o.folds {
        let (tr, val) = split(&intents, f);
        let tr = if o.aug { augment(&tr, &tags, SEED) } else { tr };
        let mut tb = Ensemble::train(&tr, tags.clone(), &teacher_list(TeacherKind::Bow), SEED + f as u64, 1.0, 0);
        tb.fit_temp(&val);
        sc_bow.add(&tb, &val, &mut rng);
        let mut th = Ensemble::train(&tr, tags.clone(), &teacher_list(TeacherKind::Het), SEED + f as u64, 1.0, 0);
        th.fit_temp(&val);
        sc_het.add(&th, &val, &mut rng);
        t_bow.push(tb);
        t_het.push(th);
        folds.push((tr, val));
        eprintln!("  fold {f} teachers done ({:.0?})", t0.elapsed());
    }
    println!("folds {:?}, {} held-out phrases, augmentation {}, teacher for distillation: {}", o.folds, sc_het.n, if o.aug { "on" } else { "off" }, if o.teacher == TeacherKind::Het { "het" } else { "bow" });
    header();
    row("teacher 5x bow", t_bow[0].nominal_params(), model_kb(&t_bow[0]), &sc_bow, 0.0);
    row("teacher 3bow+cnn+gru", t_het[0].nominal_params(), model_kb(&t_het[0]), &sc_het, 0.0);
    let teachers = if o.teacher == TeacherKind::Het { &t_het } else { &t_bow };
    for &arch in &o.archs {
        for tier in &o.tiers {
            let cfg = student_cfg(arch, tier, c);
            let (mut sc, mut secs, mut kb, mut params) = (Score::default(), 0.0, 0.0, 0);
            for (i, (tr, val)) in folds.iter().enumerate() {
                let t1 = Instant::now();
                let mut s = Ensemble::distill(&teachers[i], tr, cfg, SEED + o.folds[i] as u64, 1.0, 0);
                s.fit_temp(val);
                secs += t1.elapsed().as_secs_f32();
                sc.add(&s, val, &mut rng);
                kb = model_kb(&s);
                params = s.nominal_params();
            }
            row(&format!("{}-{}", arch.name(), tier.name), params, kb, &sc, secs / folds.len() as f32);
        }
    }
    println!("total {:.0?}", t0.elapsed());
}

/// The tier table: what each architecture costs at each size.
pub fn cmd_models(classes: usize) {
    println!("{:<7} {:>9}   {:<24} {:<26} {:<26}", "tier", "target", "bow (dim,hid)", "cnn (dim,filters,hid)", "gru (dim,hidden,hid)");
    for t in &TIERS {
        let f = |a: Arch| {
            let c = student_cfg(a, t, classes);
            let extra = if a == Arch::Bow { String::new() } else { format!(",{}", c.ch) };
            format!("{} ({}{extra},{})", fmt_params(c.nominal(classes)), c.dim, c.hid)
        };
        println!("{:<7} {:>9}   {:<24} {:<26} {:<26}", t.name, fmt_params(t.target), f(Arch::Bow), f(Arch::Cnn), f(Arch::Gru));
    }
}

/// Training step cost of every architecture at every size (for choosing epochs).
pub fn cmd_speed(classes: usize) {
    println!("{:<7} {:>12} {:>12} {:>12}   (microseconds per training step)", "tier", "bow", "cnn", "gru");
    for t in &TIERS {
        let us: Vec<String> = [Arch::Bow, Arch::Cnn, Arch::Gru].iter().map(|&a| format!("{:.0}", model::bench_step_us(student_cfg(a, t, classes), classes))).collect();
        println!("{:<7} {:>12} {:>12} {:>12}", t.name, us[0], us[1], us[2]);
    }
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
    let all = augment(&all, &tags, SEED);
    let scfg = student_cfg(DEFAULT_ARCH, model::tier(DEFAULT_TIER).expect("tier"), intents.len());
    let (mut base_tot, mut flow_tot, mut n_tot) = (0, 0, 0);
    // several independently trained students: a 36-case probe is too small to trust a single seed
    for seed in 0..4u64 {
        let t = Ensemble::train(&all, tags.clone(), &model::teacher_cfgs_bow(), SEED + 100 * seed, 1.0, 0);
        let m = Ensemble::distill(&t, &all, scfg, SEED + 100 * seed, 1.0, 0);
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
    let n = n_tot / 4;
    let d = 4.0 * n as f32;
    println!("context probe, {n} short follow-ups x 4 seeds: {:.1}% without context, {:.1}% with the flow prior", 100.0 * base_tot as f32 / d, 100.0 * flow_tot as f32 / d);
}
