//! Intent classifiers, all hand-written: a bag-of-features MLP (fastText-style), a temporal CNN and a
//! bidirectional GRU (see seqnet.rs), bagged into ensembles and distilled into one compact student.
//!
//! Parameter efficiency: a heterogeneous "teacher" ensemble is trained once, then labels thousands of
//! synthetic phrases (typos, dropped words, spliced phrases) and ONE small student learns to reproduce
//! its soft predictions. The student ships, with int8 embedding rows and only the trained rows stored.

use crate::rng::Rng;
use crate::seqnet::{softmax, CnnNet, GruNet, Out};
use crate::text::{bucket_in, fnv, noise, Input, BUCKETS, CHAR_BASE, FNV_INIT};
use std::sync::atomic::{AtomicU32, Ordering};
use std::{fs, io};

// ---------------------------------------------------------------- configuration

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Bow,
    Cnn,
    Gru,
}

impl Arch {
    pub fn id(self) -> u8 {
        self as u8
    }
    pub fn from_id(i: u8) -> Option<Arch> {
        [Arch::Bow, Arch::Cnn, Arch::Gru].get(i as usize).copied()
    }
    pub fn name(self) -> &'static str {
        ["bow", "cnn", "gru"][self as usize]
    }
    pub fn parse(s: &str) -> Option<Arch> {
        [Arch::Bow, Arch::Cnn, Arch::Gru].into_iter().find(|a| a.name() == s.to_lowercase())
    }
}

/// `ch` is the CNN filter count per kernel size or the GRU hidden size per direction (unused by Bow).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cfg {
    pub arch: Arch,
    pub dim: usize,
    pub hid: usize,
    pub bits: u32, // embedding rows = 1 << bits
    pub ch: usize,
    pub epochs: usize,
}

impl Cfg {
    /// Parameters allocated (what the model costs in memory), not just the rows that were trained.
    pub fn nominal(&self, c: usize) -> usize {
        let emb = (1usize << self.bits) * self.dim;
        let in_ = 2 * self.dim;
        let head = |pin: usize| pin * self.hid + self.hid + self.hid * c + c;
        emb + match self.arch {
            Arch::Bow => head(in_),
            Arch::Cnn => self.ch * in_ * 6 + 3 * self.ch + head(3 * self.ch),
            Arch::Gru => 2 * (3 * self.ch * in_ + 3 * self.ch * self.ch + 6 * self.ch) + head(2 * self.ch),
        }
    }

    pub fn tag(&self) -> String {
        format!("{}-d{}h{}c{}b{}", self.arch.name(), self.dim, self.hid, self.ch, self.bits)
    }
}

pub struct Tier {
    pub name: &'static str,
    pub target: usize,
    pub bits: u32,
}

/// Six sizes from the original 68k student up to 4M. Embedding rows grow first, then width.
pub const TIERS: [Tier; 6] = [
    Tier { name: "nano", target: 68_000, bits: 12 },
    Tier { name: "small", target: 100_000, bits: 12 },
    Tier { name: "base", target: 250_000, bits: 13 },
    Tier { name: "large", target: 500_000, bits: 14 },
    Tier { name: "xl", target: 1_000_000, bits: 14 },
    Tier { name: "max", target: 4_000_000, bits: 15 },
];

pub fn tier(name: &str) -> Option<&'static Tier> {
    TIERS.iter().find(|t| t.name == name.to_lowercase())
}

/// Student epochs per pass over the distillation pool, by architecture.
pub fn student_epochs(arch: Arch) -> usize {
    match arch {
        Arch::Bow => 40,
        Arch::Cnn => 10,
        Arch::Gru => 8,
    }
}

/// The configuration of `arch` whose parameter count is closest to the tier's target at `classes` classes.
pub fn fit(arch: Arch, t: &Tier, classes: usize, epochs: usize) -> Cfg {
    let mut best: Option<(f32, Cfg)> = None;
    for dim in 4..=200usize {
        let hid = (2 * dim).clamp(32, 192);
        let ch = (dim * 3 / 2).clamp(16, 96);
        let cfg = Cfg { arch, dim, hid, bits: t.bits, ch: if arch == Arch::Bow { 0 } else { ch }, epochs };
        let err = (cfg.nominal(classes) as f32 - t.target as f32).abs() / t.target as f32;
        if best.is_none_or(|(e, _)| err < e) {
            best = Some((err, cfg));
        }
    }
    best.map(|b| b.1).unwrap_or(Cfg { arch, dim: 16, hid: 32, bits: 12, ch: 0, epochs })
}

pub const TEACHER_BOW: Cfg = Cfg { arch: Arch::Bow, dim: 48, hid: 64, bits: 14, ch: 0, epochs: 160 };
pub const TEACHER_CNN: Cfg = Cfg { arch: Arch::Cnn, dim: 24, hid: 64, bits: 14, ch: 48, epochs: 30 };
pub const TEACHER_GRU: Cfg = Cfg { arch: Arch::Gru, dim: 24, hid: 64, bits: 14, ch: 32, epochs: 24 };

/// Heterogeneous teacher: different architectures make different mistakes, so their average is stronger.
pub fn teacher_cfgs() -> Vec<Cfg> {
    vec![TEACHER_BOW, TEACHER_BOW, TEACHER_BOW, TEACHER_CNN, TEACHER_GRU]
}

/// the 5x bag-of-features teacher used before sequence models existed
pub fn teacher_cfgs_bow() -> Vec<Cfg> {
    vec![TEACHER_BOW; 5]
}

const EMB_BOOST: f32 = 4.0;
const SMOOTH: f32 = 0.04;
const MAGIC: &[u8] = b"TINYBOT5";

// decision thresholds (calibrated probability)
pub const P_ANSWER: f32 = 0.55;
static ANSWER_BITS: AtomicU32 = AtomicU32::new(0);

/// How sure the bot must be to answer outright. Raise it for fewer mistakes and more "did you mean...?".
pub fn answer_threshold() -> f32 {
    match ANSWER_BITS.load(Ordering::Relaxed) {
        0 => P_ANSWER,
        b => f32::from_bits(b),
    }
}

pub fn set_answer_threshold(t: f32) {
    ANSWER_BITS.store(t.clamp(0.3, 0.95).to_bits(), Ordering::Relaxed);
}
pub const P_CLARIFY: f32 = 0.28;
pub const COV_MIN: f32 = 0.30;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Decision {
    Answer(usize),
    Clarify(usize),
    Reject,
}

pub fn argmax(p: &[f32]) -> usize {
    p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|x| x.0).unwrap_or(0)
}

pub fn decide(p: &[f32], cov: f32) -> Decision {
    if p.is_empty() || cov < COV_MIN {
        return Decision::Reject;
    }
    let k = argmax(p);
    if p[k] >= answer_threshold() {
        Decision::Answer(k)
    } else if p[k] >= P_CLARIFY {
        Decision::Clarify(k)
    } else {
        Decision::Reject
    }
}

fn hard_target(y: usize, c: usize) -> Vec<f32> {
    let off = SMOOTH / (c.max(2) - 1) as f32;
    (0..c).map(|k| if k == y { 1.0 - SMOOTH } else { off }).collect()
}

// ---------------------------------------------------------------- bag-of-features net

pub struct BowNet {
    cfg: Cfg,
    emb: Vec<f32>, // rows x dim
    w1: Vec<f32>,  // 2*dim x hid
    b1: Vec<f32>,
    w2: Vec<f32>, // hid x classes
    b2: Vec<f32>,
    touched: Vec<bool>, // rows that ever received a gradient (the rest are untrained noise)
}

impl BowNet {
    fn new(c: usize, cfg: Cfg, rng: &mut Rng) -> BowNet {
        let rows = 1usize << cfg.bits;
        let inp = 2 * cfg.dim;
        let mut uni = |n: usize, lim: f32| -> Vec<f32> { (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * lim).collect() };
        BowNet {
            cfg,
            emb: uni(rows * cfg.dim, 0.3),
            w1: uni(inp * cfg.hid, (6.0 / inp as f32).sqrt()),
            b1: vec![0.0; cfg.hid],
            w2: uni(cfg.hid * c, (6.0 / cfg.hid as f32).sqrt()),
            b2: vec![0.0; c],
            touched: vec![false; rows],
        }
    }

    /// (pooled input, hidden activations, class probabilities). Word/bigram features and
    /// char n-grams pool into separate halves so one decisive word isn't averaged away.
    fn forward(&self, feats: &[usize]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (dim, hid, c) = (self.cfg.dim, self.cfg.hid, self.b2.len());
        let mut h0 = vec![0.0; 2 * dim];
        let mut cnt = [0usize; 2];
        for &f in feats {
            let g = (f >= CHAR_BASE) as usize;
            cnt[g] += 1;
            let b = bucket_in(f, self.cfg.bits);
            let r = &self.emb[b * dim..(b + 1) * dim];
            for d in 0..dim {
                h0[g * dim + d] += r[d];
            }
        }
        for g in 0..2 {
            let inv = 1.0 / cnt[g].max(1) as f32;
            h0[g * dim..(g + 1) * dim].iter_mut().for_each(|v| *v *= inv);
        }
        let mut z1 = self.b1.clone();
        for d in 0..2 * dim {
            let row = &self.w1[d * hid..(d + 1) * hid];
            for j in 0..hid {
                z1[j] += h0[d] * row[j];
            }
        }
        z1.iter_mut().for_each(|v| *v = v.max(0.0));
        let mut lg = self.b2.clone();
        for j in 0..hid {
            if z1[j] == 0.0 {
                continue;
            }
            let row = &self.w2[j * c..(j + 1) * c];
            for k in 0..c {
                lg[k] += z1[j] * row[k];
            }
        }
        softmax(&mut lg);
        (h0, z1, lg)
    }

    fn infer(&self, feats: &[usize]) -> Out {
        let kept: Vec<usize> = feats.iter().cloned().filter(|&f| self.touched[bucket_in(f, self.cfg.bits)]).collect();
        let (emb, hid, probs) = self.forward(&kept);
        Out { emb, hid, probs }
    }

    /// One SGD step towards a target distribution.
    fn step(&mut self, feats: &[usize], target: &[f32], lr: f32) {
        if feats.is_empty() {
            return;
        }
        let (dim, hid) = (self.cfg.dim, self.cfg.hid);
        for &f in feats {
            self.touched[bucket_in(f, self.cfg.bits)] = true;
        }
        let (h0, z1, mut dl) = self.forward(feats);
        let c = dl.len();
        for k in 0..c {
            dl[k] -= target[k];
        }
        let mut dz1 = vec![0.0; hid];
        for j in 0..hid {
            if z1[j] <= 0.0 {
                continue;
            }
            let row = &self.w2[j * c..(j + 1) * c];
            dz1[j] = (0..c).map(|k| row[k] * dl[k]).sum();
        }
        for j in 0..hid {
            if z1[j] == 0.0 {
                continue;
            }
            let row = &mut self.w2[j * c..(j + 1) * c];
            for k in 0..c {
                row[k] -= lr * z1[j] * dl[k];
            }
        }
        for k in 0..c {
            self.b2[k] -= lr * dl[k];
        }
        let mut dh0 = vec![0.0; 2 * dim];
        for d in 0..2 * dim {
            let row = &mut self.w1[d * hid..(d + 1) * hid];
            for j in 0..hid {
                dh0[d] += row[j] * dz1[j];
                row[j] -= lr * h0[d] * dz1[j];
            }
        }
        for j in 0..hid {
            self.b1[j] -= lr * dz1[j];
        }
        let mut cnt = [0usize; 2];
        for &f in feats {
            cnt[(f >= CHAR_BASE) as usize] += 1;
        }
        for &f in feats {
            let g = (f >= CHAR_BASE) as usize;
            let scale = lr * EMB_BOOST / cnt[g] as f32;
            let b = bucket_in(f, self.cfg.bits);
            let r = &mut self.emb[b * dim..(b + 1) * dim];
            for d in 0..dim {
                r[d] -= scale * dh0[g * dim + d];
            }
        }
    }
}

// ---------------------------------------------------------------- any architecture

pub enum Net {
    Bow(BowNet),
    Cnn(CnnNet),
    Gru(GruNet),
}

fn quant_row(r: &[f32]) -> (f32, Vec<i8>) {
    let m = r.iter().fold(0.0f32, |a, &x| a.max(x.abs())).max(1e-8);
    let scale = m / 127.0;
    (scale, r.iter().map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8).collect())
}

impl Net {
    fn new(c: usize, cfg: Cfg, rng: &mut Rng) -> Net {
        match cfg.arch {
            Arch::Bow => Net::Bow(BowNet::new(c, cfg, rng)),
            Arch::Cnn => Net::Cnn(CnnNet::new(c, cfg, rng)),
            Arch::Gru => Net::Gru(GruNet::new(c, cfg, rng)),
        }
    }

    pub fn cfg(&self) -> Cfg {
        match self {
            Net::Bow(n) => n.cfg,
            Net::Cnn(n) => n.cfg,
            Net::Gru(n) => n.cfg,
        }
    }

    fn infer(&self, inp: &Input) -> Out {
        match self {
            Net::Bow(n) => n.infer(&inp.flat),
            Net::Cnn(n) => n.infer(&inp.toks),
            Net::Gru(n) => n.infer(&inp.toks),
        }
    }

    fn step(&mut self, inp: &Input, target: &[f32], lr: f32) {
        match self {
            Net::Bow(n) => n.step(&inp.flat, target, lr),
            Net::Cnn(n) => n.step(&inp.toks, target, lr),
            Net::Gru(n) => n.step(&inp.toks, target, lr),
        }
    }

    /// (embedding table, which rows are trained, dim)
    fn emb_ref(&self) -> (&Vec<f32>, &Vec<bool>, usize) {
        match self {
            Net::Bow(n) => (&n.emb, &n.touched, n.cfg.dim),
            Net::Cnn(n) => (&n.emb.w, &n.emb.touched, n.cfg.dim),
            Net::Gru(n) => (&n.emb.w, &n.emb.touched, n.cfg.dim),
        }
    }

    fn emb_mut(&mut self) -> (&mut Vec<f32>, &mut Vec<bool>) {
        match self {
            Net::Bow(n) => (&mut n.emb, &mut n.touched),
            Net::Cnn(n) => (&mut n.emb.w, &mut n.emb.touched),
            Net::Gru(n) => (&mut n.emb.w, &mut n.emb.touched),
        }
    }

    fn dense(&self) -> Vec<&Vec<f32>> {
        match self {
            Net::Bow(n) => vec![&n.w1, &n.b1, &n.w2, &n.b2],
            Net::Cnn(n) => n.ps.iter().map(|p| &p.w).collect(),
            Net::Gru(n) => n.ps.iter().map(|p| &p.w).collect(),
        }
    }

    fn dense_mut(&mut self) -> Vec<&mut Vec<f32>> {
        match self {
            Net::Bow(n) => vec![&mut n.w1, &mut n.b1, &mut n.w2, &mut n.b2],
            Net::Cnn(n) => n.ps.iter_mut().map(|p| &mut p.w).collect(),
            Net::Gru(n) => n.ps.iter_mut().map(|p| &mut p.w).collect(),
        }
    }

    pub fn live_params(&self) -> usize {
        let (_, touched, dim) = self.emb_ref();
        touched.iter().filter(|&&t| t).count() * dim + self.dense().iter().map(|a| a.len()).sum::<usize>()
    }

    /// Snap embedding rows to the int8 grid `save` stores them on.
    fn quantize(&mut self) {
        let dim = self.emb_ref().2;
        let touched: Vec<bool> = self.emb_ref().1.clone();
        let (w, _) = self.emb_mut();
        for (i, &t) in touched.iter().enumerate() {
            if t {
                let (scale, q) = quant_row(&w[i * dim..(i + 1) * dim]);
                for (d, v) in q.iter().enumerate() {
                    w[i * dim + d] = *v as f32 * scale;
                }
            }
        }
    }
}

fn train_net(examples: &[(usize, String)], clean: &[Input], c: usize, cfg: Cfg, seed: u64) -> Net {
    let mut rng = Rng::new(seed);
    let mut net = Net::new(c, cfg, &mut rng);
    let targets: Vec<Vec<f32>> = examples.iter().map(|(y, _)| hard_target(*y, c)).collect();
    let mut order: Vec<usize> = (0..examples.len()).collect();
    for e in 0..cfg.epochs {
        let lr = 0.12 * (1.0 - e as f32 / cfg.epochs as f32) + 0.005;
        rng.shuffle(&mut order);
        for &i in &order {
            let noisy;
            let inp: &Input = if rng.f32() < 0.5 {
                noisy = Input::of(&noise(&examples[i].1, &mut rng));
                &noisy
            } else {
                &clean[i]
            };
            net.step(inp, &targets[i], lr);
        }
    }
    net
}

pub struct Pred {
    pub probs: Vec<f32>,
    pub cov: f32,
}

pub struct Ensemble {
    pub tags: Vec<String>,
    pub seen: Vec<bool>,
    pub nets: Vec<Net>,
    pub temp: f32,
    /// cosine similarity the nearest known phrase needs before "did you mean...?" is worth asking;
    /// calibrated per model at training time because each architecture's embeddings have their own scale
    pub clarify_sim: f32,
    pub data_hash: u64,
}

pub fn data_hash(tags: &[String], examples: &[(usize, String)], salt: u64) -> u64 {
    let mut h = fnv(&salt.to_le_bytes(), FNV_INIT);
    for t in tags {
        h = fnv(t.as_bytes(), h);
        h = fnv(&[0], h);
    }
    for (k, e) in examples {
        h = fnv(&(*k as u32).to_le_bytes(), h);
        h = fnv(e.as_bytes(), h);
        h = fnv(&[0], h);
    }
    h
}

/// Stable identity of a configuration, so models of different sizes never get mixed up on disk.
pub fn cfg_salt(cfg: &Cfg) -> u64 {
    let mut h = FNV_INIT;
    for v in [cfg.arch.id() as usize, cfg.dim, cfg.hid, cfg.bits as usize, cfg.ch, cfg.epochs] {
        h = fnv(&(v as u32).to_le_bytes(), h);
    }
    h
}

fn seen_table(clean: &[Input]) -> Vec<bool> {
    let mut seen = vec![false; BUCKETS];
    for f in clean.iter().flat_map(|i| i.flat.iter()) {
        seen[f & (BUCKETS - 1)] = true;
    }
    seen
}

impl Ensemble {
    /// One net per config, trained in parallel (one thread each) with different seeds.
    pub fn train(examples: &[(usize, String)], tags: Vec<String>, cfgs: &[Cfg], seed: u64, temp: f32, salt: u64) -> Ensemble {
        let c = tags.len();
        let clean: Vec<Input> = examples.iter().map(|(_, t)| Input::of(t)).collect();
        let nets: Vec<Net> = std::thread::scope(|s| {
            let hs: Vec<_> = cfgs
                .iter()
                .enumerate()
                .map(|(i, cfg)| {
                    let (ex, cl, cfg) = (examples, &clean, *cfg);
                    s.spawn(move || train_net(ex, cl, c, cfg, seed.wrapping_add(i as u64 * 7919)))
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("trainer thread panicked")).collect()
        });
        Ensemble { data_hash: data_hash(&tags, examples, salt), seen: seen_table(&clean), tags, nets, temp, clarify_sim: 0.75 }
    }

    /// Knowledge distillation: the teacher labels a large synthetic pool (typos, dropped words,
    /// spliced phrases) and one small net learns to reproduce its soft predictions.
    pub fn distill(teacher: &Ensemble, examples: &[(usize, String)], cfg: Cfg, seed: u64, temp: f32, salt: u64) -> Ensemble {
        let c = teacher.tags.len();
        let mut rng = Rng::new(seed ^ 0xD15);
        let mut pool: Vec<(String, Option<usize>)> = Vec::new();
        for (y, t) in examples {
            pool.push((t.clone(), Some(*y)));
            for _ in 0..6 {
                pool.push((noise(t, &mut rng), None));
            }
            let ws: Vec<&str> = t.split_whitespace().collect();
            if ws.len() > 2 {
                for _ in 0..2 {
                    let drop = rng.below(ws.len());
                    pool.push((ws.iter().enumerate().filter(|(i, _)| *i != drop).map(|(_, w)| *w).collect::<Vec<_>>().join(" "), None));
                }
            }
        }
        for _ in 0..examples.len() * 3 {
            let a: Vec<&str> = examples[rng.below(examples.len())].1.split_whitespace().collect();
            let b: Vec<&str> = examples[rng.below(examples.len())].1.split_whitespace().collect();
            let (ka, kb) = (1 + rng.below(a.len()), rng.below(b.len()));
            let mut w: Vec<&str> = a[..ka].to_vec();
            w.extend_from_slice(&b[kb..]);
            pool.push((w.join(" "), None));
        }
        let inputs: Vec<Input> = pool.iter().map(|(t, _)| Input::of(t)).collect();
        // teacher soft labels, softened a little so the student sees the "dark knowledge"
        let targets: Vec<Vec<f32>> = pool
            .iter()
            .zip(&inputs)
            .map(|((_, hard), inp)| {
                let mut p = teacher.raw_mean(inp);
                let mut s = 0.0;
                for x in p.iter_mut() {
                    *x = (*x + 1e-9).powf(0.7);
                    s += *x;
                }
                p.iter_mut().for_each(|x| *x /= s);
                if let Some(y) = hard {
                    let h = hard_target(*y, c);
                    for k in 0..c {
                        p[k] = 0.5 * p[k] + 0.5 * h[k];
                    }
                }
                p
            })
            .collect();
        let mut net = Net::new(c, cfg, &mut rng);
        let mut order: Vec<usize> = (0..pool.len()).collect();
        for e in 0..cfg.epochs {
            let lr = 0.10 * (1.0 - e as f32 / cfg.epochs as f32) + 0.004;
            rng.shuffle(&mut order);
            for &i in &order {
                net.step(&inputs[i], &targets[i], lr);
            }
        }
        net.quantize();
        let clean: Vec<Input> = examples.iter().map(|(_, t)| Input::of(t)).collect();
        Ensemble { data_hash: data_hash(&teacher.tags, examples, salt), seen: seen_table(&clean), tags: teacher.tags.clone(), nets: vec![net], temp, clarify_sim: teacher.clarify_sim }
    }

    fn raw_mean(&self, inp: &Input) -> Vec<f32> {
        let c = self.tags.len();
        let mut p = vec![0.0; c];
        for net in &self.nets {
            let q = net.infer(inp).probs;
            for k in 0..c {
                p[k] += q[k] / self.nets.len() as f32;
            }
        }
        p
    }

    fn calibrate(&self, p: &mut [f32]) {
        if (self.temp - 1.0).abs() < 1e-3 {
            return;
        }
        let inv = 1.0 / self.temp;
        let mut s = 0.0;
        for x in p.iter_mut() {
            *x = (*x + 1e-9).powf(inv);
            s += *x;
        }
        p.iter_mut().for_each(|x| *x /= s);
    }

    pub fn predict(&self, text: &str) -> Pred {
        let inp = Input::of(text);
        if inp.flat.is_empty() {
            return Pred { probs: vec![1.0 / self.tags.len() as f32; self.tags.len()], cov: 0.0 };
        }
        let cov = inp.flat.iter().filter(|&&i| self.seen[i & (BUCKETS - 1)]).count() as f32 / inp.flat.len() as f32;
        let mut probs = self.raw_mean(&inp);
        self.calibrate(&mut probs);
        Pred { probs, cov }
    }

    /// Hidden-layer activations of the first net, average-pooled down to at most 32 cells for the heatmap.
    pub fn activations(&self, text: &str) -> Vec<f32> {
        let h = self.nets[0].infer(&Input::of(text)).hid;
        if h.len() <= 32 {
            return h;
        }
        let per = h.len().div_ceil(32);
        h.chunks(per).map(|c| c.iter().sum::<f32>() / c.len() as f32).collect()
    }

    /// Sentence vector for nearest-phrase lookup: concatenated, L2-normalised pooled representations.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let inp = Input::of(text);
        let mut out = Vec::new();
        for net in &self.nets {
            let mut e = net.infer(&inp).emb;
            let n = e.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            e.iter_mut().for_each(|x| *x /= n);
            out.extend(e);
        }
        out
    }

    /// Grid-search the temperature that minimises NLL on held-out data.
    pub fn fit_temp(&mut self, val: &[(usize, String)]) {
        if val.is_empty() {
            return;
        }
        self.temp = 1.0;
        let raws: Vec<(usize, Vec<f32>)> = val.iter().map(|(k, t)| (*k, self.raw_mean(&Input::of(t)))).collect();
        let (mut best_t, mut best) = (1.0, f32::MAX);
        for i in 4..=50 {
            let t = i as f32 * 0.1;
            let mut nll = 0.0;
            for (k, p) in &raws {
                let mut q = p.clone();
                let mut s = 0.0;
                for x in q.iter_mut() {
                    *x = (*x + 1e-9).powf(1.0 / t);
                    s += *x;
                }
                nll -= (q[*k] / s).max(1e-9).ln();
            }
            if nll < best {
                best = nll;
                best_t = t;
            }
        }
        self.temp = best_t;
    }

    pub fn cfg(&self) -> Cfg {
        self.nets[0].cfg()
    }

    /// parameters allocated across all nets
    pub fn nominal_params(&self) -> usize {
        self.nets.iter().map(|n| n.cfg().nominal(self.tags.len())).sum()
    }

    /// (trained embedding params, dense params, total live params)
    pub fn param_report(&self) -> (usize, usize, usize) {
        let total: usize = self.nets.iter().map(|n| n.live_params()).sum();
        let dense: usize = self.nets.iter().map(|n| n.dense().iter().map(|a| a.len()).sum::<usize>()).sum();
        (total - dense, dense, total)
    }

    // ---------- serialisation: int8 rows (per-row scale) for trained rows only, bitmaps packed ----------
    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut b = MAGIC.to_vec();
        for v in [self.tags.len(), self.nets.len()] {
            b.extend_from_slice(&(v as u32).to_le_bytes());
        }
        b.extend_from_slice(&self.temp.to_le_bytes());
        b.extend_from_slice(&self.clarify_sim.to_le_bytes());
        b.extend_from_slice(&self.data_hash.to_le_bytes());
        for t in &self.tags {
            b.extend_from_slice(&(t.len() as u32).to_le_bytes());
            b.extend_from_slice(t.as_bytes());
        }
        pack_bits(&self.seen, &mut b);
        for net in &self.nets {
            let cfg = net.cfg();
            b.push(cfg.arch.id());
            for v in [cfg.dim, cfg.hid, cfg.bits as usize, cfg.ch] {
                b.extend_from_slice(&(v as u32).to_le_bytes());
            }
            let (w, touched, dim) = net.emb_ref();
            pack_bits(touched, &mut b);
            for (i, &t) in touched.iter().enumerate() {
                if t {
                    let (scale, q) = quant_row(&w[i * dim..(i + 1) * dim]);
                    b.extend_from_slice(&scale.to_le_bytes());
                    b.extend(q.iter().map(|&x| x as u8));
                }
            }
            for arr in net.dense() {
                for x in arr.iter() {
                    b.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        let tmp = format!("{path}.tmp");
        fs::write(&tmp, b)?;
        fs::rename(tmp, path)
    }

    pub fn load(path: &str) -> Option<Ensemble> {
        let data = fs::read(path).ok()?;
        let mut r = Reader { d: &data, p: 0 };
        if r.take(MAGIC.len())? != MAGIC {
            return None;
        }
        let (c, n) = (r.u32()?, r.u32()?);
        if c == 0 || c > 4096 || n == 0 || n > 64 {
            return None;
        }
        let temp = f32::from_le_bytes(r.take(4)?.try_into().ok()?);
        let clarify_sim = f32::from_le_bytes(r.take(4)?.try_into().ok()?);
        let data_hash = u64::from_le_bytes(r.take(8)?.try_into().ok()?);
        let mut tags = Vec::new();
        for _ in 0..c {
            let l = r.u32()?;
            tags.push(String::from_utf8(r.take(l)?.to_vec()).ok()?);
        }
        let seen = unpack_bits(&mut r, BUCKETS)?;
        let mut nets = Vec::new();
        for _ in 0..n {
            let arch = Arch::from_id(r.take(1)?[0])?;
            let (dim, hid, bits, ch) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?);
            if !(2..=512).contains(&dim) || !(2..=1024).contains(&hid) || !(6..=16).contains(&bits) || ch > 1024 || (arch != Arch::Bow && ch == 0) {
                return None;
            }
            let cfg = Cfg { arch, dim, hid, bits: bits as u32, ch, epochs: student_epochs(arch) };
            let mut net = Net::new(c, cfg, &mut Rng::new(0));
            let rows = 1usize << bits;
            let touched = unpack_bits(&mut r, rows)?;
            {
                let (w, t) = net.emb_mut();
                w.fill(0.0);
                for (i, &tr) in touched.iter().enumerate() {
                    if tr {
                        let scale = f32::from_le_bytes(r.take(4)?.try_into().ok()?);
                        for (d, &q) in r.take(dim)?.iter().enumerate() {
                            w[i * dim + d] = (q as i8) as f32 * scale;
                        }
                    }
                }
                *t = touched;
            }
            for arr in net.dense_mut() {
                let len = arr.len();
                *arr = r.f32s(len)?;
            }
            nets.push(net);
        }
        Some(Ensemble { tags, seen, nets, temp, clarify_sim, data_hash })
    }
}

fn pack_bits(bits: &[bool], out: &mut Vec<u8>) {
    for chunk in bits.chunks(8) {
        out.push(chunk.iter().enumerate().fold(0u8, |a, (i, &b)| a | ((b as u8) << i)));
    }
}

fn unpack_bits(r: &mut Reader, n: usize) -> Option<Vec<bool>> {
    let bytes = r.take(n.div_ceil(8))?;
    Some((0..n).map(|i| bytes[i / 8] >> (i % 8) & 1 == 1).collect())
}

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let e = self.p.checked_add(n)?;
        let s = self.d.get(self.p..e)?;
        self.p = e;
        Some(s)
    }
    fn u32(&mut self) -> Option<usize> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?) as usize)
    }
    fn f32s(&mut self, n: usize) -> Option<Vec<f32>> {
        let b = self.take(n.checked_mul(4)?)?;
        Some(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

/// Microseconds per training step (forward + backward + update) on a typical short utterance.
pub fn bench_step_us(cfg: Cfg, c: usize) -> f32 {
    let mut rng = Rng::new(1);
    let mut net = Net::new(c, cfg, &mut rng);
    let inp = Input::of("hello there how are you doing today");
    let tgt = hard_target(1, c);
    let t = std::time::Instant::now();
    for _ in 0..150 {
        net.step(&inp, &tgt, 0.1);
    }
    t.elapsed().as_secs_f32() * 1e6 / 150.0
}
