//! Bagged fastText-style nets: hashed word/bigram/char-ngram embeddings -> two mean-pooled
//! channels -> ReLU layer -> softmax. Hand-written backprop + SGD.
//!
//! Parameter efficiency: a big "teacher" ensemble is trained once, then its knowledge is
//! distilled (soft targets on noised / spliced synthetic text it labels for free) into a
//! single compact "student" net. The student ships, stored as int8 rows.

use crate::rng::Rng;
use crate::text::{bucket_in, features, fnv, noise, BUCKETS, CHAR_BASE, FNV_INIT};
use std::{fs, io};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cfg {
    pub dim: usize,
    pub hid: usize,
    pub bits: u32, // embedding rows = 1 << bits
    pub epochs: usize,
}

pub const TEACHER: Cfg = Cfg { dim: 48, hid: 64, bits: 14, epochs: 160 };
pub const STUDENT: Cfg = Cfg { dim: 16, hid: 32, bits: 12, epochs: 40 };
pub const TEACHER_NETS: usize = 5;

const EMB_BOOST: f32 = 4.0;
const SMOOTH: f32 = 0.04;
const MAGIC: &[u8] = b"TINYBOT4";

// decision thresholds (calibrated probability)
pub const P_ANSWER: f32 = 0.55;
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
    if p[k] >= P_ANSWER {
        Decision::Answer(k)
    } else if p[k] >= P_CLARIFY {
        Decision::Clarify(k)
    } else {
        Decision::Reject
    }
}

fn softmax(v: &mut [f32]) {
    let m = v.iter().cloned().fold(f32::MIN, f32::max);
    let mut s = 0.0;
    for x in v.iter_mut() {
        *x = (*x - m).exp();
        s += *x;
    }
    for x in v.iter_mut() {
        *x /= s;
    }
}

fn hard_target(y: usize, c: usize) -> Vec<f32> {
    let off = SMOOTH / (c.max(2) - 1) as f32;
    (0..c).map(|k| if k == y { 1.0 - SMOOTH } else { off }).collect()
}

pub struct Net {
    cfg: Cfg,
    emb: Vec<f32>, // rows x dim
    w1: Vec<f32>,  // 2*dim x hid
    b1: Vec<f32>,
    w2: Vec<f32>, // hid x classes
    b2: Vec<f32>,
    touched: Vec<bool>, // rows that ever received a gradient (the rest are untrained noise)
}

impl Net {
    fn new(c: usize, cfg: Cfg, rng: &mut Rng) -> Net {
        let rows = 1usize << cfg.bits;
        let inp = 2 * cfg.dim;
        let mut uni = |n: usize, lim: f32| -> Vec<f32> { (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * lim).collect() };
        Net {
            cfg,
            emb: uni(rows * cfg.dim, 0.3),
            w1: uni(inp * cfg.hid, (6.0 / inp as f32).sqrt()),
            b1: vec![0.0; cfg.hid],
            w2: uni(cfg.hid * c, (6.0 / cfg.hid as f32).sqrt()),
            b2: vec![0.0; c],
            touched: vec![false; rows],
        }
    }

    fn classes(&self) -> usize {
        self.b2.len()
    }

    /// (pooled input, hidden activations, class probabilities). Word/bigram features and
    /// char n-grams pool into separate halves so one decisive word isn't averaged away.
    fn forward(&self, feats: &[usize]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (dim, hid) = (self.cfg.dim, self.cfg.hid);
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
        let c = self.classes();
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

    /// Inference: rows that never trained are dropped instead of adding their random init.
    fn infer(&self, feats: &[usize]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let kept: Vec<usize> = feats.iter().cloned().filter(|&f| self.touched[bucket_in(f, self.cfg.bits)]).collect();
        self.forward(&kept)
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

    fn live_params(&self) -> usize {
        self.touched.iter().filter(|&&t| t).count() * self.cfg.dim + self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    /// Snap embedding rows to an int8 grid (per-row scale) exactly as `save` stores them.
    fn quantize(&mut self) {
        let dim = self.cfg.dim;
        for (i, &t) in self.touched.iter().enumerate() {
            if t {
                let (scale, q) = quant_row(&self.emb[i * dim..(i + 1) * dim]);
                for (d, v) in q.iter().enumerate() {
                    self.emb[i * dim + d] = *v as f32 * scale;
                }
            }
        }
    }
}

fn quant_row(r: &[f32]) -> (f32, Vec<i8>) {
    let m = r.iter().fold(0.0f32, |a, &x| a.max(x.abs())).max(1e-8);
    let scale = m / 127.0;
    (scale, r.iter().map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8).collect())
}

fn train_net(examples: &[(usize, String)], clean: &[Vec<usize>], c: usize, cfg: Cfg, seed: u64) -> Net {
    let mut rng = Rng::new(seed);
    let mut net = Net::new(c, cfg, &mut rng);
    let targets: Vec<Vec<f32>> = examples.iter().map(|(y, _)| hard_target(*y, c)).collect();
    let mut order: Vec<usize> = (0..examples.len()).collect();
    for e in 0..cfg.epochs {
        let lr = 0.12 * (1.0 - e as f32 / cfg.epochs as f32) + 0.005;
        rng.shuffle(&mut order);
        for &i in &order {
            let noisy;
            let feats: &[usize] = if rng.f32() < 0.5 {
                noisy = features(&noise(&examples[i].1, &mut rng));
                &noisy
            } else {
                &clean[i]
            };
            net.step(feats, &targets[i], lr);
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
    pub cfg: Cfg,
    pub temp: f32,
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
    for v in [STUDENT.dim, STUDENT.hid, STUDENT.bits as usize, STUDENT.epochs, TEACHER.dim, TEACHER.epochs, TEACHER_NETS] {
        h = fnv(&(v as u32).to_le_bytes(), h);
    }
    h
}

fn seen_table(clean: &[Vec<usize>]) -> Vec<bool> {
    let mut seen = vec![false; BUCKETS];
    for f in clean.iter().flatten() {
        seen[f & (BUCKETS - 1)] = true;
    }
    seen
}

impl Ensemble {
    /// `n` nets in parallel, one thread each, different seeds.
    pub fn train(examples: &[(usize, String)], tags: Vec<String>, n: usize, cfg: Cfg, seed: u64, temp: f32, salt: u64) -> Ensemble {
        let c = tags.len();
        let clean: Vec<Vec<usize>> = examples.iter().map(|(_, t)| features(t)).collect();
        let nets: Vec<Net> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..n)
                .map(|i| {
                    let (ex, cl) = (examples, &clean);
                    s.spawn(move || train_net(ex, cl, c, cfg, seed.wrapping_add(i as u64 * 7919)))
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("trainer thread panicked")).collect()
        });
        Ensemble { data_hash: data_hash(&tags, examples, salt), seen: seen_table(&clean), tags, nets, cfg, temp }
    }

    /// Knowledge distillation: the teacher labels a large synthetic pool (typos, dropped words,
    /// spliced phrases) and one small net learns to reproduce its soft predictions.
    pub fn distill(teacher: &Ensemble, examples: &[(usize, String)], cfg: Cfg, seed: u64, temp: f32, salt: u64) -> Ensemble {
        let c = teacher.tags.len();
        let mut rng = Rng::new(seed ^ 0xD15);
        // (text, hard label if it is a real example, weight of the hard label)
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
        let feats: Vec<Vec<usize>> = pool.iter().map(|(t, _)| features(t)).collect();
        // teacher soft labels (softened a little so the student sees the "dark knowledge")
        let targets: Vec<Vec<f32>> = pool
            .iter()
            .map(|(t, hard)| {
                let mut p = teacher.raw_mean(&features(t));
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
                net.step(&feats[i], &targets[i], lr);
            }
        }
        net.quantize();
        let clean: Vec<Vec<usize>> = examples.iter().map(|(_, t)| features(t)).collect();
        Ensemble { data_hash: data_hash(&teacher.tags, examples, salt), seen: seen_table(&clean), tags: teacher.tags.clone(), nets: vec![net], cfg, temp }
    }

    fn raw_mean(&self, feats: &[usize]) -> Vec<f32> {
        let c = self.tags.len();
        let mut p = vec![0.0; c];
        for net in &self.nets {
            let (_, _, q) = net.infer(feats);
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
        let f = features(text);
        if f.is_empty() {
            return Pred { probs: vec![1.0 / self.tags.len() as f32; self.tags.len()], cov: 0.0 };
        }
        let cov = f.iter().filter(|&&i| self.seen[i & (BUCKETS - 1)]).count() as f32 / f.len() as f32;
        let mut probs = self.raw_mean(&f);
        self.calibrate(&mut probs);
        Pred { probs, cov }
    }

    /// Hidden-layer activations of the first net.
    pub fn activations(&self, text: &str) -> Vec<f32> {
        self.nets[0].infer(&features(text)).1
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        let f = features(text);
        let mut out = Vec::new();
        for net in &self.nets {
            let (mut h0, _, _) = net.infer(&f);
            let n = h0.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            h0.iter_mut().for_each(|x| *x /= n);
            out.extend(h0);
        }
        out
    }

    /// Grid-search the temperature that minimises NLL on held-out data.
    pub fn fit_temp(&mut self, val: &[(usize, String)]) {
        if val.is_empty() {
            return;
        }
        self.temp = 1.0;
        let raws: Vec<(usize, Vec<f32>)> = val.iter().map(|(k, t)| (*k, self.raw_mean(&features(t)))).collect();
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

    /// (trained embedding params, dense params, total live params)
    pub fn param_report(&self) -> (usize, usize, usize) {
        let total: usize = self.nets.iter().map(|n| n.live_params()).sum();
        let dense: usize = self.nets.iter().map(|n| n.w1.len() + n.b1.len() + n.w2.len() + n.b2.len()).sum();
        (total - dense, dense, total)
    }

    // ---------- serialisation: int8 rows (per-row scale) for trained rows only ----------
    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut b = MAGIC.to_vec();
        for v in [self.cfg.dim, self.cfg.hid, self.cfg.bits as usize, self.tags.len(), self.nets.len()] {
            b.extend_from_slice(&(v as u32).to_le_bytes());
        }
        b.extend_from_slice(&self.temp.to_le_bytes());
        b.extend_from_slice(&self.data_hash.to_le_bytes());
        for t in &self.tags {
            b.extend_from_slice(&(t.len() as u32).to_le_bytes());
            b.extend_from_slice(t.as_bytes());
        }
        b.extend(self.seen.iter().map(|&s| s as u8));
        for net in &self.nets {
            b.extend(net.touched.iter().map(|&s| s as u8));
            for (i, &t) in net.touched.iter().enumerate() {
                if t {
                    let (scale, q) = quant_row(&net.emb[i * self.cfg.dim..(i + 1) * self.cfg.dim]);
                    b.extend_from_slice(&scale.to_le_bytes());
                    b.extend(q.iter().map(|&x| x as u8));
                }
            }
            for arr in [&net.w1, &net.b1, &net.w2, &net.b2] {
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
        let (dim, hid, bits, c, n) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?, r.u32()?);
        if !(4..=256).contains(&dim) || !(4..=1024).contains(&hid) || !(6..=16).contains(&bits) || c == 0 || c > 4096 || n == 0 || n > 64 {
            return None;
        }
        let cfg = Cfg { dim, hid, bits: bits as u32, epochs: 0 };
        let rows = 1usize << bits;
        let temp = f32::from_le_bytes(r.take(4)?.try_into().ok()?);
        let data_hash = u64::from_le_bytes(r.take(8)?.try_into().ok()?);
        let mut tags = Vec::new();
        for _ in 0..c {
            let l = r.u32()?;
            tags.push(String::from_utf8(r.take(l)?.to_vec()).ok()?);
        }
        let seen = r.take(BUCKETS)?.iter().map(|&b| b != 0).collect();
        let mut nets = Vec::new();
        for _ in 0..n {
            let touched: Vec<bool> = r.take(rows)?.iter().map(|&b| b != 0).collect();
            let mut emb = vec![0.0; rows * dim];
            for (i, &t) in touched.iter().enumerate() {
                if t {
                    let scale = f32::from_le_bytes(r.take(4)?.try_into().ok()?);
                    for (d, &q) in r.take(dim)?.iter().enumerate() {
                        emb[i * dim + d] = (q as i8) as f32 * scale;
                    }
                }
            }
            nets.push(Net { cfg, emb, touched, w1: r.f32s(2 * dim * hid)?, b1: r.f32s(hid)?, w2: r.f32s(hid * c)?, b2: r.f32s(c)? });
        }
        Some(Ensemble { tags, seen, nets, cfg, temp, data_hash })
    }
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
