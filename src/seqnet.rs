//! Sequence models for TinyBot: a temporal CNN and a bidirectional GRU.
//!
//! Both read each token as [word channel | char-ngram channel] vectors pulled from a hashed
//! embedding table (so typos still land near their word), keep word order, and finish with the same
//! small dense head as the bag-of-features net. Everything is hand-written: forward passes, backprop
//! through time for the GRU, Adam for the dense weights, plain SGD for the embedding rows. The tests
//! check every gradient against finite differences.

use crate::model::Cfg;
use crate::rng::Rng;
use crate::text::{bucket_in, Tok, MAXT};

pub struct Out {
    pub emb: Vec<f32>, // pooled representation before the dense head (for nearest-phrase lookup)
    pub hid: Vec<f32>, // hidden layer activations (for the neuron heatmap)
    pub probs: Vec<f32>,
}

// ---------------- small linear algebra ----------------
pub fn softmax(v: &mut [f32]) {
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

fn glorot(n: usize, fan_in: usize, fan_out: usize, rng: &mut Rng) -> Vec<f32> {
    let lim = (6.0 / (fan_in + fan_out) as f32).sqrt();
    (0..n).map(|_| (rng.f32() * 2.0 - 1.0) * lim).collect()
}

/// out += W x   (W is rows x cols, row-major)
fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut s = 0.0;
        for c in 0..cols {
            s += row[c] * x[c];
        }
        out[r] += s;
    }
}

/// dx += W^T dy
fn matvec_t(w: &[f32], rows: usize, cols: usize, dy: &[f32], dx: &mut [f32]) {
    for r in 0..rows {
        let d = dy[r];
        if d == 0.0 {
            continue;
        }
        let row = &w[r * cols..(r + 1) * cols];
        for c in 0..cols {
            dx[c] += row[c] * d;
        }
    }
}

/// G += dy x^T
fn outer_add(g: &mut [f32], cols: usize, dy: &[f32], x: &[f32]) {
    for (r, &d) in dy.iter().enumerate() {
        if d == 0.0 {
            continue;
        }
        let row = &mut g[r * cols..(r + 1) * cols];
        for c in 0..cols {
            row[c] += d * x[c];
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------- parameters ----------------
pub struct P {
    pub w: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl P {
    fn new(w: Vec<f32>) -> P {
        let n = w.len();
        P { w, m: vec![0.0; n], v: vec![0.0; n] }
    }
    fn zeros(n: usize) -> P {
        P::new(vec![0.0; n])
    }
    fn adam(&mut self, g: &[f32], lr: f32, t: u32, clip: f32) {
        let (b1, b2) = (0.9f32, 0.999f32);
        let (c1, c2) = (1.0 - b1.powi(t as i32), 1.0 - b2.powi(t as i32));
        for i in 0..self.w.len() {
            let gi = g[i] * clip;
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
            self.w[i] -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
    }
}

// ---------------- hashed embedding table ----------------
pub struct EmbTable {
    pub dim: usize,
    pub bits: u32,
    pub w: Vec<f32>,
    pub touched: Vec<bool>,
}

impl EmbTable {
    pub fn new(dim: usize, bits: u32, rng: &mut Rng) -> EmbTable {
        let rows = 1usize << bits;
        EmbTable { dim, bits, w: (0..rows * dim).map(|_| (rng.f32() * 2.0 - 1.0) * 0.3).collect(), touched: vec![false; rows] }
    }

    /// mean of the rows for `feats` into `out`; at inference rows that never trained are skipped
    fn pool(&self, feats: &[usize], filter: bool, out: &mut [f32]) {
        out.fill(0.0);
        let mut n = 0;
        for &f in feats {
            let b = bucket_in(f, self.bits);
            if filter && !self.touched[b] {
                continue;
            }
            let r = &self.w[b * self.dim..(b + 1) * self.dim];
            for d in 0..self.dim {
                out[d] += r[d];
            }
            n += 1;
        }
        if n > 0 {
            let inv = 1.0 / n as f32;
            out.iter_mut().for_each(|v| *v *= inv);
        }
    }

    fn mark(&mut self, feats: &[usize]) {
        for &f in feats {
            self.touched[bucket_in(f, self.bits)] = true;
        }
    }

    fn scatter(&mut self, feats: &[usize], d: &[f32], lr: f32) {
        if feats.is_empty() {
            return;
        }
        let k = lr / feats.len() as f32;
        for &f in feats {
            let b = bucket_in(f, self.bits);
            let r = &mut self.w[b * self.dim..(b + 1) * self.dim];
            for j in 0..self.dim {
                r[j] -= k * d[j];
            }
        }
    }
}

/// tokens -> T x (2*dim): [word channel | char channel]
fn embed_tokens(emb: &EmbTable, toks: &[Tok], filter: bool, pad_to: usize) -> (Vec<f32>, usize) {
    let (dim, t) = (emb.dim, toks.len().min(MAXT));
    let in_ = 2 * dim;
    let tp = t.max(pad_to);
    let mut xs = vec![0.0; tp * in_];
    for (i, tok) in toks.iter().take(t).enumerate() {
        emb.pool(&tok.w, filter, &mut xs[i * in_..i * in_ + dim]);
        emb.pool(&tok.c, filter, &mut xs[i * in_ + dim..(i + 1) * in_]);
    }
    (xs, t)
}

fn scatter_tokens(emb: &mut EmbTable, toks: &[Tok], dx: &[f32], lr: f32) {
    let (dim, in_) = (emb.dim, 2 * emb.dim);
    for (i, tok) in toks.iter().take(MAXT).enumerate() {
        emb.scatter(&tok.w, &dx[i * in_..i * in_ + dim], lr);
        emb.scatter(&tok.c, &dx[i * in_ + dim..(i + 1) * in_], lr);
    }
}

// ---------------- shared dense head: pooled -> ReLU hidden -> softmax ----------------
// ps[iw] = Wh (hid x pin), ps[iw+1] = bh, ps[iw+2] = Wo (c x hid), ps[iw+3] = bo
fn head_forward(ps: &[P], iw: usize, pin: usize, hid: usize, c: usize, p: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut z1 = ps[iw + 1].w.clone();
    matvec(&ps[iw].w, hid, pin, p, &mut z1);
    z1.iter_mut().for_each(|v| *v = v.max(0.0));
    let mut lg = ps[iw + 3].w.clone();
    matvec(&ps[iw + 2].w, c, hid, &z1, &mut lg);
    softmax(&mut lg);
    (z1, lg)
}

fn head_backward(ps: &[P], gs: &mut [Vec<f32>], iw: usize, pin: usize, hid: usize, c: usize, p: &[f32], z1: &[f32], dl: &[f32]) -> Vec<f32> {
    outer_add(&mut gs[iw + 2], hid, dl, z1);
    for k in 0..c {
        gs[iw + 3][k] += dl[k];
    }
    let mut dz = vec![0.0; hid];
    matvec_t(&ps[iw + 2].w, c, hid, dl, &mut dz);
    for j in 0..hid {
        if z1[j] <= 0.0 {
            dz[j] = 0.0;
        }
    }
    outer_add(&mut gs[iw], pin, &dz, p);
    for j in 0..hid {
        gs[iw + 1][j] += dz[j];
    }
    let mut dp = vec![0.0; pin];
    matvec_t(&ps[iw].w, hid, pin, &dz, &mut dp);
    dp
}

fn ce_loss(probs: &[f32], target: &[f32]) -> f32 {
    -probs.iter().zip(target).map(|(p, t)| t * (p + 1e-9).ln()).sum::<f32>()
}

fn adam_all(ps: &mut [P], gs: &[Vec<f32>], lr_scale: f32, steps: &mut u32) {
    let norm = gs.iter().flat_map(|g| g.iter()).map(|x| x * x).sum::<f32>().sqrt();
    let clip = if norm > 5.0 { 5.0 / norm } else { 1.0 };
    *steps += 1;
    for (p, g) in ps.iter_mut().zip(gs) {
        p.adam(g, 0.003 * lr_scale, *steps, clip);
    }
}

// ============================ temporal CNN ============================
const KS: [usize; 3] = [1, 2, 3];

pub struct CnnNet {
    pub cfg: Cfg,
    pub emb: EmbTable,
    pub ps: Vec<P>, // W1, W2, W3, bias(3F), head x4
    gs: Vec<Vec<f32>>,
    c: usize,
    steps: u32,
}

struct CnnFwd {
    xs: Vec<f32>,
    pooled: Vec<f32>,
    arg: Vec<i32>,
    z1: Vec<f32>,
    probs: Vec<f32>,
    t: usize,
}

impl CnnNet {
    pub fn new(c: usize, cfg: Cfg, rng: &mut Rng) -> CnnNet {
        let (in_, f) = (2 * cfg.dim, cfg.ch);
        let mut ps = Vec::new();
        for k in KS {
            ps.push(P::new(glorot(f * k * in_, k * in_, f, rng)));
        }
        ps.push(P::zeros(3 * f));
        ps.push(P::new(glorot(cfg.hid * 3 * f, 3 * f, cfg.hid, rng)));
        ps.push(P::zeros(cfg.hid));
        ps.push(P::new(glorot(c * cfg.hid, cfg.hid, c, rng)));
        ps.push(P::zeros(c));
        let gs = ps.iter().map(|p| vec![0.0; p.w.len()]).collect();
        CnnNet { cfg, emb: EmbTable::new(cfg.dim, cfg.bits, rng), ps, gs, c, steps: 0 }
    }

    fn forward(&self, toks: &[Tok], filter: bool) -> CnnFwd {
        let (in_, f) = (2 * self.cfg.dim, self.cfg.ch);
        let (xs, t) = embed_tokens(&self.emb, toks, filter, 3);
        let tp = t.max(3);
        let mut pooled = vec![0.0; 3 * f];
        let mut arg = vec![-1i32; 3 * f];
        for (kk, k) in KS.iter().enumerate() {
            let w = &self.ps[kk].w;
            for pos in 0..=(tp - k) {
                let win = &xs[pos * in_..(pos + k) * in_];
                for fi in 0..f {
                    let row = &w[fi * k * in_..(fi + 1) * k * in_];
                    let mut s = self.ps[3].w[kk * f + fi];
                    for c in 0..k * in_ {
                        s += row[c] * win[c];
                    }
                    if s > pooled[kk * f + fi] {
                        pooled[kk * f + fi] = s;
                        arg[kk * f + fi] = pos as i32;
                    }
                }
            }
        }
        let (z1, probs) = head_forward(&self.ps, 4, 3 * f, self.cfg.hid, self.c, &pooled);
        CnnFwd { xs, pooled, arg, z1, probs, t }
    }

    pub fn infer(&self, toks: &[Tok]) -> Out {
        let f = self.forward(toks, true);
        Out { emb: f.pooled, hid: f.z1, probs: f.probs }
    }

    #[cfg(test)]
    pub fn loss(&self, toks: &[Tok], target: &[f32]) -> f32 {
        ce_loss(&self.forward(toks, false).probs, target)
    }

    /// accumulates parameter gradients into `gs`; returns (loss, dL/d token vectors for the real tokens)
    pub fn grad(&mut self, toks: &[Tok], target: &[f32]) -> (f32, Vec<f32>) {
        self.gs.iter_mut().for_each(|g| g.fill(0.0));
        let (in_, f) = (2 * self.cfg.dim, self.cfg.ch);
        let c = self.forward(toks, false);
        let loss = ce_loss(&c.probs, target);
        let dl: Vec<f32> = c.probs.iter().zip(target).map(|(p, t)| p - t).collect();
        let dp = head_backward(&self.ps, &mut self.gs, 4, 3 * f, self.cfg.hid, self.c, &c.pooled, &c.z1, &dl);
        let tp = c.t.max(3);
        let mut dxs = vec![0.0; tp * in_];
        for (kk, k) in KS.iter().enumerate() {
            for fi in 0..f {
                let a = c.arg[kk * f + fi];
                let g = dp[kk * f + fi];
                if a < 0 || g == 0.0 {
                    continue;
                }
                let a = a as usize;
                self.gs[3][kk * f + fi] += g;
                let win = &c.xs[a * in_..(a + k) * in_];
                let gw = &mut self.gs[kk][fi * k * in_..(fi + 1) * k * in_];
                let row = &self.ps[kk].w[fi * k * in_..(fi + 1) * k * in_];
                for j in 0..k * in_ {
                    gw[j] += g * win[j];
                    dxs[a * in_ + j] += g * row[j];
                }
            }
        }
        dxs.truncate(c.t * in_);
        (loss, dxs)
    }

    pub fn step(&mut self, toks: &[Tok], target: &[f32], lr: f32) {
        if toks.is_empty() {
            return;
        }
        for t in toks.iter().take(MAXT) {
            self.emb.mark(&t.w);
            self.emb.mark(&t.c);
        }
        let (_, dx) = self.grad(toks, target);
        let norm = self.gs.iter().flat_map(|g| g.iter()).map(|x| x * x).sum::<f32>().sqrt();
        let clip = if norm > 5.0 { 5.0 / norm } else { 1.0 };
        adam_all(&mut self.ps, &self.gs, lr / 0.125, &mut self.steps);
        let dx: Vec<f32> = dx.iter().map(|v| v * clip).collect();
        scatter_tokens(&mut self.emb, toks, &dx, lr * 2.0);
    }

}

// ============================ bidirectional GRU ============================
pub struct GruNet {
    pub cfg: Cfg,
    pub emb: EmbTable,
    pub ps: Vec<P>, // per direction: Wx, Wh, bx, bh; then head x4
    gs: Vec<Vec<f32>>,
    c: usize,
    steps: u32,
}

struct DirCache {
    hs: Vec<f32>, // (T+1) x H, hs[0] = 0
    r: Vec<f32>,
    z: Vec<f32>,
    n: Vec<f32>,
    ghn: Vec<f32>,
}

struct GruFwd {
    xs: [Vec<f32>; 2], // forward and reversed token order
    dc: [DirCache; 2],
    t: usize,
    pooled: Vec<f32>,
    z1: Vec<f32>,
    probs: Vec<f32>,
}

fn gru_dir_forward(wx: &[f32], wh: &[f32], bx: &[f32], bh: &[f32], xs: &[f32], t: usize, in_: usize, h: usize) -> DirCache {
    let mut c = DirCache { hs: vec![0.0; (t + 1) * h], r: vec![0.0; t * h], z: vec![0.0; t * h], n: vec![0.0; t * h], ghn: vec![0.0; t * h] };
    let mut gi = vec![0.0; 3 * h];
    let mut gh = vec![0.0; 3 * h];
    for s in 0..t {
        gi.copy_from_slice(bx);
        gh.copy_from_slice(bh);
        matvec(wx, 3 * h, in_, &xs[s * in_..(s + 1) * in_], &mut gi);
        let (done, rest) = c.hs.split_at_mut((s + 1) * h);
        let hprev = &done[s * h..];
        matvec(wh, 3 * h, h, hprev, &mut gh);
        for j in 0..h {
            let r = sigmoid(gi[j] + gh[j]);
            let z = sigmoid(gi[h + j] + gh[h + j]);
            let n = (gi[2 * h + j] + r * gh[2 * h + j]).tanh();
            c.r[s * h + j] = r;
            c.z[s * h + j] = z;
            c.n[s * h + j] = n;
            c.ghn[s * h + j] = gh[2 * h + j];
            rest[j] = (1.0 - z) * n + z * hprev[j];
        }
    }
    c
}

#[allow(clippy::too_many_arguments)]
fn gru_dir_backward(wx: &[f32], wh: &[f32], c: &DirCache, xs: &[f32], t: usize, in_: usize, h: usize, dhout: &[f32], gwx: &mut [f32], gwh: &mut [f32], gbx: &mut [f32], gbh: &mut [f32], dxs: &mut [f32]) {
    let mut dh_next = vec![0.0; h];
    let mut dgi = vec![0.0; 3 * h];
    let mut dgh = vec![0.0; 3 * h];
    for s in (0..t).rev() {
        let hprev = &c.hs[s * h..(s + 1) * h];
        let mut dhprev = vec![0.0; h];
        for j in 0..h {
            let (r, z, n, ghn) = (c.r[s * h + j], c.z[s * h + j], c.n[s * h + j], c.ghn[s * h + j]);
            let dh = dhout[s * h + j] + dh_next[j];
            let dn = dh * (1.0 - z);
            let dz = dh * (hprev[j] - n);
            dhprev[j] = dh * z;
            let dn_pre = dn * (1.0 - n * n);
            let dr = dn_pre * ghn;
            let dz_pre = dz * z * (1.0 - z);
            let dr_pre = dr * r * (1.0 - r);
            dgi[j] = dr_pre;
            dgi[h + j] = dz_pre;
            dgi[2 * h + j] = dn_pre;
            dgh[j] = dr_pre;
            dgh[h + j] = dz_pre;
            dgh[2 * h + j] = dn_pre * r;
        }
        let x = &xs[s * in_..(s + 1) * in_];
        outer_add(gwx, in_, &dgi, x);
        outer_add(gwh, h, &dgh, hprev);
        for j in 0..3 * h {
            gbx[j] += dgi[j];
            gbh[j] += dgh[j];
        }
        matvec_t(wx, 3 * h, in_, &dgi, &mut dxs[s * in_..(s + 1) * in_]);
        matvec_t(wh, 3 * h, h, &dgh, &mut dhprev);
        dh_next = dhprev;
    }
}

impl GruNet {
    pub fn new(c: usize, cfg: Cfg, rng: &mut Rng) -> GruNet {
        let (in_, h) = (2 * cfg.dim, cfg.ch);
        let mut ps = Vec::new();
        for _ in 0..2 {
            ps.push(P::new(glorot(3 * h * in_, in_, 3 * h, rng)));
            ps.push(P::new(glorot(3 * h * h, h, 3 * h, rng)));
            ps.push(P::zeros(3 * h));
            ps.push(P::zeros(3 * h));
        }
        ps.push(P::new(glorot(cfg.hid * 2 * h, 2 * h, cfg.hid, rng)));
        ps.push(P::zeros(cfg.hid));
        ps.push(P::new(glorot(c * cfg.hid, cfg.hid, c, rng)));
        ps.push(P::zeros(c));
        let gs = ps.iter().map(|p| vec![0.0; p.w.len()]).collect();
        GruNet { cfg, emb: EmbTable::new(cfg.dim, cfg.bits, rng), ps, gs, c, steps: 0 }
    }

    fn forward(&self, toks: &[Tok], filter: bool) -> GruFwd {
        let (in_, h) = (2 * self.cfg.dim, self.cfg.ch);
        let (xs, t) = embed_tokens(&self.emb, toks, filter, 0);
        let mut xr = vec![0.0; t * in_];
        for s in 0..t {
            xr[s * in_..(s + 1) * in_].copy_from_slice(&xs[(t - 1 - s) * in_..(t - s) * in_]);
        }
        let mk = |d: usize, x: &[f32]| gru_dir_forward(&self.ps[4 * d].w, &self.ps[4 * d + 1].w, &self.ps[4 * d + 2].w, &self.ps[4 * d + 3].w, x, t, in_, h);
        let dc = [mk(0, &xs), mk(1, &xr)];
        let mut pooled = vec![0.0; 2 * h];
        let inv = 1.0 / t.max(1) as f32;
        for (d, cache) in dc.iter().enumerate() {
            for s in 0..t {
                for j in 0..h {
                    pooled[d * h + j] += cache.hs[(s + 1) * h + j] * inv;
                }
            }
        }
        let (z1, probs) = head_forward(&self.ps, 8, 2 * h, self.cfg.hid, self.c, &pooled);
        GruFwd { xs: [xs, xr], dc, t, pooled, z1, probs }
    }

    pub fn infer(&self, toks: &[Tok]) -> Out {
        let f = self.forward(toks, true);
        Out { emb: f.pooled, hid: f.z1, probs: f.probs }
    }

    #[cfg(test)]
    pub fn loss(&self, toks: &[Tok], target: &[f32]) -> f32 {
        ce_loss(&self.forward(toks, false).probs, target)
    }

    pub fn grad(&mut self, toks: &[Tok], target: &[f32]) -> (f32, Vec<f32>) {
        self.gs.iter_mut().for_each(|g| g.fill(0.0));
        let (in_, h) = (2 * self.cfg.dim, self.cfg.ch);
        let f = self.forward(toks, false);
        let loss = ce_loss(&f.probs, target);
        let dl: Vec<f32> = f.probs.iter().zip(target).map(|(p, t)| p - t).collect();
        let dp = head_backward(&self.ps, &mut self.gs, 8, 2 * h, self.cfg.hid, self.c, &f.pooled, &f.z1, &dl);
        let t = f.t;
        let inv = 1.0 / t.max(1) as f32;
        let mut dxs = [vec![0.0; t * in_], vec![0.0; t * in_]];
        for d in 0..2 {
            let mut dhout = vec![0.0; t * h];
            for s in 0..t {
                for j in 0..h {
                    dhout[s * h + j] = dp[d * h + j] * inv;
                }
            }
            let (head, tail) = self.gs.split_at_mut(4 * d + 1);
            let gwx = &mut head[4 * d];
            let (gwh, rest) = tail.split_at_mut(1);
            let (gbx, rest) = rest.split_at_mut(1);
            let gbh = &mut rest[0];
            gru_dir_backward(&self.ps[4 * d].w, &self.ps[4 * d + 1].w, &f.dc[d], &f.xs[d], t, in_, h, &dhout, gwx, &mut gwh[0], &mut gbx[0], gbh, &mut dxs[d]);
        }
        let mut dx = dxs[0].clone();
        for s in 0..t {
            for j in 0..in_ {
                dx[s * in_ + j] += dxs[1][(t - 1 - s) * in_ + j];
            }
        }
        (loss, dx)
    }

    pub fn step(&mut self, toks: &[Tok], target: &[f32], lr: f32) {
        if toks.is_empty() {
            return;
        }
        for t in toks.iter().take(MAXT) {
            self.emb.mark(&t.w);
            self.emb.mark(&t.c);
        }
        let (_, dx) = self.grad(toks, target);
        let norm = self.gs.iter().flat_map(|g| g.iter()).map(|x| x * x).sum::<f32>().sqrt();
        let clip = if norm > 5.0 { 5.0 / norm } else { 1.0 };
        adam_all(&mut self.ps, &self.gs, lr / 0.125, &mut self.steps);
        let dx: Vec<f32> = dx.iter().map(|v| v * clip).collect();
        scatter_tokens(&mut self.emb, toks, &dx, lr * 2.0);
    }

}

// ============================ gradient checks ============================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Arch, Cfg};

    fn toks() -> Vec<Tok> {
        // distinct feature ids per token so each embedding row belongs to exactly one place
        vec![
            Tok { w: vec![1, 2], c: vec![(1 << 20) + 10, (1 << 20) + 11] },
            Tok { w: vec![3], c: vec![(1 << 20) + 12] },
            Tok { w: vec![4], c: vec![(1 << 20) + 13, (1 << 20) + 14] },
            Tok { w: vec![5], c: vec![] },
        ]
    }

    const TARGET: [f32; 4] = [0.1, 0.7, 0.1, 0.1];

    fn cfg(arch: Arch) -> Cfg {
        Cfg { arch, dim: 3, hid: 5, bits: 6, ch: 4, epochs: 1 }
    }

    /// numeric vs analytic gradient for a sample of entries of every dense parameter array
    fn check<N>(net: &mut N, ps_len: usize, get: impl Fn(&mut N, usize) -> &mut Vec<f32>, loss: impl Fn(&N) -> f32, grads: impl Fn(&N, usize) -> Vec<f32>, name: &str) {
        let eps = 1e-2f32;
        let (mut ok, mut total) = (0, 0);
        for pi in 0..ps_len {
            let g = grads(net, pi);
            let n = get(net, pi).len();
            for k in 0..n.min(10) {
                let idx = (k * 7919 + pi * 31) % n;
                let orig = get(net, pi)[idx];
                get(net, pi)[idx] = orig + eps;
                let lp = loss(net);
                get(net, pi)[idx] = orig - eps;
                let lm = loss(net);
                get(net, pi)[idx] = orig;
                let num = (lp - lm) / (2.0 * eps);
                total += 1;
                if (num - g[idx]).abs() <= 3e-3 + 0.05 * num.abs().max(g[idx].abs()) {
                    ok += 1;
                } else {
                    eprintln!("{name}: param {pi}[{idx}] numeric {num:.5} analytic {:.5}", g[idx]);
                }
            }
        }
        assert!(ok as f32 / total as f32 >= 0.95, "{name}: only {ok}/{total} gradient entries matched finite differences");
    }

    #[test]
    fn cnn_gradients_match_finite_differences() {
        let mut rng = Rng::new(3);
        let mut net = CnnNet::new(4, cfg(Arch::Cnn), &mut rng);
        let tk = toks();
        net.grad(&tk, &TARGET);
        let n = net.ps.len();
        check(&mut net, n, |m, i| &mut m.ps[i].w, |m| m.loss(&tk, &TARGET), |m, i| m.gs[i].clone(), "cnn");
    }

    #[test]
    fn gru_gradients_match_finite_differences() {
        let mut rng = Rng::new(5);
        let mut net = GruNet::new(4, cfg(Arch::Gru), &mut rng);
        let tk = toks();
        net.grad(&tk, &TARGET);
        let n = net.ps.len();
        check(&mut net, n, |m, i| &mut m.ps[i].w, |m| m.loss(&tk, &TARGET), |m, i| m.gs[i].clone(), "gru");
    }

    /// dL/d(row of feature f in token t's word channel) = dx_t[0..dim] / (number of word features in t)
    fn emb_check<N>(net: &mut N, dx: &[f32], bump: impl Fn(&mut N, usize, f32), loss: impl Fn(&N) -> f32, name: &str) {
        let tk = toks();
        let (dim, in_, eps) = (3usize, 6usize, 1e-2f32);
        let (mut ok, mut total) = (0, 0);
        // features 3, 4, 5 are the sole word feature of tokens 1, 2, 3
        for (tok_i, feat) in [(1usize, 3usize), (2, 4), (3, 5)] {
            let n_w = tk[tok_i].w.len() as f32;
            for j in 0..dim {
                let analytic = dx[tok_i * in_ + j] / n_w;
                bump(net, feat * dim + j, eps);
                let lp = loss(net);
                bump(net, feat * dim + j, -2.0 * eps);
                let lm = loss(net);
                bump(net, feat * dim + j, eps);
                let num = (lp - lm) / (2.0 * eps);
                total += 1;
                if (num - analytic).abs() <= 3e-3 + 0.05 * num.abs().max(analytic.abs()) {
                    ok += 1;
                }
            }
        }
        assert!(ok as f32 / total as f32 >= 0.9, "{name}: embedding gradient mismatch {ok}/{total}");
    }

    #[test]
    fn cnn_embedding_gradients() {
        let mut rng = Rng::new(9);
        let tk = toks();
        let mut net = CnnNet::new(4, cfg(Arch::Cnn), &mut rng);
        let (_, dx) = net.grad(&tk, &TARGET);
        emb_check(&mut net, &dx, |n, i, d| n.emb.w[i] += d, |n| n.loss(&tk, &TARGET), "cnn");
    }

    #[test]
    fn gru_embedding_gradients() {
        let mut rng = Rng::new(9);
        let tk = toks();
        let mut net = GruNet::new(4, cfg(Arch::Gru), &mut rng);
        let (_, dx) = net.grad(&tk, &TARGET);
        emb_check(&mut net, &dx, |n, i, d| n.emb.w[i] += d, |n| n.loss(&tk, &TARGET), "gru");
    }

    #[test]
    fn nets_can_overfit_a_tiny_problem() {
        // sanity: a few hundred Adam steps should drive the loss near zero on two examples
        let a = vec![Tok { w: vec![1], c: vec![(1 << 20) + 2] }, Tok { w: vec![3], c: vec![(1 << 20) + 4] }];
        let b = vec![Tok { w: vec![5], c: vec![(1 << 20) + 6] }, Tok { w: vec![7], c: vec![(1 << 20) + 8] }];
        let (ta, tb) = ([0.97, 0.01, 0.01, 0.01], [0.01, 0.97, 0.01, 0.01]);
        for arch in [Arch::Cnn, Arch::Gru] {
            let mut rng = Rng::new(1);
            let (mut cnn, mut gru) = (CnnNet::new(4, cfg(arch), &mut rng), GruNet::new(4, cfg(arch), &mut rng));
            for _ in 0..900 {
                if arch == Arch::Cnn {
                    cnn.step(&a, &ta, 0.125);
                    cnn.step(&b, &tb, 0.125);
                } else {
                    gru.step(&a, &ta, 0.125);
                    gru.step(&b, &tb, 0.125);
                }
            }
            let la = if arch == Arch::Cnn { cnn.loss(&a, &ta) + cnn.loss(&b, &tb) } else { gru.loss(&a, &ta) + gru.loss(&b, &tb) };
            assert!(la < 0.6, "{arch:?} failed to fit two examples, loss {la} (the floor with label smoothing is about 0.35)");
        }
    }
}
