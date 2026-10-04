//! Pure-ANSI terminal UI (no crates, no raw mode). Every frame is composed into one String and
//! written in a single go with cursor-home, so animation doesn't flicker.
//!
//! Left  : animated face (18 emotions + particles), live brain, status
//! Right : chat (your words shaded by how much each one drove the decision), then a CONSOLE pane
//!         underneath that holds command output, lists and ASCII art so the chat stays clean
//!
//! Input is read on its own thread, so the face keeps blinking, dozing off and ringing timers
//! while you think about what to type.

use crate::bot::Analysis;
use crate::cmds::{self, App};
use crate::rng::Rng;
use crate::skills::fmt_dur;
use crate::train;
use std::io::{self, BufRead, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const LW: usize = 34; // left panel inner width

type Seg = (String, Option<u8>); // text, background colour (256-colour index)

enum Who {
    You,
    Bot,
    Sys,
    Toast,
}

struct Line {
    who: Who,
    segs: Vec<Seg>,
}

struct Ui {
    w: usize,
    h: usize,
    log: Vec<Line>,
    console: Vec<String>,
    con_title: String,
    emo: &'static str,
    emo_left: u32,
    talking: bool,
    frame: u32,
    analysis: Option<Analysis>,
    shown: Option<Vec<f32>>,
    scan: Option<usize>,
    flash: u32,
    anim: bool,
    kb: f32,
    rng: Rng,
    last_input: Instant,
}

fn term_size() -> (usize, usize) {
    let from_stty = || {
        let tty = std::fs::File::open("/dev/tty").ok()?;
        let o = Command::new("stty").arg("size").stdin(Stdio::from(tty)).output().ok()?;
        let s = String::from_utf8_lossy(&o.stdout).to_string();
        let mut it = s.split_whitespace().map(|x| x.parse::<usize>().ok());
        Some((it.next()??, it.next()??))
    };
    let (rows, cols) = from_stty().unwrap_or_else(|| {
        let e = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        (e("LINES", 36), e("COLUMNS", 112))
    });
    (cols.clamp(84, 150), rows.clamp(30, 60))
}

// ---------------- colours ----------------
fn mood_colour(label: &str) -> u8 {
    match label {
        "ecstatic" => 213,
        "happy" => 120,
        "calm" => 81,
        "grumpy" => 208,
        "sad" => 105,
        _ => 196,
    }
}

fn fg(c: u8, s: &str) -> String {
    format!("\x1b[38;5;{c}m{s}\x1b[0m")
}

/// dark -> ember -> white-hot, for the neuron heatmap
fn heat(v: f32) -> u8 {
    const RAMP: [u8; 10] = [233, 53, 89, 125, 161, 197, 202, 208, 214, 229];
    RAMP[((v.clamp(0.0, 1.0)) * 9.0).round() as usize]
}

fn sal_bg(d: f32) -> Option<u8> {
    if d < 0.04 {
        return None;
    }
    const RAMP: [u8; 6] = [237, 52, 88, 124, 160, 166];
    Some(RAMP[((d * 8.0) as usize).min(5)])
}

fn hash32(a: u32, b: u32) -> u32 {
    let mut x = a.wrapping_mul(0x9E37_79B1) ^ b.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^ (x >> 12)
}

// ---------------- text helpers ----------------
struct Row {
    s: String,
    w: usize,
}

impl Row {
    fn new() -> Row {
        Row { s: String::new(), w: 0 }
    }
    fn plain(mut self, t: &str) -> Row {
        self.w += t.chars().count();
        self.s.push_str(t);
        self
    }
    fn styled(mut self, c: u8, t: &str) -> Row {
        self.w += t.chars().count();
        self.s.push_str(&fg(c, t));
        self
    }
    fn raw(mut self, ansi: String, visible: usize) -> Row {
        self.w += visible;
        self.s.push_str(&ansi);
        self
    }
    fn pad(mut self, to: usize) -> String {
        if self.w < to {
            self.s.push_str(&" ".repeat(to - self.w));
        }
        self.s
    }
}

fn wrap(segs: &[Seg], width: usize) -> Vec<Vec<Seg>> {
    let mut lines: Vec<Vec<Seg>> = vec![vec![]];
    let mut col = 0;
    for (text, bg) in segs {
        for (i, word) in text.split(' ').enumerate() {
            if word.is_empty() && i > 0 {
                continue;
            }
            let mut word = word.to_string();
            while word.chars().count() > width {
                let head: String = word.chars().take(width).collect();
                word = word.chars().skip(width).collect();
                if col > 0 {
                    lines.push(vec![]);
                }
                lines.last_mut().unwrap().push((head, *bg));
                lines.push(vec![]);
                col = 0;
            }
            let wl = word.chars().count();
            let need = if col == 0 { wl } else { wl + 1 };
            if col + need > width && col > 0 {
                lines.push(vec![]);
                col = 0;
            }
            let cur = lines.last_mut().unwrap();
            if col > 0 {
                cur.push((" ".into(), None));
                col += 1;
            }
            cur.push((word, *bg));
            col += wl;
        }
    }
    lines
}

// ---------------- face ----------------
struct Canvas {
    w: usize,
    ch: Vec<Vec<char>>,
    co: Vec<Vec<u8>>, // 0 = default colour
}

impl Canvas {
    fn new(w: usize, h: usize) -> Canvas {
        Canvas { w, ch: vec![vec![' '; w]; h], co: vec![vec![0; w]; h] }
    }
    fn put(&mut self, x: i32, y: i32, c: char, col: u8, only_blank: bool) {
        if x < 0 || y < 0 || y as usize >= self.ch.len() || x as usize >= self.w {
            return;
        }
        let (x, y) = (x as usize, y as usize);
        if only_blank && self.ch[y][x] != ' ' {
            return;
        }
        self.ch[y][x] = c;
        self.co[y][x] = col;
    }
    fn text(&mut self, x: i32, y: i32, t: &str, col: u8) {
        for (i, c) in t.chars().enumerate() {
            self.put(x + i as i32, y, c, col, false);
        }
    }
    fn row(&self, y: usize) -> Row {
        let mut r = Row::new();
        let (mut run, mut cur) = (String::new(), 0u8);
        let flush = |r: Row, run: &mut String, cur: u8| -> Row {
            let out = if run.is_empty() { r } else if cur == 0 { r.plain(run) } else { r.styled(cur, run) };
            run.clear();
            out
        };
        for x in 0..self.w {
            if self.co[y][x] != cur {
                r = flush(r, &mut run, cur);
                cur = self.co[y][x];
            }
            run.push(self.ch[y][x]);
        }
        flush(r, &mut run, cur)
    }
}

struct Spec {
    el: char,
    er: char,
    bl: char,
    br: char,
    mouth: String,
    col: Option<u8>,
    cheeks: bool,
}

pub const EMOTIONS: [&str; 18] = [
    "neutral", "joy", "ecstatic", "love", "laugh", "anger", "grumpy", "sad", "concern", "wow", "confused", "curious", "think", "sleepy", "asleep",
    "proud", "wink", "sulk",
];

fn spec(emo: &str, t: u32, talking: bool) -> Spec {
    let tf = t as usize;
    let s = |el, er, bl, br, mouth: &str, col, cheeks| Spec { el, er, bl, br, mouth: mouth.to_string(), col, cheeks };
    let spin = ['◔', '◑', '◕', '◒'][tf / 2 % 4];
    let mut sp = match emo {
        "joy" => s('◠', '◠', ' ', ' ', " ╰───╯ ", Some(120), true),
        "ecstatic" => s('★', '★', ' ', ' ', " \\___/ ", Some(213), true),
        "love" => s('♥', '♥', ' ', ' ', " ╰───╯ ", Some(204), true),
        "laugh" => s('^', '^', ' ', ' ', if tf / 2 % 2 == 0 { " \\___/ " } else { " \\_o_/ " }, Some(120), true),
        "anger" => s('●', '●', '\\', '/', "  ▔▔▔  ", Some(196), false),
        "grumpy" => s('▬', '▬', '\\', '/', "  ▔▔▔  ", Some(208), false),
        "sad" => s('╥', '╥', '/', '\\', " ╭───╮ ", Some(105), false),
        "concern" => s('●', '●', '/', '\\', "  ───  ", Some(111), false),
        "wow" => s('O', 'O', '‾', '‾', "   O   ", Some(226), false),
        "confused" => s('o', 'O', ' ', '/', "  ─?─  ", Some(220), false),
        "curious" => s('●', '◉', ' ', '‾', "  ─⌒─  ", Some(117), false),
        "think" => s(spin, spin, ' ', ' ', ["  ·    ", "  · ·  ", "  · · ·", "       "][tf / 3 % 4], Some(45), false),
        "sleepy" => s('─', '─', ' ', ' ', if tf / 6 % 2 == 0 { "  ─o─  " } else { "  ───  " }, Some(244), false),
        "asleep" => s('─', '─', ' ', ' ', "  ─ ─  ", Some(240), false),
        "proud" => s('◠', '◠', ' ', ' ', " ╰───╯ ", Some(220), true),
        "wink" => s('●', '─', ' ', ' ', " ╰───╯ ", Some(159), false),
        "sulk" => s('─', '─', '\\', '/', "  ~~~  ", Some(196), false),
        _ => s('●', '●', ' ', ' ', "  ───  ", None, false),
    };
    if talking {
        sp.mouth = [" ╰───╯ ", " ╰ o ╯ ", "  ─o─  ", " ╰ ◦ ╯ "][tf / 2 % 4].to_string();
    }
    sp
}

fn fx(cv: &mut Canvas, emo: &str, t: u32) {
    let ti = t as i32;
    match emo {
        "love" => {
            for i in 0..3i32 {
                let ph = (ti / 3 + i * 3) % 7;
                let sway = [0, 1, 0, -1][((ti / 2 + i) % 4) as usize];
                cv.put([7, 26, 30][i as usize] + sway, 6 - ph, '♥', [204, 197, 211][i as usize], true);
            }
        }
        "sad" | "concern" => {
            for (i, x) in [9i32, 23].iter().enumerate() {
                cv.put(*x, 4 + ((ti / 2 + i as i32 * 2) % 3), '•', 75, true);
            }
        }
        "sleepy" | "asleep" => {
            for i in 0..3i32 {
                let p = (ti / 4 + i * 2) % 6;
                cv.put(23 + p, 5 - p, if p < 3 { 'z' } else { 'Z' }, 244, true);
            }
        }
        "anger" => {
            let g = if (ti / 3) % 2 == 0 { '#' } else { '*' };
            for (x, y) in [(8, 2), (24, 3), (25, 1), (7, 4)] {
                cv.put(x, y, g, 196, true);
            }
        }
        "laugh" => {
            cv.text(24, 3 + (ti / 3) % 2, "ha!", 120);
            cv.text(4, 4 - (ti / 4) % 2, "HA", 120);
        }
        "wow" => {
            if (ti / 2) % 2 == 0 {
                cv.text(8, 1, "!", 226);
                cv.text(25, 2, "!!", 226);
            } else {
                cv.text(7, 2, "!", 226);
                cv.text(26, 1, "!", 226);
            }
        }
        "confused" => {
            cv.put(24, 2 + (ti / 4) % 2, '?', 220, true);
            cv.put(8, 3 - (ti / 5) % 2, '?', 220, true);
        }
        "curious" => cv.put(25, 2 + (ti / 5) % 2, '?', 117, true),
        "think" => {
            let n = (ti / 3) % 4;
            if n >= 1 {
                cv.put(24, 4, '·', 45, true);
            }
            if n >= 2 {
                cv.put(26, 3, 'o', 45, true);
            }
            if n >= 3 {
                cv.put(28, 1, 'O', 45, true);
            }
        }
        "ecstatic" | "proud" | "joy" => {
            let n = if emo == "joy" { 3 } else { 7 };
            for i in 0..n {
                let h = hash32(i, t / 3);
                let (x, y) = (2 + (h % 30) as i32, ((h >> 8) % 7) as i32);
                cv.put(x, y, ['✦', '✧', '·', '*'][((h >> 16) % 4) as usize], [220, 213, 159, 229][((h >> 20) % 4) as usize], true);
            }
        }
        "sulk" => {
            cv.text(22, 0, ".-~~~-.", 244);
            cv.text(21, 1, "(_______)", 244);
            for i in 0..3i32 {
                cv.put(24 + i * 2, 2 + (ti + i) % 3, '|', 75, true);
            }
        }
        _ => {}
    }
}

fn draw_face(emo: &str, t: u32, base: u8, talking: bool, look: i32, label: &str) -> Vec<Row> {
    let mut cv = Canvas::new(LW, 8);
    let sp = spec(emo, t, talking);
    let col = sp.col.unwrap_or(base);
    let x0 = 10;
    if emo != "sulk" {
        let a = if t % 17 == 0 || (emo == "think" && t % 4 < 2) { '✦' } else { '·' };
        cv.put(16, 0, a, col, false);
        cv.put(16, 1, '│', col, false);
    }
    cv.text(x0, 2, "╭───────────╮", col);
    for y in 3..=5 {
        cv.text(x0, y, "│           │", col);
    }
    cv.text(x0, 6, "╰───────────╯", col);
    let blink = t % 31 == 0 && !matches!(emo, "sleepy" | "asleep" | "wink");
    let shift = if matches!(emo, "neutral" | "joy") { look } else { 0 };
    let eye = |c: char| if blink && c != ' ' { '─' } else { c };
    cv.put(13 + shift, 4, eye(sp.el), col, false);
    cv.put(19 + shift, 4, eye(sp.er), col, false);
    cv.put(13, 3, sp.bl, col, false);
    cv.put(19, 3, sp.br, col, false);
    cv.text(13, 5, &sp.mouth, col);
    if sp.cheeks {
        cv.put(11, 5, '░', 217, false);
        cv.put(21, 5, '░', 217, false);
    }
    fx(&mut cv, emo, t);
    let lw = label.chars().count().min(LW);
    cv.text(((LW - lw) / 2) as i32, 7, label, 245);
    (0..8).map(|y| cv.row(y)).collect()
}

// ---------------- frame ----------------
impl Ui {
    fn emotion(&self, app: &App) -> &'static str {
        if self.emo_left > 0 {
            return self.emo;
        }
        let idle = self.last_input.elapsed().as_secs();
        let p = &app.s.persona;
        if idle > 75 || (p.tired() && idle > 20) {
            return "asleep";
        }
        if idle > 30 || p.tired() {
            return "sleepy";
        }
        match p.mood_label() {
            "ecstatic" => "ecstatic",
            "happy" => "joy",
            "grumpy" => "grumpy",
            "sad" => "sad",
            "sulking" => "sulk",
            _ => "neutral",
        }
    }

    fn set_emo(&mut self, e: &'static str, frames: u32) {
        self.emo = e;
        self.emo_left = frames;
    }

    fn console_rows(&self, rw: usize, max: usize) -> Vec<String> {
        if self.console.is_empty() || max < 2 {
            return vec![];
        }
        let title = format!("─ CONSOLE · {} ", self.con_title);
        let mut out = vec![Row::new().styled(240, &title).styled(240, &"─".repeat(rw.saturating_sub(title.chars().count()))).pad(rw)];
        let mut body: Vec<String> = Vec::new();
        for l in &self.console {
            let cs: Vec<char> = l.chars().collect();
            if cs.is_empty() {
                body.push(String::new());
            }
            for chunk in cs.chunks(rw) {
                body.push(chunk.iter().collect());
            }
        }
        let room = max - 1;
        let hidden = body.len() > room;
        let shown = if hidden { room - 1 } else { body.len() };
        for l in body.iter().take(shown) {
            out.push(Row::new().styled(250, l).pad(rw));
        }
        if hidden {
            out.push(Row::new().styled(240, &format!("… {} more line(s), enlarge the window or /clear", body.len() - shown)).pad(rw));
        }
        out
    }

    fn rng_val(&self, i: usize) -> f32 {
        (hash32(i as u32, self.frame) % 1000) as f32 / 1000.0
    }

    fn frame_string(&mut self, app: &App) -> String {
        let (w, h) = (self.w, self.h);
        let p = &app.s.persona;
        let label = p.mood_label();
        let rw = w - LW - 7;
        let body_h = h - 4;

        // ---- header ----
        let (a, b) = p.progress();
        let bar_n = ((a as f32 / b as f32) * 10.0).round() as usize;
        let hearts = "♥".repeat(p.hearts()) + &"♡".repeat(5 - p.hearts());
        let mut head = format!(" TinyBot ─ Lv {} {} ─ {}{} {a}/{b} XP ─ mood {label} ─ {hearts} ", p.level(), p.title(), "▓".repeat(bar_n), "░".repeat(10 - bar_n));
        if app.retraining() {
            head.push_str(&format!("─ {} learning ", ["◜", "◝", "◞", "◟"][(self.frame / 2 % 4) as usize]));
        }
        if let Some((d, l)) = app.s.next_timer() {
            head.push_str(&format!("─ ⏱ {} {} ", fmt_dur(d.as_secs()), l.chars().take(12).collect::<String>()));
        }
        let head_w = head.chars().count();
        let hc = if self.flash > 0 && self.frame % 2 == 0 { 229 } else { 208 };
        let mut out = String::new();
        out.push_str(&format!("\x1b[H{}\x1b[K\n", fg(240, "╭") + &fg(hc, &head) + &fg(240, &format!("{}╮", "─".repeat(w.saturating_sub(head_w + 2))))));

        // ---- left panel ----
        let emo = self.emotion(app);
        let look = match (self.frame / 45) % 4 {
            1 => -1,
            3 => 1,
            _ => 0,
        };
        let mut left: Vec<String> = Vec::new();
        for r in draw_face(emo, self.frame, mood_colour(label), self.talking, look, &format!("{label} · {}", p.title())) {
            left.push(r.pad(LW));
        }
        left.push(" ".repeat(LW));
        left.push(Row::new().styled(240, "─ BRAIN ").styled(240, &"─".repeat(LW - 8)).pad(LW));
        if let Some(an) = &self.analysis {
            let probs = self.shown.as_ref().unwrap_or(&an.probs);
            let mut idx: Vec<usize> = (0..probs.len()).collect();
            idx.sort_by(|&x, &y| probs[y].total_cmp(&probs[x]));
            for &i in idx.iter().take(5) {
                let name: String = app.brain.model.tags[i].chars().take(14).collect();
                let n = (probs[i] * 10.0).round() as usize;
                let col = if i == idx[0] { 208 } else { 244 };
                left.push(Row::new().styled(col, &format!("{name:<14} {}{} {:>3.0}%", "█".repeat(n), "░".repeat(10 - n), probs[i] * 100.0)).pad(LW));
            }
            left.push(Row::new().styled(245, &format!("coverage {:.0}%  ·  {:.0} µs", an.cov * 100.0, an.micros)).pad(LW));
            left.push(Row::new().styled(240, "─ NEURONS ").styled(240, &"─".repeat(LW - 10)).pad(LW));
            let mx = an.acts.iter().cloned().fold(1e-6f32, f32::max);
            for row in 0..an.acts.len().div_ceil(8) {
                let mut r = Row::new().plain("  ");
                for c in 0..8 {
                    let mut v = an.acts.get(row * 8 + c).copied().unwrap_or(0.0) / mx;
                    if let Some(sc) = self.scan {
                        let dist = (sc as i32 - c as i32).abs();
                        v = if dist == 0 {
                            v.max(0.95)
                        } else if dist == 1 {
                            (v + 0.5).min(1.0)
                        } else {
                            (v * 0.5 + self.rng_val(row * 8 + c) * 0.5).min(1.0)
                        };
                    }
                    r = r.raw(format!("\x1b[48;5;{}m  \x1b[0m", heat(v)), 2).plain(" ");
                }
                left.push(r.pad(LW));
            }
        } else {
            left.push(Row::new().styled(245, "say something and I'll show").pad(LW));
            left.push(Row::new().styled(245, "you what's going on in here.").pad(LW));
        }
        left.push(Row::new().styled(240, "─ STATUS ").styled(240, &"─".repeat(LW - 9)).pad(LW));
        let en = (p.energy * 5.0).ceil() as usize;
        left.push(Row::new().styled(204, &"♥".repeat(p.hearts())).styled(240, &"♡".repeat(5 - p.hearts())).styled(250, &format!(" {} ({} pts)", p.bond_label(), p.bond)).pad(LW));
        left.push(Row::new().styled(220, &"▮".repeat(en)).styled(240, &"▯".repeat(5 - en)).styled(250, &format!(" energy {:.0}%", p.energy * 100.0)).pad(LW));
        left.push(Row::new().styled(250, &format!("facts {}/{} · dex {}/{}", p.factbook.len(), app.brain.facts.items.len(), p.dex.len(), app.brain.intents.len())).pad(LW));
        left.push(Row::new().styled(240, &format!("{} params · {:.0} KB", train::params_label(&app.brain.model), self.kb)).pad(LW));
        left.truncate(body_h);
        while left.len() < body_h {
            left.push(" ".repeat(LW));
        }

        // ---- right panel: chat on top, console underneath ----
        let con = self.console_rows(rw, body_h / 2 + 1);
        let chat_h = body_h - con.len();
        let mut chat: Vec<String> = Vec::new();
        for l in &self.log {
            let (label, col) = match l.who {
                Who::You => ("you ", 45u8),
                Who::Bot => ("bot ", 208),
                Who::Sys => ("    ", 245),
                Who::Toast => (" ★  ", 220),
            };
            for (i, line) in wrap(&l.segs, rw - 5).iter().enumerate() {
                let mut r = Row::new().styled(col, if i == 0 { label } else { "    " }).plain(" ");
                for (t, bg) in line {
                    let styled = match (&l.who, bg) {
                        (Who::You, Some(b)) => format!("\x1b[38;5;255;48;5;{b}m{t}\x1b[0m"),
                        (Who::You, None) => fg(255, t),
                        (Who::Bot, _) => fg(223, t),
                        (Who::Sys, _) => fg(245, t),
                        (Who::Toast, _) => fg(220, t),
                    };
                    r = r.raw(styled, t.chars().count());
                }
                chat.push(r.pad(rw));
            }
            if matches!(l.who, Who::Bot) {
                chat.push(" ".repeat(rw));
            }
        }
        let start = chat.len().saturating_sub(chat_h);
        let mut chat: Vec<String> = chat.split_off(start);
        while chat.len() < chat_h {
            chat.push(" ".repeat(rw));
        }
        chat.extend(con);

        let bar = fg(240, "│");
        for i in 0..body_h {
            out.push_str(&format!("{bar} {} {bar} {} {bar}\x1b[K\n", left[i], chat[i]));
        }
        out.push_str(&format!("{}\x1b[K", fg(240, &format!("╰{}╯", "─".repeat(w - 2)))));
        out
    }

    /// full redraw during an animation: park the cursor on the prompt row afterwards
    fn draw(&mut self, app: &App) {
        let f = self.frame_string(app);
        let mut o = io::stdout().lock();
        let _ = write!(o, "{f}\x1b[{};9H", self.h - 1);
        let _ = o.flush();
        self.frame += 1;
    }

    /// redraw while the user may be mid-sentence: leave their cursor and text alone
    fn draw_idle(&mut self, app: &App) {
        let f = self.frame_string(app);
        let mut o = io::stdout().lock();
        let _ = write!(o, "\x1b7{f}\x1b8");
        let _ = o.flush();
        self.frame += 1;
    }

    fn prompt(&self) {
        let mut o = io::stdout().lock();
        let _ = write!(o, "\x1b[{};1H\x1b[K\x1b[38;5;45m you ▸ \x1b[0m\x1b[?25h", self.h - 1);
        let _ = o.flush();
    }

    fn clear_input_rows(&self) {
        print!("\x1b[{};1H\x1b[K\x1b[{};1H\x1b[K\x1b[?25l", self.h - 1, self.h);
    }

    fn sleep(&self, ms: u64) {
        if self.anim {
            std::thread::sleep(Duration::from_millis(ms));
        }
    }

    fn console_set(&mut self, title: &str, lines: Vec<String>) {
        self.con_title = title.to_string();
        self.console = lines;
    }
}

fn say(ui: &mut Ui, who: Who, text: &str) {
    ui.log.push(Line { who, segs: vec![(text.to_string(), None)] });
}

/// Reveal a bot message gradually: the mouth moves, punctuation gets a beat.
fn typewrite(ui: &mut Ui, app: &App, text: &str) {
    ui.log.push(Line { who: Who::Bot, segs: vec![(String::new(), None)] });
    let chars: Vec<char> = text.chars().collect();
    let step = if ui.anim { 2 } else { chars.len().max(1) };
    let mut n = 0;
    ui.talking = true;
    while n < chars.len() {
        let end = (n + step).min(chars.len());
        let pause: u64 = chars[n..end]
            .iter()
            .map(|c| match c {
                '.' | '!' | '?' => 95,
                ',' | ';' | ':' => 45,
                _ => 11,
            })
            .sum();
        n = end;
        if let Some(l) = ui.log.last_mut() {
            l.segs = vec![(chars[..n].iter().collect(), None)];
        }
        ui.draw(app);
        ui.sleep(pause);
    }
    ui.talking = false;
}

fn show_reply(ui: &mut Ui, app: &App, tag: &str, text: &str) {
    let mut it = text.split('\n');
    let first = it.next().unwrap_or("");
    let rest: Vec<String> = it.map(String::from).collect();
    if rest.is_empty() {
        typewrite(ui, app, text);
    } else {
        typewrite(ui, app, &format!("{first} ↓"));
        ui.console_set(tag, rest);
    }
}

fn welcome(app: &App) -> (String, &'static str) {
    let p = &app.s.persona;
    let name = app.s.mem.name().map(String::from);
    let params = train::params_label(&app.brain.model);
    if crate::skills::birthday_today(&app.s.mem) {
        let who = name.map(|n| format!(", {}", n.to_uppercase())).unwrap_or_default();
        return (format!("HAPPY BIRTHDAY{who}!! I don't have cake, but I do have {params} parameters of pure enthusiasm."), "ecstatic");
    }
    // time-of-day greeting about 60% of the time (when the time zone is known), a plain one otherwise
    let timed = crate::skills::local_hour().is_some() && crate::rng::time_seed() % 100 < 60;
    let opener = if timed { crate::skills::greet_word() } else { "Hey" };
    let (mut hello, mut emo) = (format!("{opener}{}! I'm TinyBot: {params} parameters, written from scratch in Rust.", name.as_ref().map(|n| format!(", {n}")).unwrap_or_default()), "joy");
    if p.away_days >= 3 && p.bond >= 10 {
        hello = format!("{}, you're back after {} days! I missed you.", name.as_deref().unwrap_or("Hey"), p.away_days);
        emo = "love";
    }
    if let Some(t) = &p.returning {
        let t: Vec<String> = t.split(',').map(|x| x.replace("skill:", "").replace('_', " ")).collect();
        hello.push_str(&format!(" Last time we talked about {}.", t.join(", ")));
    }
    if p.fresh_streak {
        hello.push_str(&format!(" Day {} in a row, nice!", p.streak));
    }
    (hello, emo)
}

pub fn run(app: &mut App) {
    let (w, h) = term_size();
    app.s.realtime = true;
    app.width = w - LW - 7;
    let mut ui = Ui {
        w,
        h,
        log: Vec::new(),
        console: Vec::new(),
        con_title: String::new(),
        emo: "neutral",
        emo_left: 0,
        talking: false,
        frame: 1,
        analysis: None,
        shown: None,
        scan: None,
        flash: 0,
        anim: true,
        kb: train::model_kb(&app.brain.model),
        rng: Rng::new(crate::rng::time_seed()),
        last_input: Instant::now(),
    };
    print!("\x1b[2J\x1b[?25l");

    // input lives on its own thread so the face stays alive while we wait
    let (tx, rx) = mpsc::channel::<Option<String>>();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        loop {
            let mut l = String::new();
            match stdin.lock().read_line(&mut l) {
                Ok(0) | Err(_) => {
                    let _ = tx.send(None);
                    break;
                }
                Ok(_) => {
                    if tx.send(Some(l)).is_err() {
                        break;
                    }
                }
            }
        }
    });

    let (hello, emo) = welcome(app);
    ui.draw(app);
    ui.set_emo(emo, 40);
    show_reply(&mut ui, app, "welcome", &hello);
    say(&mut ui, Who::Sys, "type /help for commands, /quit to leave");
    ui.draw(app);
    ui.prompt();

    loop {
        let tick = if ui.anim { 140 } else { 1000 };
        let line = match rx.recv_timeout(Duration::from_millis(tick)) {
            Ok(Some(l)) => l,
            Ok(None) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // idle: rest, ring timers, blink, doze
                if let Some(msg) = app.poll_retrain() {
                    ui.kb = train::model_kb(&app.brain.model);
                    say(&mut ui, Who::Sys, &msg);
                }
                let p = &mut app.s.persona;
                p.energy = (p.energy + 0.0015).min(1.0);
                ui.emo_left = ui.emo_left.saturating_sub(1);
                ui.flash = ui.flash.saturating_sub(1);
                for t in app.s.take_due() {
                    print!("\x07");
                    say(&mut ui, Who::Toast, &format!("⏰ {t}"));
                    ui.set_emo("wow", 40);
                    ui.flash = 20;
                }
                ui.draw_idle(app);
                continue;
            }
        };
        ui.last_input = Instant::now();
        ui.clear_input_rows();
        let line = line.trim().to_string();
        if line.is_empty() {
            ui.console.clear();
            ui.draw(app);
            ui.prompt();
            continue;
        }

        if line.starts_with('/') {
            if line == "/anim" {
                ui.anim = !ui.anim;
                let msg = if ui.anim { "animations on" } else { "animations off" };
                ui.console_set("/anim", vec![msg.to_string()]);
            } else if let Some(arg) = line.strip_prefix("/emote") {
                match EMOTIONS.iter().find(|e| **e == arg.trim()) {
                    Some(e) => {
                        ui.set_emo(e, 70);
                        ui.console_set("/emote", vec![format!("making a {e} face")]);
                    }
                    None => ui.console_set("/emote", vec!["usage: /emote <name>".to_string(), EMOTIONS.join(" ")]),
                }
            } else {
                let r = cmds::run(app, &line);
                ui.console_set(&line, r.lines);
                if r.quit {
                    break;
                }
                ui.kb = train::model_kb(&app.brain.model);
            }
            app.s.mem.save();
            ui.draw(app);
            ui.prompt();
            continue;
        }

        // 1. show the user's message right away
        ui.log.push(Line { who: Who::You, segs: vec![(line.clone(), None)] });
        // 2. think: the brain panel flickers and a scanner sweeps the neurons, then it settles
        let reply = app.brain.reply(&mut app.s, &line);
        let mut an = app.brain.analyze(&app.s, &line);
        if let Some(p) = &app.s.last_pred {
            an.probs = p.clone();
        }
        let final_probs = an.probs.clone();
        let sal = an.saliency.clone();
        ui.analysis = Some(an);
        if ui.anim {
            ui.set_emo("think", 30);
            let n = final_probs.len();
            for step in 0..14 {
                let t = step as f32 / 13.0;
                ui.scan = Some(step % 8);
                ui.shown = Some((0..n).map(|i| final_probs[i] * t + ui.rng.f32() * 0.25 * (1.0 - t)).collect());
                ui.draw(app);
                ui.sleep(30);
            }
        }
        ui.shown = None;
        ui.scan = None;
        let words: Vec<&str> = line.split_whitespace().collect();
        if let Some(l) = ui.log.last_mut() {
            l.segs = words.iter().enumerate().map(|(i, wd)| (wd.to_string(), sal.get(i).and_then(|(_, d)| sal_bg(*d)))).collect();
        }
        ui.set_emo(reply.emo, 34);
        ui.draw(app);
        ui.sleep(120);

        // 3. the answer, typed out; lists and art go to the console
        show_reply(&mut ui, app, &reply.tag, &reply.text);
        for t in &reply.toasts {
            ui.sleep(160);
            let big = t.starts_with("LEVEL") || t.starts_with("Achievement") || t.starts_with("Bond");
            say(&mut ui, Who::Toast, t);
            if big {
                ui.set_emo(if t.starts_with("Bond") { "love" } else { "proud" }, 40);
                ui.flash = 12;
                for _ in 0..8 {
                    ui.draw(app);
                    ui.sleep(60);
                }
            }
        }
        if let Some((k, phrase)) = reply.learn {
            let _ = crate::intents::append_learned(&app.brain.model.tags[k].clone(), &phrase, None);
            ui.set_emo("think", 30);
            app.start_retrain();
            say(&mut ui, Who::Sys, &format!("learning {phrase:?} in the background, keep chatting (/unlearn undoes it)"));
        }
        app.s.persona.store(&mut app.s.mem);
        app.s.mem.save();
        app.s.save_flow("flow.txt");
        ui.last_input = Instant::now();
        if reply.done {
            ui.draw(app);
            break;
        }
        ui.draw(app);
        ui.prompt();
    }
    app.s.persona.store(&mut app.s.mem);
    app.s.mem.save();
    print!("\x1b[{};1H\x1b[?25h\n", ui.h);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_emotion_renders_a_well_formed_face() {
        for e in EMOTIONS {
            for t in [0u32, 1, 7, 31, 62, 99] {
                let rows = draw_face(e, t, 81, t % 2 == 0, 0, "label");
                assert_eq!(rows.len(), 8, "{e}");
                for r in rows {
                    assert_eq!(r.w, LW, "{e} at t={t} has a row of width {} (strip ANSI: {:?})", r.w, r.s);
                }
            }
        }
    }

    #[test]
    fn wrap_never_exceeds_width() {
        let segs = vec![("supercalifragilisticexpialidocious_and_then_some a b c".to_string(), None)];
        for l in wrap(&segs, 12) {
            assert!(l.iter().map(|(t, _)| t.chars().count()).sum::<usize>() <= 12);
        }
    }
}
