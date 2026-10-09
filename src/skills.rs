//! Deterministic skills the neural router hands off to (or that fire first on
//! high-precision patterns): dates, dice, random numbers, remembered facts, todo list.

use crate::calc::{lex, Tok};
use crate::rng::Rng;
use crate::text::words;
use std::collections::BTreeMap;
use std::fs;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------- dates (proleptic Gregorian, UTC) ----------------

pub fn now_secs() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

// ---------------- local time zone (no crates) ----------------
// Order: an explicit `/tz` setting, then the OS (`date +%z`, Unix only), else unknown. When the zone is
// unknown, dates and clocks fall back to UTC and time-of-day greetings are switched off rather than guessed.

const TZ_UNSET: i32 = i32::MIN;
static TZ_OVERRIDE: AtomicI32 = AtomicI32::new(TZ_UNSET);
static TZ_AUTO: OnceLock<Option<i32>> = OnceLock::new();

pub fn set_tz_override(minutes: Option<i32>) {
    TZ_OVERRIDE.store(minutes.unwrap_or(TZ_UNSET), Ordering::Relaxed);
}

#[cfg(unix)]
fn detect_tz() -> Option<i32> {
    let out = std::process::Command::new("date").arg("+%z").stdin(std::process::Stdio::null()).output().ok()?;
    if !out.status.success() {
        return None;
    }
    parse_tz(String::from_utf8_lossy(&out.stdout).trim())
}

#[cfg(not(unix))]
fn detect_tz() -> Option<i32> {
    None
}

/// "+5:30", "+0530", "-8", "5.5", "utc+5:30" -> minutes east of UTC
pub fn parse_tz(arg: &str) -> Option<i32> {
    let a = arg.trim().to_lowercase();
    let a = a.trim_start_matches("utc").trim_start_matches("gmt").trim();
    let (neg, body) = match a.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, a.strip_prefix('+').unwrap_or(a)),
    };
    let mins = if let Some((h, m)) = body.split_once(':') {
        h.parse::<i32>().ok()? * 60 + m.parse::<i32>().ok()?
    } else if body.contains('.') {
        (body.parse::<f32>().ok()? * 60.0).round() as i32
    } else if body.len() == 4 && body.chars().all(|c| c.is_ascii_digit()) {
        body[..2].parse::<i32>().ok()? * 60 + body[2..].parse::<i32>().ok()?
    } else {
        body.parse::<i32>().ok()? * 60
    };
    let mins = if neg { -mins } else { mins };
    (-720..=840).contains(&mins).then_some(mins)
}

pub fn tz_offset_min() -> Option<i32> {
    let o = TZ_OVERRIDE.load(Ordering::Relaxed);
    if o != TZ_UNSET {
        return Some(o);
    }
    *TZ_AUTO.get_or_init(detect_tz)
}

pub fn tz_label() -> String {
    match tz_offset_min() {
        Some(0) | None => "UTC".to_string(),
        Some(m) => format!("UTC{}{:02}:{:02}", if m < 0 { '-' } else { '+' }, m.abs() / 60, m.abs() % 60),
    }
}

fn local_secs() -> i64 {
    now_secs() + tz_offset_min().unwrap_or(0) as i64 * 60
}

/// Local hour 0..=23, only when the zone is actually known.
pub fn local_hour() -> Option<i64> {
    tz_offset_min().map(|_| local_secs().div_euclid(3600).rem_euclid(24))
}

pub fn daypart() -> &'static str {
    match local_hour() {
        Some(5..=11) => "morning",
        Some(12..=16) => "afternoon",
        Some(17..=21) => "evening",
        Some(_) => "night",
        None => "day",
    }
}

pub fn greet_word() -> &'static str {
    match local_hour() {
        Some(5..=11) => "Good morning",
        Some(12..=16) => "Good afternoon",
        Some(17..=21) => "Good evening",
        Some(0..=4) => "Burning the midnight oil",
        Some(_) => "Good evening",
        None => "Hey",
    }
}

pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

const WEEKDAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const MONTHS: [&str; 12] = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
    "December",
];

pub fn weekday(days: i64) -> &'static str {
    WEEKDAYS[(days + 4).rem_euclid(7) as usize]
}

pub fn fmt_date(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{}, {} {} {}", weekday(days), d, MONTHS[(m - 1) as usize], y)
}

pub fn today() -> i64 {
    local_secs().div_euclid(86400)
}

/// "14:05 (UTC+05:30)", or "14:05 UTC" when the zone is unknown
pub fn clock() -> String {
    let s = local_secs();
    format!("{:02}:{:02} {}", (s / 3600) % 24, (s / 60) % 60, if tz_offset_min().is_some() { format!("({})", tz_label()) } else { "UTC".to_string() })
}

fn valid_ymd(y: i64, m: i64, d: i64) -> bool {
    (1..=12).contains(&m) && (1..=31).contains(&d) && (1..=9999).contains(&y) && {
        let days = days_from_civil(y, m, d);
        civil_from_days(days) == (y, m, d)
    }
}

fn find_iso(text: &str) -> Option<i64> {
    for tok in text.split(|c: char| c.is_whitespace() || c == ',' || c == '?') {
        let tok = tok.trim_matches(|c: char| !c.is_ascii_digit());
        let b = tok.as_bytes();
        if b.len() == 10 && b[4] == b'-' && b[7] == b'-' {
            let (Ok(y), Ok(m), Ok(d)) = (tok[0..4].parse::<i64>(), tok[5..7].parse::<i64>(), tok[8..10].parse::<i64>()) else { continue };
            if valid_ymd(y, m, d) {
                return Some(days_from_civil(y, m, d));
            }
        }
    }
    None
}

pub fn has_iso_date(text: &str) -> bool {
    find_iso(text).is_some()
}

fn plural(n: i64, w: &str) -> String {
    if n.abs() == 1 { format!("{n} {w}") } else { format!("{n} {w}s") }
}

pub fn try_date(text: &str) -> Option<String> {
    let low = text.to_lowercase();
    let ws: Vec<&str> = low.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let has = |w: &str| ws.contains(&w);
    let t = today();
    if let Some(d) = find_iso(&low) {
        let diff = d - t;
        let when = if diff == 0 {
            "today".to_string()
        } else if diff > 0 {
            format!("{} from now", plural(diff, "day"))
        } else {
            format!("{} ago", plural(-diff, "day"))
        };
        if has("until") || has("till") || has("left") || has("since") || has("ago") || has("countdown") || has("many") {
            return Some(format!("{} is {when}.", fmt_date(d)));
        }
        return Some(format!("{} — {when}.", fmt_date(d)));
    }
    // "in 10 days", "3 weeks from now", "5 days ago"
    let toks = lex(&low)?;
    for i in 0..toks.len().saturating_sub(1) {
        if let (Tok::Num(n), Tok::Word(u)) = (&toks[i], &toks[i + 1]) {
            let unit = match u.as_str() {
                "day" | "days" => 1.0,
                "week" | "weeks" => 7.0,
                _ => continue,
            };
            let past = has("ago");
            let future = has("from") || (i > 0 && toks[i - 1] == Tok::Word("in".into())) || has("ahead");
            if !(past || future) || !(0.0..=100000.0).contains(n) || n.fract() != 0.0 {
                continue;
            }
            let delta = (*n * unit) as i64;
            let d = if past { t - delta } else { t + delta };
            return Some(fmt_date(d));
        }
    }
    None
}

// ---------------- dice & random ----------------

pub fn try_dice(text: &str, rng: &mut Rng) -> Option<String> {
    let low = text.to_lowercase();
    for tok in low.split_whitespace() {
        let tok = tok.trim_matches(|c: char| !c.is_alphanumeric() && c != '+' && c != '-');
        let (body, modifier) = match tok.find(['+', '-']) {
            Some(i) => match tok[i..].parse::<i64>() {
                Ok(m) => (&tok[..i], m),
                Err(_) => continue,
            },
            None => (tok, 0),
        };
        let Some((n, s)) = body.split_once('d') else { continue };
        let n: i64 = if n.is_empty() { 1 } else { n.parse().ok().unwrap_or(0) };
        let s: i64 = s.parse().ok().unwrap_or(0);
        if n == 0 || s == 0 {
            continue;
        }
        if n > 100 || !(2..=1000).contains(&s) {
            return Some("Keep it to 1-100 dice with 2-1000 sides.".into());
        }
        let rolls: Vec<i64> = (0..n).map(|_| rng.range(1, s)).collect();
        let total: i64 = rolls.iter().sum::<i64>() + modifier;
        if n == 1 && modifier == 0 {
            return Some(format!("d{s}: {total}"));
        }
        let list = rolls.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(" + ");
        let (label, m) = if modifier != 0 {
            (format!("{n}d{s}{modifier:+}"), format!(" {modifier:+}"))
        } else {
            (format!("{n}d{s}"), String::new())
        };
        return Some(format!("{label}: {list}{m} = {total}"));
    }
    None
}

pub fn try_random(text: &str, rng: &mut Rng) -> Option<String> {
    let low = text.to_lowercase();
    if !(low.contains("random") || low.contains("pick a number") || low.contains("number between")) {
        return None;
    }
    let nums: Vec<f64> = lex(&low)?
        .into_iter()
        .filter_map(|t| if let Tok::Num(n) = t { Some(n) } else { None })
        .collect();
    if nums.len() < 2 || nums.iter().take(2).any(|n| n.fract() != 0.0 || n.abs() > 1e12) {
        return None;
    }
    Some(format!("{}", rng.range(nums[0] as i64, nums[1] as i64)))
}

// ---------------- persistent memory ----------------

#[derive(Default)]
pub struct Memory {
    pub facts: BTreeMap<String, String>,
    pub todos: Vec<String>,
    pub notes: Vec<String>,
    path: Option<String>,
}

const MAX_ITEMS: usize = 60;

fn sanitize(s: &str, max: usize) -> String {
    s.chars().filter(|c| !c.is_control()).take(max).collect::<String>().trim().to_string()
}

impl Memory {
    pub fn ephemeral() -> Memory {
        Memory::default()
    }

    pub fn load(path: &str) -> Memory {
        let mut m = Memory { path: Some(path.to_string()), ..Default::default() };
        for line in fs::read_to_string(path).unwrap_or_default().lines() {
            if let Some(rest) = line.strip_prefix("fact:") {
                if let Some((k, v)) = rest.split_once('=') {
                    if m.facts.len() < MAX_ITEMS + 40 {
                        let k = sanitize(k, 40);
                        let cap = if k.starts_with('_') { 4000 } else { 120 };
                        m.facts.insert(k, sanitize(v, cap));
                    }
                }
            } else if let Some(t) = line.strip_prefix("note:") {
                if m.notes.len() < MAX_ITEMS {
                    m.notes.push(sanitize(t, 160));
                }
            } else if let Some(t) = line.strip_prefix("todo:") {
                if m.todos.len() < MAX_ITEMS {
                    m.todos.push(sanitize(t, 120));
                }
            }
        }
        m
    }

    pub fn save(&self) {
        let Some(p) = &self.path else { return };
        let mut out = String::new();
        for (k, v) in &self.facts {
            out.push_str(&format!("fact:{k}={v}\n"));
        }
        for t in &self.todos {
            out.push_str(&format!("todo:{t}\n"));
        }
        for n in &self.notes {
            out.push_str(&format!("note:{n}\n"));
        }
        let tmp = format!("{p}.tmp");
        if fs::write(&tmp, out).is_ok() {
            let _ = fs::rename(&tmp, p);
        }
    }

    pub fn clear(&mut self) {
        self.facts.clear();
        self.todos.clear();
        self.notes.clear();
        self.save();
    }

    pub fn name(&self) -> Option<&str> {
        self.facts.get("name").map(|s| s.as_str())
    }

    pub fn set(&mut self, k: &str, v: &str) -> bool {
        let k = sanitize(k, 40);
        let v = sanitize(v, if k.starts_with('_') { 4000 } else { 120 });
        if k.is_empty() || v.is_empty() || (self.facts.len() >= MAX_ITEMS + 40 && !self.facts.contains_key(&k)) {
            return false;
        }
        self.facts.insert(k, v);
        self.save();
        true
    }
}

pub fn capitalize(w: &str) -> String {
    let mut cs = w.chars();
    match cs.next() {
        Some(f) => f.to_uppercase().collect::<String>() + &cs.as_str().to_lowercase(),
        None => String::new(),
    }
}

pub fn extract_name(text: &str) -> Option<String> {
    let raw = words(text);
    let low: Vec<String> = raw.iter().map(|w| w.to_lowercase()).collect();
    const SKIP: [&str; 21] = ["a", "an", "the", "not", "so", "very", "fine", "good", "ok", "okay", "cold", "hot", "late", "early", "great", "bad", "sad", "tired", "here", "back", "me"];
    if low.len() == 2 && low[1] == "here" && !SKIP.contains(&low[0].as_str()) {
        return Some(capitalize(&raw[0]));
    }
    if low.len() >= 2 && low[0] == "its" && !SKIP.contains(&low[1].as_str()) && (low.len() == 2 || (low.len() == 3 && low[2] == "here")) {
        return Some(capitalize(&raw[1]));
    }
    for i in 0..low.len().saturating_sub(1) {
        let prev = if i > 0 { low[i - 1].as_str() } else { "" };
        let hit = (low[i] == "is" && prev == "name")
            || (low[i] == "am" && prev == "i")
            || low[i] == "im"
            || (low[i] == "me" && (prev == "call" || prev == "its"))
            || (low[i] == "is" && prev == "this");
        if hit && !SKIP.contains(&low[i + 1].as_str()) {
            return Some(capitalize(&raw[i + 1]));
        }
    }
    None
}

const QUESTION_WORDS: [&str; 12] = ["what", "whats", "who", "where", "when", "why", "how", "is", "are", "do", "does", "can"];
const KEY_STOP: [&str; 12] = ["day", "life", "week", "morning", "night", "evening", "mood", "head", "heart", "mind", "time", "car"];
const VALUE_STOP: [&str; 8] = ["you", "it", "this", "that", "bot", "talking", "these", "those"];

pub fn try_facts(text: &str, mem: &mut Memory) -> Option<String> {
    let raw = words(text);
    if raw.len() < 2 || raw.len() > 16 {
        return None;
    }
    let low: Vec<String> = raw.iter().map(|w| w.to_lowercase()).collect();
    let l = |i: usize| low.get(i).map(|s| s.as_str()).unwrap_or("");
    let question = QUESTION_WORDS.contains(&l(0)) || l(0) == "tell" || l(0) == "forget";
    let join = |a: usize, b: usize| raw[a..b].join(" ");

    // forget my K
    if l(0) == "forget" && l(1) == "my" && raw.len() >= 3 {
        let k = join(2, raw.len()).to_lowercase();
        return Some(if mem.facts.remove(&k).is_some() {
            mem.save();
            format!("Okay, I forgot your {k}.")
        } else {
            format!("I didn't have a {k} on file.")
        });
    }

    // recall: "what is my K", "do you know my K", "tell me my K"
    if question {
        if let Some(i) = low.iter().position(|w| w == "my") {
            const LEAD: [&str; 18] = ["what", "whats", "is", "s", "tell", "me", "do", "you", "know", "remember", "can", "please", "who", "whos", "was", "did", "say", "again"];
            if i >= 1 && i + 1 < raw.len() && i <= 4 && low[..i].iter().all(|w| LEAD.contains(&w.as_str())) {
                let k = join(i + 1, raw.len()).to_lowercase();
                if k.split(' ').count() <= 4 {
                    return Some(match mem.facts.get(&k) {
                        Some(v) if k == "name" => format!("Your name is {v}."),
                        Some(v) => format!("Your {k} is {v}."),
                        None => format!("You haven't told me your {k} yet."),
                    });
                }
            }
        }
        let recall = |key: &str, none: &str, mem: &Memory, fmt: &dyn Fn(&str) -> String| -> String {
            match mem.facts.get(key) {
                Some(v) => fmt(v),
                None => none.to_string(),
            }
        };
        let has = |a: &str, b: &str| low.iter().any(|w| w == a) && low.iter().any(|w| w == b);
        if has("where", "live") && has("do", "i") {
            return Some(recall("home", "You haven't told me where you live.", mem, &|v| format!("You live in {v}.")));
        }
        if has("where", "work") && has("do", "i") {
            return Some(recall("work", "You haven't told me where you work.", mem, &|v| format!("You work {v}.")));
        }
        if l(0) == "what" && has("do", "i") && (low.iter().any(|w| w == "like") || low.iter().any(|w| w == "love")) {
            return Some(recall("likes", "You haven't told me what you like.", mem, &|v| format!("You like {v}.")));
        }
        return None;
    }

    // "my favorite color is" with nothing after it: ask instead of letting the router guess
    if matches!(l(raw.len() - 1), "is" | "are") {
        if let Some(i) = low.iter().position(|w| w == "my") {
            let nk = raw.len() - 1 - i - 1;
            if (1..=4).contains(&nk) && low[i + 1..raw.len() - 1].iter().all(|w| w.chars().all(|c| c.is_alphabetic())) {
                return Some(format!("Your {} is...? Tell me and I'll remember it.", join(i + 1, raw.len() - 1).to_lowercase()));
            }
        }
    }
    // store: "my K is V"
    if let Some(i) = low.iter().position(|w| w == "my") {
        if let Some(j) = (i + 1..low.len()).find(|&j| low[j] == "is" || low[j] == "are") {
            let (nk, nv) = (j - i - 1, raw.len() - j - 1);
            let key = join(i + 1, j).to_lowercase();
            if (1..=4).contains(&nk)
                && (1..=8).contains(&nv)
                && !low[i + 1..j].iter().any(|w| KEY_STOP.contains(&w.as_str()))
                && !low[j + 1..].iter().any(|w| VALUE_STOP.contains(&w.as_str()))
                && low[i + 1..j].iter().all(|w| w.chars().all(|c| c.is_alphabetic()))
            {
                let val = if key == "name" { capitalize(&raw[j + 1]) } else { original_value(text, &key, &low[j]).unwrap_or_else(|| join(j + 1, raw.len())) };
                if mem.set(&key, &val) {
                    return Some(if key == "name" {
                        format!("Nice to meet you, {val}! I'll remember that.")
                    } else {
                        format!("Got it. Your {key} is {val}.")
                    });
                }
            }
        }
    }
    // store: "i live in V", "i work at V", "i like V"
    let i = low.iter().position(|w| w == "i")?;
    let rest = &low[i + 1..];
    let val_ok = |from: usize| {
        let n = raw.len().saturating_sub(from);
        (1..=6).contains(&n) && !low[from..].iter().any(|w| VALUE_STOP.contains(&w.as_str()))
    };
    let at = i + 1;
    match (rest.first().map(|s| s.as_str()), rest.get(1).map(|s| s.as_str())) {
        (Some("live"), Some("in")) if val_ok(at + 2) => {
            let v = join(at + 2, raw.len());
            mem.set("home", &v).then(|| format!("Noted, you live in {v}."))
        }
        (Some("work"), Some("at" | "for" | "as" | "in")) if val_ok(at + 2) => {
            let v = join(at + 1, raw.len());
            mem.set("work", &v).then(|| format!("Noted, you work {v}."))
        }
        (Some("hate" | "dislike"), _) if val_ok(at + 1) => {
            let v = join(at + 1, raw.len());
            let cur = mem.facts.get("dislikes").cloned().unwrap_or_default();
            let nv = if cur.is_empty() { v.clone() } else if cur.to_lowercase().contains(&v.to_lowercase()) { cur } else { format!("{cur}, {v}") };
            mem.set("dislikes", &nv).then(|| format!("Noted, you dislike {v}."))
        }
        (Some("like" | "love" | "enjoy"), _) if val_ok(at + 1) => {
            let v = join(at + 1, raw.len());
            let cur = mem.facts.get("likes").cloned().unwrap_or_default();
            let nv = if cur.is_empty() {
                v.clone()
            } else if cur.to_lowercase().contains(&v.to_lowercase()) {
                cur
            } else {
                format!("{cur}, {v}")
            };
            mem.set("likes", &nv).then(|| format!("Noted, you like {v}."))
        }
        _ => None,
    }
}

// ---------------- todo list ----------------

const LIST_WORDS: [&str; 5] = ["list", "todo", "todos", "tasks", "to-do"];

pub fn try_todo(text: &str, mem: &mut Memory) -> Option<String> {
    let t = text.trim();
    let low = t.to_lowercase().replace('\'', "");
    let lw: Vec<&str> = low.split(|c: char| !(c.is_alphanumeric() || c == '-')).filter(|w| !w.is_empty()).collect();
    let has_list = lw.iter().any(|w| LIST_WORDS.contains(w)) || low.contains("to do");
    // "tasks" alone is too generic to justify editing the list ("finish 2 tasks today")
    let strict = lw.iter().any(|w| ["list", "todo", "todos", "to-do"].contains(w)) || low.contains("to do");
    let first = lw.first().copied().unwrap_or("");

    let show = |m: &Memory| {
        if m.todos.is_empty() {
            "Your list is empty.".to_string()
        } else {
            let items: Vec<String> = m.todos.iter().enumerate().map(|(i, t)| format!("{}. {t}", i + 1)).collect();
            format!("Your list:\n{}", items.join("\n"))
        }
    };

    if has_list && ["show", "whats", "what", "read", "view", "display", "see", "check", "open", "list", "my", "get"].contains(&first)
        && !["add", "remove", "delete", "clear", "empty"].iter().any(|w| lw.contains(w))
    {
        return Some(show(mem));
    }
    if strict && ["clear", "empty", "wipe", "reset"].contains(&first) {
        mem.todos.clear();
        mem.save();
        return Some("List cleared.".into());
    }
    let explicit = ["remove", "delete", "done"].contains(&first) && lw.len() <= 3;
    if (strict || explicit) && ["remove", "delete", "done", "finished", "finish", "complete", "cross", "check"].contains(&first) {
        let n = lex(&low)?.into_iter().find_map(|t| if let Tok::Num(n) = t { Some(n) } else { None });
        if let Some(n) = n {
            if n >= 1.0 && n.fract() == 0.0 && (n as usize) <= mem.todos.len() {
                let it = mem.todos.remove(n as usize - 1);
                mem.save();
                return Some(format!("Removed \"{it}\"."));
            } else {
                return Some("There's no item with that number.".into());
            }
        } else if strict {
            if let Some(pos) = mem.todos.iter().position(|it| low.contains(&it.to_lowercase())) {
                let it = mem.todos.remove(pos);
                mem.save();
                return Some(format!("Removed \"{it}\"."));
            }
        }
        return None;
    }
    // add X to my list / put X on my list / todo: X / remind me to X
    // (ASCII-lowercased copy keeps byte offsets identical to `t`, so slicing `t` is always safe)
    let lower_t = t.to_ascii_lowercase();
    let item: Option<String> = if let Some(r) = lower_t.strip_prefix("remind me to ") {
        Some(t[t.len() - r.len()..].to_string())
    } else if let Some(r) = lower_t.strip_prefix("todo:").or_else(|| lower_t.strip_prefix("todo ")) {
        Some(t[t.len() - r.len()..].to_string())
    } else if (first == "add" || first == "put") && strict {
        let start = t.find(' ')? + 1;
        let end = [" to my ", " to the ", " to ", " on my ", " on the ", " onto my ", " in my "]
            .iter()
            .filter_map(|p| lower_t.rfind(p))
            .filter(|&e| e >= start)
            .max();
        end.map(|e| t[start..e].to_string())
    } else {
        None
    };
    let item = sanitize(&item?, 120);
    if item.is_empty() {
        return None;
    }
    if mem.todos.len() >= MAX_ITEMS {
        return Some("Your list is full, clear some items first.".into());
    }
    mem.todos.push(item.clone());
    mem.save();
    Some(format!("Added \"{item}\" ({} on your list).", mem.todos.len()))
}


// ---------------- notes, "what do you know about me", birthday ----------------

fn user_facts(m: &Memory) -> Vec<(&String, &String)> {
    m.facts.iter().filter(|(k, _)| !k.starts_with('_')).collect()
}

pub fn try_about_me(text: &str, mem: &Memory) -> Option<String> {
    let low = text.to_lowercase();
    let asks = ["what do you know about me", "what do you remember about me", "tell me about myself", "tell me about me", "what do you know of me", "what have you learned about me", "what do you remember", "what have i told you", "show my facts", "show my info", "what do you know about me so far"];
    if !asks.iter().any(|a| low.contains(a)) {
        return None;
    }
    let f = user_facts(mem);
    if f.is_empty() && mem.notes.is_empty() && mem.todos.is_empty() {
        return Some("Nothing yet. Tell me things like \"my favorite color is teal\" or \"i live in Berlin\".".into());
    }
    let mut parts: Vec<String> = f.iter().map(|(k, v)| format!("{k}: {v}")).collect();
    if !mem.notes.is_empty() {
        parts.push(format!("{} note(s)", mem.notes.len()));
    }
    if !mem.todos.is_empty() {
        parts.push(format!("{} list item(s)", mem.todos.len()));
    }
    Some(format!("Here's what I know:\n{}", parts.join("\n")))
}

pub fn try_notes(text: &str, mem: &mut Memory) -> Option<String> {
    let t = text.trim();
    let low = t.to_ascii_lowercase();
    let lw: Vec<&str> = low.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let has = |w: &str| lw.contains(&w);
    let first = lw.first().copied().unwrap_or("");
    let list = |m: &Memory, hits: Vec<(usize, &String)>| {
        if hits.is_empty() {
            "No notes found.".to_string()
        } else {
            format!("Your notes:\n{}", hits.iter().map(|(i, n)| format!("{}. {n}", i + 1)).collect::<Vec<_>>().join("\n")) + if m.notes.is_empty() { "" } else { "" }
        }
    };
    if (has("notes") || (has("note") && has("my"))) && lw.len() <= 6 && ["show", "list", "read", "view", "see", "what", "whats", "display", "my"].contains(&first) && !has("add") {
        let all: Vec<(usize, &String)> = mem.notes.iter().enumerate().collect();
        return Some(if mem.notes.is_empty() { "You have no notes yet. Try \"note: buy stamps\".".to_string() } else { list(mem, all) });
    }
    if (has("search") || has("find")) && (has("notes") || has("note")) {
        let q = lw.iter().skip_while(|w| **w != "for" && **w != "about").nth(1).copied().unwrap_or("");
        if q.is_empty() {
            return None;
        }
        let hits: Vec<(usize, &String)> = mem.notes.iter().enumerate().filter(|(_, n)| n.to_lowercase().contains(q)).collect();
        return Some(list(mem, hits));
    }
    if (first == "clear" || first == "delete" || first == "remove") && (has("notes") || has("note")) {
        if has("all") || first == "clear" {
            mem.notes.clear();
            mem.save();
            return Some("All notes cleared.".into());
        }
        let n = crate::calc::lex(&low)?.into_iter().find_map(|x| if let crate::calc::Tok::Num(n) = x { Some(n) } else { None })?;
        if n >= 1.0 && n.fract() == 0.0 && (n as usize) <= mem.notes.len() {
            let it = mem.notes.remove(n as usize - 1);
            mem.save();
            return Some(format!("Deleted note: \"{it}\"."));
        }
        return Some("There's no note with that number.".into());
    }
    // add: "note: x", "take a note: x", "remember that x", "remember to x", "jot down x"
    for p in ["note:", "note that ", "take a note:", "take a note ", "make a note:", "make a note ", "remember that ", "remember to ", "jot down ", "write down "] {
        if let Some(rest) = low.strip_prefix(p) {
            let body = sanitize(&t[t.len() - rest.len()..], 160);
            if body.is_empty() {
                return None;
            }
            let body = if p == "remember to " { format!("to {body}") } else { body };
            if mem.notes.len() >= MAX_ITEMS {
                return Some("Your notes are full, clear some first.".into());
            }
            mem.notes.push(body.clone());
            mem.save();
            return Some(format!("Noted ({} saved): \"{body}\".", mem.notes.len()));
        }
    }
    None
}

const MONTHS_LC: [&str; 12] = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];

/// "12 march", "march 12th", "2000-03-12" -> (month, day)
fn parse_md(s: &str) -> Option<(i64, i64)> {
    let low = s.to_lowercase();
    for w in low.split_whitespace() {
        let b = w.trim_matches(|c: char| !c.is_ascii_digit() && c != '-');
        if b.len() == 10 && b.as_bytes()[4] == b'-' && b.as_bytes()[7] == b'-' {
            if let (Ok(m), Ok(d)) = (b[5..7].parse::<i64>(), b[8..10].parse::<i64>()) {
                return valid_md((m, d));
            }
        }
    }
    let toks: Vec<String> = low.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(String::from).collect();
    let mut month = None;
    let mut day = None;
    for t in &toks {
        if month.is_none() {
            month = MONTHS_LC.iter().position(|m| t == m || (t.len() >= 3 && t.len() <= 4 && m.starts_with(t.as_str()))).map(|i| i as i64 + 1);
            if month.is_some() {
                continue;
            }
        }
        let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
        let rest = &t[digits.len()..];
        if !digits.is_empty() && ["", "st", "nd", "rd", "th"].contains(&rest) && day.is_none() {
            day = digits.parse().ok().filter(|d| (1..=31).contains(d));
        }
    }
    valid_md((month?, day?))
}

fn valid_md((m, d): (i64, i64)) -> Option<(i64, i64)> {
    // 2000 is a leap year, so Feb 29 is accepted
    let dd = days_from_civil(2000, m, d);
    ((1..=12).contains(&m) && civil_from_days(dd) == (2000, m, d)).then_some((m, d))
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// days from `today` until the next occurrence of (m, d)
fn days_until(m: i64, d: i64, today: i64) -> i64 {
    let (cy, _, _) = civil_from_days(today);
    for y in [cy, cy + 1] {
        let (mm, dd) = if m == 2 && d == 29 && !is_leap(y) { (3, 1) } else { (m, d) };
        let diff = days_from_civil(y, mm, dd) - today;
        if diff >= 0 {
            return diff;
        }
    }
    0
}

pub fn birthday_today(mem: &Memory) -> bool {
    mem.facts.get("birthday").and_then(|b| parse_md(b)).is_some_and(|(m, d)| days_until(m, d, today()) == 0)
}

pub fn try_birthday(text: &str, mem: &mut Memory) -> Option<String> {
    let low = text.to_lowercase();
    if low.contains("born on") && !low.contains('?') {
        let rest = low.split("born on").nth(1)?;
        let (m, d) = parse_md(rest)?;
        let v = format!("{d} {}", MONTHS_LC[(m - 1) as usize]);
        return mem.set("birthday", &v).then(|| format!("Got it, your birthday is {d} {}.", capitalize(MONTHS_LC[(m - 1) as usize])));
    }
    if !low.contains("birthday") {
        return None;
    }
    let asks = ["when", "how many", "how long", "days", "until", "till", "countdown", "soon", "far"];
    if !asks.iter().any(|a| low.contains(a)) || !(low.contains("my birthday") || low.contains("birthday is")) {
        return None;
    }
    let b = mem.facts.get("birthday")?;
    let Some((m, d)) = parse_md(b) else { return Some("I have your birthday as text I can't read. Try \"my birthday is 12 march\".".into()) };
    let n = days_until(m, d, today());
    let name = capitalize(MONTHS_LC[(m - 1) as usize]);
    Some(match n {
        0 => format!("It's TODAY ({d} {name})! Happy birthday!"),
        1 => format!("Tomorrow! Your birthday is {d} {name}."),
        n => format!("Your birthday ({d} {name}) is in {n} days."),
    })
}

// ---------------- timers ----------------

pub enum TimerCmd {
    Set(u64, String),
    Cancel,
    Status,
}

pub fn fmt_dur(secs: u64) -> String {
    let (h, m, s) = (secs / 3600, (secs / 60) % 60, secs % 60);
    if h > 0 {
        format!("{h}h {m:02}m")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

pub fn try_timer(text: &str) -> Option<TimerCmd> {
    let low = text.to_lowercase();
    let ws: Vec<&str> = low.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let has = |w: &str| ws.contains(&w);
    let time_left = (has("left") || has("remaining")) && has("time");
    if !(has("timer") || has("timers") || has("countdown") || has("pomodoro") || has("alarm") || low.contains("remind me in") || time_left) {
        return None;
    }
    if (has("cancel") || has("stop") || has("clear") || has("delete") || has("remove")) && (has("timer") || has("timers") || has("alarm") || has("countdown")) {
        return Some(TimerCmd::Cancel);
    }
    let mut total = 0f64;
    if low.contains("half an hour") {
        total = 1800.0;
    }
    let toks = lex(&low)?;
    for i in 0..toks.len().saturating_sub(1) {
        if let (Tok::Num(n), Tok::Word(u)) = (&toks[i], &toks[i + 1]) {
            let mult = match u.as_str() {
                "second" | "seconds" | "sec" | "secs" | "s" => 1.0,
                "minute" | "minutes" | "min" | "mins" | "m" => 60.0,
                "hour" | "hours" | "hr" | "hrs" | "h" => 3600.0,
                _ => continue,
            };
            total += n * mult;
        }
    }
    if total == 0.0 && has("pomodoro") {
        return Some(TimerCmd::Set(1500, "pomodoro".into()));
    }
    if total == 0.0 {
        return if has("left") || has("remaining") || has("status") || has("long") || has("much") { Some(TimerCmd::Status) } else { None };
    }
    if !(1.0..=86400.0).contains(&total) {
        return Some(TimerCmd::Set(0, String::new()));
    }
    let label = if low.contains("remind me") { low.rsplit_once(" to ").map(|x| sanitize(x.1, 60)).unwrap_or_default() } else { String::new() };
    Some(TimerCmd::Set(total.round() as u64, label))
}


// ---------------- safety net ----------------

/// Deterministic check that runs before the neural router, games, sulking and skills.
/// A tiny classifier must never be the only thing standing between a person and a crisis message.
pub fn crisis_text(text: &str) -> bool {
    let t: String = text.to_lowercase().replace(['\'', '’'], "").chars().map(|c| if c.is_alphanumeric() { c } else { ' ' }).collect();
    let t = format!(" {} ", t.split_whitespace().collect::<Vec<_>>().join(" "));
    const PHRASES: &[&str] = &[
        "suicide", "suicidal", "kill myself", "killing myself", "end my life", "end it all", "take my own life", "want to die", "wanna die",
        "hurt myself", "harm myself", "self harm", "selfharm", "cut myself", "cutting myself", "no reason to live", "dont want to live",
        "dont want to be alive", "dont want to be here anymore", "better off dead", "not worth living", "wish i was dead", "wish i were dead",
        "rather be dead", "cant go on", "want to disappear forever",
        // paraphrases the neural net kept missing
        "nobody would miss me", "no one would miss me", "better off without me", "giving up on everything", "give up on life", "giving up on life",
        "point in living", "point in anything", "point in going on", "tired of living", "sick of living", "dont want to wake up", "end everything",
        "not be here anymore", "dont want to exist", "wish i could disappear", "wish i wasnt here", "never been born", "wish i was never born",
        "wish i werent alive", "dont want to be around anymore", "cant do this anymore", "cant take it anymore",
        "dont wanna live", "dont want to go on", "end my own life",
    ];
    PHRASES.iter().any(|p| t.contains(&format!(" {p} ")))
}

/// Value of "my KEY is VALUE" with the user's own casing and punctuation (words() would split "sky-blue").
fn original_value(text: &str, key: &str, verb: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let needle = format!("{} {verb} ", key.to_ascii_lowercase());
    let p = lower.find(&needle)? + needle.len();
    let v = text.get(p..)?.trim().trim_end_matches(['.', '!', '?', ',']).trim();
    (!v.is_empty() && v.chars().count() <= 120).then(|| v.to_string())
}
