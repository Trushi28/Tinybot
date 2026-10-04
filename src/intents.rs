use std::fs::{self, OpenOptions};
use std::io::Write;

pub const INTENTS_PATH: &str = "intents.txt";
pub const LEARNED_PATH: &str = "learned.txt";
const DEFAULT_INTENTS: &str = include_str!("../intents.txt");

#[derive(Clone)]
pub struct Intent {
    pub tag: String,
    pub examples: Vec<String>,
    pub responses: Vec<String>,
    /// `>@prev_tag reply`: only used when the previous intent was prev_tag
    pub ctx: Vec<(String, String)>,
}

pub fn parse(src: &str, out: &mut Vec<Intent>) {
    let mut cur: Option<usize> = None;
    for line in src.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(t) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            let t = t.trim();
            cur = Some(match out.iter().position(|i| i.tag == t) {
                Some(i) => i,
                None => {
                    out.push(Intent { tag: t.to_string(), examples: vec![], responses: vec![], ctx: vec![] });
                    out.len() - 1
                }
            });
        } else if let Some(i) = cur {
            match line.strip_prefix('>') {
                Some(r) => {
                    let r = r.trim();
                    match r.strip_prefix('@').and_then(|x| x.split_once(char::is_whitespace)) {
                        Some((prev, text)) => out[i].ctx.push((prev.to_string(), text.trim().to_string())),
                        None => out[i].responses.push(r.to_string()),
                    }
                }
                None => {
                    if !out[i].examples.iter().any(|e| e == line) {
                        out[i].examples.push(line.to_string());
                    }
                }
            }
        }
    }
}

/// intents.txt (or the copy baked into the binary) merged with learned.txt.
pub fn load() -> Vec<Intent> {
    let src = fs::read_to_string(INTENTS_PATH).unwrap_or_else(|_| DEFAULT_INTENTS.to_string());
    let mut v = Vec::new();
    parse(&src, &mut v);
    if let Ok(l) = fs::read_to_string(LEARNED_PATH) {
        parse(&l, &mut v);
    }
    v.retain(|i| !i.examples.is_empty());
    // a tag that only exists in learned.txt via /teach keeps its replies; empty-example ghosts are dropped
    v
}

pub fn flatten(intents: &[Intent]) -> (Vec<String>, Vec<(usize, String)>) {
    let tags = intents.iter().map(|i| i.tag.clone()).collect();
    let mut ex = Vec::new();
    for (k, it) in intents.iter().enumerate() {
        for e in &it.examples {
            ex.push((k, e.clone()));
        }
    }
    (tags, ex)
}

/// Strip anything that could break the line-based file format.
pub fn clean_line(s: &str) -> String {
    let s: String = s.chars().filter(|c| !c.is_control()).collect();
    s.trim().trim_start_matches(['[', '#', '>']).trim().chars().take(200).collect()
}

pub fn clean_tag(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .take(40)
        .collect()
}

pub fn append_learned(tag: &str, phrase: &str, reply: Option<&str>) -> std::io::Result<()> {
    let (tag, phrase) = (clean_tag(tag), clean_line(phrase));
    if tag.is_empty() || phrase.is_empty() {
        return Ok(());
    }
    let mut f = OpenOptions::new().create(true).append(true).open(LEARNED_PATH)?;
    writeln!(f, "\n[{tag}]\n{phrase}")?;
    if let Some(r) = reply {
        let r = clean_line(r);
        if !r.is_empty() {
            writeln!(f, "> {r}")?;
        }
    }
    Ok(())
}

/// Drops the most recently appended block ("\n[tag]\nphrase[\n> reply]") from learned.txt text.
pub fn strip_last_block(src: &str) -> (String, Option<String>) {
    match src.rfind("\n[") {
        Some(i) => {
            let removed = src[i..].trim().replace('\n', " | ");
            (src[..i].to_string(), Some(removed))
        }
        None => (src.to_string(), None),
    }
}

pub fn unlearn_last() -> Option<String> {
    let src = fs::read_to_string(LEARNED_PATH).ok()?;
    let (rest, removed) = strip_last_block(&src);
    removed.as_ref()?;
    fs::write(LEARNED_PATH, rest.trim_end().to_string() + "\n").ok()?;
    removed
}
