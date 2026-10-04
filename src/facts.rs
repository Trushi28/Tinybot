//! The fact book: categorised trivia from facts.txt, with "have you heard this one" tracking.

use crate::rng::Rng;
use crate::text::{fnv, FNV_INIT};
use std::collections::BTreeSet;
use std::fs;

const DEFAULT: &str = include_str!("../facts.txt");

pub struct Fact {
    pub cat: String,
    pub text: String,
    pub id: String, // stable short hash, survives reordering facts.txt
}

pub struct Facts {
    pub items: Vec<Fact>,
}

pub fn load() -> Facts {
    let src = fs::read_to_string("facts.txt").unwrap_or_else(|_| DEFAULT.to_string());
    let mut items = Vec::new();
    for line in src.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((cat, text)) = line.split_once(':') {
            let (cat, text) = (cat.trim().to_lowercase(), text.trim().to_string());
            if !cat.is_empty() && !text.is_empty() {
                let id = format!("{:05x}", fnv(text.as_bytes(), FNV_INIT) & 0xfffff);
                items.push(Fact { cat, text, id });
            }
        }
    }
    Facts { items }
}

impl Facts {
    pub fn categories(&self) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        for f in &self.items {
            if !v.contains(&f.cat) {
                v.push(f.cat.clone());
            }
        }
        v
    }

    /// Prefers facts you haven't heard yet.
    pub fn pick(&self, cat: Option<&str>, seen: &BTreeSet<String>, rng: &mut Rng) -> Option<&Fact> {
        let pool: Vec<&Fact> = self.items.iter().filter(|f| cat.is_none_or(|c| f.cat == c)).collect();
        let fresh: Vec<&Fact> = pool.iter().copied().filter(|f| !seen.contains(&f.id)).collect();
        let from = if fresh.is_empty() { &pool } else { &fresh };
        if from.is_empty() {
            None
        } else {
            Some(from[rng.below(from.len())])
        }
    }

    pub fn daily(&self, day: i64) -> Option<&Fact> {
        if self.items.is_empty() {
            None
        } else {
            Some(&self.items[day.rem_euclid(self.items.len() as i64) as usize])
        }
    }
}

pub fn detect_category(text: &str) -> Option<&'static str> {
    const MAP: [(&str, &[&str]); 7] = [
        ("space", &["space", "planet", "planets", "star", "stars", "moon", "galaxy", "universe", "astronomy", "mars", "venus", "jupiter", "saturn", "sun", "cosmos"]),
        ("animals", &["animal", "animals", "creature", "creatures", "bird", "birds", "fish", "shark", "octopus", "wildlife", "pet", "pets"]),
        ("science", &["science", "physics", "chemistry", "scientific", "nature", "weather"]),
        ("tech", &["tech", "technology", "computer", "computers", "programming", "coding", "code", "software", "internet", "rust", "hardware"]),
        ("history", &["history", "historical", "ancient", "past", "war", "empire"]),
        ("body", &["body", "human", "health", "brain", "anatomy", "biology"]),
        ("math", &["math", "maths", "mathematics", "number", "numbers", "geometry"]),
    ];
    let low = text.to_lowercase();
    let ws: Vec<&str> = low.split(|c: char| !c.is_alphanumeric()).collect();
    MAP.iter().find(|(_, kws)| kws.iter().any(|k| ws.contains(k))).map(|(c, _)| *c)
}
