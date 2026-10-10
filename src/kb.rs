//! Knowledge base: the part of TinyBot that answers open questions the intent router has never seen.
//!
//! Retrieval, not generation: BM25 over short topics from kb.txt, kb_user.txt (`/know`) and facts.txt.
//! Query terms are stemmed, misspellings snap to the nearest known word, and the answer is the single
//! best line of the best topic. `coverage` says how much of the question the topic actually explains,
//! so "what is the capital of mars" is declined instead of answered with Mars or with a capital.

use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;

const DEFAULT_KB: &str = include_str!("../kb.txt");
pub const USER_KB: &str = "kb_user.txt";
/// below this a topic only half-explains the question and should not be offered as an answer.
/// Coverage is a ratio of IDF weights, so it drifts as topics are added: at 0.70, "what is the capital of mars"
/// (the rare word "Mars" explains it, "capital" does not) crossed the line once the knowledge base grew from
/// 104 to 174 topics. Real questions score 1.0; the half-explained ones sit at 0.67 to 0.74.
pub const MIN_COVERAGE: f32 = 0.75;

/// words that carry no topic: dropped from queries and documents
const STOP: &[&str] = &[
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "am",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "but",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "do",
    "does",
    "did",
    "you",
    "your",
    "yours",
    "me",
    "my",
    "mine",
    "we",
    "us",
    "our",
    "they",
    "he",
    "she",
    "him",
    "her",
    "what",
    "whats",
    "who",
    "whos",
    "whom",
    "where",
    "when",
    "why",
    "how",
    "which",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "tell",
    "about",
    "explain",
    "define",
    "describe",
    "give",
    "know",
    "please",
    "there",
    "with",
    "as",
    "by",
    "from",
    "have",
    "has",
    "had",
    "not",
    "no",
    "yes",
    "some",
    "any",
    "more",
    "also",
    "just",
    "really",
    "very",
    "so",
    "if",
    "then",
    "than",
    "too",
    "want",
    "need",
    "like",
    "let",
    "lets",
    "thing",
    "things",
    "something",
    "work",
    "works",
    "mean",
    "means",
    "meaning",
    "live",
    "lives",
    "many",
    "much",
    "long",
    "big",
    "old",
    "far",
    "fast",
    "large",
    "small",
    "fact",
    "facts",
    "trivia",
    "interesting",
    "i",
    "im",
    "ive",
    "id",
    "ill",
    "name",
    "called",
    "time",
    "speak",
    "speaks",
    "spoken",
    "say",
];

/// one topic as written in a file (or built from a fact)
pub struct Raw {
    pub title: String,
    pub aliases: String,
    pub lines: Vec<String>,
    pub fact: bool,
}

struct Entry {
    lines: Vec<String>,
    line_terms: Vec<HashSet<String>>,
    key_terms: HashSet<String>, // title + aliases
    tf: HashMap<String, f32>,
    len: f32,
    fact: bool,
}

pub struct Kb {
    entries: Vec<Entry>,
    df: HashMap<String, usize>,
    avg_len: f32,
}

pub struct Hit {
    pub entry: usize,
    pub line: usize,
    /// share of the question's content (by rarity) that this topic contains, 0..=1
    pub coverage: f32,
    /// a question word is in the topic's title or keywords, not just somewhere in its text
    pub keyword_hit: bool,
}

// ---------------------------------------------------------------- text

fn stem(w: &str) -> String {
    let mut s = w.to_string();
    let n = s.len();
    if n > 4 && s.ends_with("ies") {
        s.truncate(n - 3);
        s.push('y');
    } else if n > 4 && s.ends_with("es") {
        s.truncate(n - 2);
    } else if n > 3
        && s.ends_with('s')
        && !(s.ends_with("ss") || s.ends_with("us") || s.ends_with("is"))
    {
        s.truncate(n - 1);
    } else if n > 5 && s.ends_with("ing") {
        s.truncate(n - 3);
        let cs: Vec<char> = s.chars().collect();
        if cs.len() > 2
            && cs[cs.len() - 1] == cs[cs.len() - 2]
            && !matches!(cs[cs.len() - 1], 'l' | 's')
        {
            s.pop();
        }
    } else if n > 4 && s.ends_with("ed") {
        s.truncate(n - 2);
    }
    if s.len() > 3 && s.ends_with('e') {
        s.pop();
    }
    s
}

/// content terms of a text: lowercase, no stop words, stemmed
pub fn terms(text: &str) -> Vec<String> {
    let low = text.to_lowercase().replace(['\'', '’'], "");
    low.split(|c: char| !c.is_alphanumeric())
        .filter(|w| {
            let digits = w.chars().all(|c| c.is_ascii_digit());
            let n = w.chars().count();
            n > 0 && if digits { n >= 3 } else { n > 1 } && !STOP.contains(w)
        })
        .map(stem)
        .collect()
}

fn lev(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let sub = prev[j - 1] + (a[i - 1] != b[j - 1]) as usize;
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

// ---------------------------------------------------------------- loading

pub fn parse(src: &str, out: &mut Vec<Raw>) {
    let mut cur: Option<usize> = None;
    for line in src.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(h) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            let (title, aliases) = h
                .split_once('|')
                .map(|(a, b)| (a.trim(), b.trim()))
                .unwrap_or((h.trim(), ""));
            if title.is_empty() {
                cur = None;
                continue;
            }
            let idx = match out
                .iter()
                .position(|r| !r.fact && r.title.eq_ignore_ascii_case(title))
            {
                Some(i) => {
                    if !aliases.is_empty() {
                        out[i].aliases.push(' ');
                        out[i].aliases.push_str(aliases);
                    }
                    i
                }
                None => {
                    out.push(Raw {
                        title: title.to_string(),
                        aliases: aliases.to_string(),
                        lines: vec![],
                        fact: false,
                    });
                    out.len() - 1
                }
            };
            cur = Some(idx);
        } else if let Some(i) = cur {
            out[i].lines.push(line.to_string());
        }
    }
}

/// kb.txt (or the copy baked into the binary) + kb_user.txt + every fact from facts.txt as its own topic
pub fn load(facts: &[(String, String)]) -> Kb {
    let src = fs::read_to_string("kb.txt").unwrap_or_else(|_| DEFAULT_KB.to_string());
    let mut raw = Vec::new();
    parse(&src, &mut raw);
    if let Ok(u) = fs::read_to_string(USER_KB) {
        parse(&u, &mut raw);
    }
    for (cat, text) in facts {
        raw.push(Raw {
            title: String::new(),
            aliases: cat.clone(),
            lines: vec![text.clone()],
            fact: true,
        });
    }
    Kb::new(raw)
}

/// `/know topic => text`: append to kb_user.txt (same topic again adds another line). false if nothing valid to save.
pub fn append_user(topic: &str, text: &str) -> bool {
    let clean = |s: &str, max: usize| -> String {
        s.chars()
            .filter(|c| !c.is_control() && !matches!(c, '[' | ']' | '|'))
            .take(max)
            .collect::<String>()
            .trim()
            .trim_start_matches('#')
            .trim()
            .to_string()
    };
    let (t, x) = (clean(topic, 60), clean(text, 300));
    if t.is_empty() || x.is_empty() {
        return false;
    }
    match OpenOptions::new().create(true).append(true).open(USER_KB) {
        Ok(mut f) => writeln!(f, "\n[{t}]\n{x}").is_ok(),
        Err(_) => false,
    }
}

// ---------------------------------------------------------------- retrieval

impl Kb {
    pub fn new(raw: Vec<Raw>) -> Kb {
        let mut entries = Vec::new();
        let mut df: HashMap<String, usize> = HashMap::new();
        for r in raw {
            if r.lines.is_empty() {
                continue;
            }
            let mut tf: HashMap<String, f32> = HashMap::new();
            let mut key_terms: HashSet<String> = HashSet::new();
            for t in terms(&r.title) {
                *tf.entry(t.clone()).or_default() += 3.0;
                key_terms.insert(t);
            }
            for t in terms(&r.aliases) {
                *tf.entry(t.clone()).or_default() += 2.0;
                key_terms.insert(t);
            }
            let mut line_terms = Vec::new();
            for l in &r.lines {
                let ts = terms(l);
                for t in &ts {
                    *tf.entry(t.clone()).or_default() += 1.0;
                }
                line_terms.push(ts.into_iter().collect::<HashSet<String>>());
            }
            for t in tf.keys() {
                *df.entry(t.clone()).or_default() += 1;
            }
            let len: f32 = tf.values().sum();
            entries.push(Entry {
                lines: r.lines,
                line_terms,
                key_terms,
                tf,
                len,
                fact: r.fact,
            });
        }
        let avg_len = if entries.is_empty() {
            1.0
        } else {
            entries.iter().map(|e| e.len).sum::<f32>() / entries.len() as f32
        };
        Kb {
            entries,
            df,
            avg_len: avg_len.max(1.0),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.iter().filter(|e| !e.fact).count()
    }

    fn idf(&self, df: usize) -> f32 {
        let n = self.entries.len() as f32;
        (1.0 + (n - df as f32 + 0.5) / (df as f32 + 0.5)).ln()
    }

    /// a query term as the knowledge base spells it: itself, else the nearest known word within a typo or two
    fn resolve(&self, t: &str) -> Option<String> {
        if self.df.contains_key(t) {
            return Some(t.to_string());
        }
        let n = t.chars().count();
        if n < 5 {
            return None;
        }
        let max_d = if n >= 9 { 2 } else { 1 };
        self.df
            .keys()
            .filter(|k| k.chars().count().abs_diff(n) <= max_d)
            .map(|k| (lev(t, k), k))
            .filter(|(d, _)| *d <= max_d)
            .min_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(b.1)))
            .map(|(_, k)| k.clone())
    }

    pub fn lookup(&self, query: &str) -> Option<Hit> {
        if self.entries.is_empty() {
            return None;
        }
        let low = query.to_lowercase();
        let fact_ok = low
            .split(|c: char| !c.is_alphanumeric())
            .any(|w| matches!(w, "about"));
        let mut seen = HashSet::new();
        let mut known: Vec<String> = Vec::new();
        let mut unknown = 0usize;
        for t in terms(query) {
            if !seen.insert(t.clone()) {
                continue;
            }
            match self.resolve(&t) {
                Some(r) => {
                    if !known.contains(&r) {
                        known.push(r);
                    }
                }
                None => unknown += 1,
            }
        }
        if known.is_empty() {
            return None;
        }
        let n = self.entries.len() as f32;
        let q_idf: Vec<f32> = known.iter().map(|t| self.idf(self.df[t])).collect();
        let total: f32 = q_idf.iter().sum::<f32>() + unknown as f32 * 0.5 * self.idf(0);
        let (k1, b) = (1.2f32, 0.75f32);
        let mut best: Option<(f32, f32, usize, bool)> = None; // (score, coverage, entry, keyword_hit)
        for (ei, e) in self.entries.iter().enumerate() {
            if e.fact && !fact_ok {
                continue;
            }
            let (mut score, mut matched, mut rare) = (0.0f32, 0.0f32, 0.0f32);
            let (mut count, mut kw) = (0usize, false);
            for (t, w) in known.iter().zip(&q_idf) {
                if let Some(&tf) = e.tf.get(t) {
                    score += w * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * e.len / self.avg_len));
                    matched += w;
                    count += 1;
                    if self.df[t] <= 2 || self.df[t] as f32 * 4.0 <= n {
                        rare += w;
                    }
                    kw |= e.key_terms.contains(t);
                }
            }
            // one stray common word in a body of text is not an answer: need a keyword hit or two shared terms
            let ok = if e.fact { count >= 1 } else { kw || count >= 2 };
            if !ok || rare == 0.0 {
                continue;
            }
            let cov = matched / total;
            let better = match best {
                None => true,
                Some((s, c, _, _)) => cov > c + 1e-4 || ((cov - c).abs() <= 1e-4 && score > s),
            };
            if better {
                best = Some((score, cov, ei, kw));
            }
        }
        let (_, coverage, entry, keyword_hit) = best?;
        // the line that shares the most (rarest) question terms; the title counts for every line
        let e = &self.entries[entry];
        let (mut line, mut top) = (0usize, -1.0f32);
        for (li, lt) in e.line_terms.iter().enumerate() {
            let ov: f32 = known
                .iter()
                .zip(&q_idf)
                .filter(|&(t, _)| lt.contains(t) || e.key_terms.contains(t))
                .map(|(_, w)| *w)
                .sum();
            if ov > top + 1e-6 {
                top = ov;
                line = li;
            }
        }
        Some(Hit {
            entry,
            line,
            coverage,
            keyword_hit,
        })
    }

    /// (text, next line to offer for "more")
    pub fn answer(&self, h: &Hit) -> (String, Option<usize>) {
        let e = &self.entries[h.entry];
        let next = if h.line > 0 {
            Some(0)
        } else if e.lines.len() > 1 {
            Some(1)
        } else {
            None
        };
        (e.lines[h.line].clone(), next)
    }

    /// continuation of an earlier answer
    pub fn more(&self, entry: usize, idx: usize) -> Option<(String, Option<usize>)> {
        let e = self.entries.get(entry)?;
        let t = e.lines.get(idx)?.clone();
        Some((
            t,
            if idx + 1 < e.lines.len() {
                Some(idx + 1)
            } else {
                None
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kb() -> Kb {
        let facts: Vec<(String, String)> = [
            ("animals", "Octopuses have three hearts and blue blood."),
            ("animals", "Wombats produce cube-shaped droppings."),
            ("space", "A day on Venus is longer than its year."),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
        load(&facts)
    }

    fn ask(k: &Kb, q: &str) -> Option<String> {
        k.lookup(q)
            .filter(|h| h.coverage >= MIN_COVERAGE)
            .map(|h| k.answer(&h).0)
    }

    #[test]
    fn stemming_is_consistent() {
        assert_eq!(stem("octopuses"), stem("octopus"));
        assert_eq!(stem("houses"), stem("house"));
        assert_eq!(stem("programming"), stem("program"));
        assert_eq!(stem("countries"), stem("country"));
        assert_eq!(
            terms("What's the capital of France?"),
            vec!["capital".to_string(), "franc".to_string()]
        );
    }

    #[test]
    fn answers_open_questions() {
        let k = kb();
        for (q, want) in [
            ("what is the capital of france", "Paris"),
            ("capital of australia", "Canberra"),
            ("whats the capital of japan", "Tokyo"),
            ("how tall is mount everest", "8,849"),
            ("define photosynthesis", "light"),
            ("what is a black hole", "light"),
            ("who invented the world wide web", "Berners-Lee"),
            ("what is gradient descent", "loss"),
            ("when did humans land on the moon", "1969"),
            ("how many planets are there", "eight"),
            ("what is a prime number", "divisible"),
            ("explain quantum entanglement", "particles"),
        ] {
            let a = ask(&k, q).unwrap_or_else(|| panic!("no answer for {q:?}"));
            assert!(a.contains(want), "{q:?} -> {a}");
        }
    }

    #[test]
    fn newer_topics_answer_and_do_not_steal_older_ones() {
        let k = kb();
        for (q, want) in [
            ("what is the capital of turkey", "Ankara"),
            ("capital of the netherlands", "Amsterdam"),
            ("what language do they speak in switzerland", "Romansh"),
            ("what is the currency of south korea", "won"),
            ("which state is hyderabad the capital of", "Telangana"),
            ("how many states are there in india", "28 states"),
            ("when did chandrayaan 3 land", "23 August 2023"),
            ("who was the first human in space", "Gagarin"),
            ("how fast is a cheetah", "100 km/h"),
            ("how many neurons are in the brain", "86 billion"),
            ("what is the speed of sound", "343"),
            ("what is absolute zero", "273.15"),
            ("what does dns do", "IP addresses"),
            ("what is binary search", "log2"),
            ("who was ada lovelace", "algorithm"),
            ("when did the berlin wall fall", "1989"),
            ("how many pieces does each side have in chess", "16"),
            // older topics must still win their own questions
            ("what is the capital of france", "Paris"),
            ("how many people live in france", "68 million"),
            ("who invented the world wide web", "Berners-Lee"),
            ("when did humans land on the moon", "1969"),
            ("how tall is mount everest", "8,849"),
            ("what is a neural network", "layers"),
            ("explain the fibonacci sequence", "sum"),
        ] {
            let a = ask(&k, q).unwrap_or_else(|| panic!("no answer for {q:?}"));
            assert!(a.contains(want), "{q:?} -> {a}");
        }
        // still declines nonsense instead of grabbing a nearby topic
for q in ["what is the capital of mars", "what is the capital of jupiter", "capital of the moon", "what is the language of saturn", "what is the population of venus"] {
            assert!(ask(&k, q).is_none(), "{q:?} should be declined");
        }
    }

    #[test]
    fn picks_the_line_that_answers() {
        let k = kb();
        assert!(
            ask(&k, "how many people live in france")
                .unwrap()
                .contains("68 million")
        );
        assert!(
            ask(&k, "what language do they speak in brazil")
                .unwrap()
                .contains("Portuguese")
        );
    }

    #[test]
    fn survives_typos() {
        let k = kb();
        assert!(ask(&k, "what is photosyntesis").unwrap().contains("light"));
        assert!(ask(&k, "capital of austrlia").unwrap().contains("Canberra"));
    }

    #[test]
    fn declines_what_it_does_not_know() {
        let k = kb();
        for q in [
            "what is the capital of mars",
            "hello how are you",
            "tell me a joke",
            "what is your name",
            "what time is it",
            "i am so tired today",
            "who won the football match last night",
            "asdf qwer zxcv",
            "",
        ] {
            assert!(
                ask(&k, q).is_none(),
                "{q:?} should not be answered: {:?}",
                ask(&k, q)
            );
        }
    }

    #[test]
    fn facts_only_answer_when_asked_about() {
        let k = kb();
        assert!(
            ask(&k, "tell me about octopuses")
                .unwrap()
                .contains("three hearts")
        );
        assert!(ask(&k, "octopus hearts").is_none());
    }

    #[test]
    fn more_walks_through_a_topic() {
        let k = kb();
        let h = k.lookup("what is the capital of france").unwrap();
        let (first, next) = k.answer(&h);
        assert!(first.contains("Paris"));
        let (second, next2) = k.more(h.entry, next.unwrap()).unwrap();
        assert!(second.contains("68 million"), "{second}");
        assert!(next2.is_some());
    }

    #[test]
    fn user_topics_merge_by_title() {
        let mut raw = Vec::new();
        parse(
            "[Zorp | planet]\nZorp is a made-up planet.\n[zorp]\nIt has two suns.\n",
            &mut raw,
        );
        assert_eq!(raw.len(), 1);
        assert_eq!(raw[0].lines.len(), 2);
        let k = Kb::new(raw);
        assert!(ask(&k, "what is zorp").unwrap().contains("made-up"));
        assert!(
            ask(&k, "how many suns does zorp have")
                .unwrap()
                .contains("two suns")
        );
    }
}
