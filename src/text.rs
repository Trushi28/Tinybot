use crate::rng::Rng;
use std::collections::HashMap;
use std::sync::OnceLock;

pub const BUCKETS: usize = 1 << 14;
/// Feature ids >= CHAR_BASE are char n-grams (pooled separately from word/bigram features).
pub const CHAR_BASE: usize = 1 << 20;

/// Row of a `1 << bits` embedding table for a feature id (drops the char-channel flag).
#[inline]
pub fn bucket_in(f: usize, bits: u32) -> usize {
    f & ((1usize << bits) - 1)
}

/// Split into alphanumeric words; apostrophes are dropped ("what's" -> "whats").
pub fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .map(|w| w.chars().filter(|&c| c != '\'').collect::<String>())
        .filter(|w| !w.is_empty())
        .collect()
}

pub fn fnv(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub const FNV_INIT: u64 = 0xcbf29ce484222325;

pub fn hash(s: &str) -> usize {
    (fnv(s.as_bytes(), FNV_INIT) % BUCKETS as u64) as usize
}

/// Lowercase; pure digit runs collapse to a single placeholder so "5 plus 3" ~ "9 plus 2".
fn norm_word(w: &str) -> String {
    if !w.is_empty() && w.chars().all(|c| c.is_ascii_digit()) {
        "<num>".to_string()
    } else {
        w.to_lowercase()
    }
}

pub fn norm_words(text: &str) -> Vec<String> {
    words(text).iter().map(|w| norm_word(w)).collect()
}


/// Small hand-written semantic lexicon. Each word also emits its class as a feature, so a
/// phrase like "i feel lonely" can lean on what "sad", "down" and "miserable" taught the model.
const NEGATORS: [&str; 8] = ["not", "no", "never", "dont", "isnt", "cant", "wasnt", "aint"];

const LEXICON: &[(&str, &str)] = &[
    ("pos_feel", "good great happy fine awesome fantastic amazing wonderful excellent nice lovely cheerful glad joyful excited thrilled pleased content fabulous superb"),
    ("neg_feel", "sad down lonely alone depressed miserable awful terrible horrible upset anxious stressed tired exhausted hopeless worried scared angry annoyed bad rough low blue wrong drained overwhelmed gloomy"),
    ("insult", "stupid dumb idiot useless trash garbage worthless pathetic suck sucks worst annoying hate ugly lame boring bad awful terrible moron"),
    ("praise", "smart awesome cool amazing great best brilliant genius impressive funny clever wonderful nice love rock rocks fantastic perfect good excellent"),
    ("farewell", "bye goodbye leaving leave later farewell night gone tomorrow go going signing"),
    ("greet", "hi hello hey yo howdy hiya greetings hola morning evening afternoon sup heya"),
    ("thank", "thanks thank thx ty cheers appreciate appreciated grateful"),
    ("affirm", "yes yeah yep yup sure ok okay absolutely correct right fine definitely"),
    ("deny", "no nope nah never negative nothing"),
    ("machine", "robot bot ai human machine computer program alive real person artificial software"),
    ("weather", "weather rain raining rainy sunny snow snowing cold hot warm temperature forecast cloudy windy storm humid"),
    ("clock", "time clock hour"),
    ("calendar", "date day today weekday month year"),
    ("humor", "joke jokes funny laugh pun humor"),
    ("naming", "name called call address"),
    ("maker", "made created built wrote programmed developer creator maker designed trained author coded"),
    ("ability", "capable capabilities features purpose skills help assist abilities"),
    ("game", "game games play bored rock paper scissors riddle puzzle guess"),
    ("coin", "coin flip toss heads tails"),
    ("dice", "dice die roll"),
    ("fact", "fact facts trivia learn interesting"),
    ("quote", "quote inspire inspiration motivate motivation wisdom"),
    ("crisis", "die suicide suicidal disappear ending"),
    ("laugh", "haha hahaha lol lmao rofl hehe heh funny"),
    ("repeat", "again another more next once repeat"),
    ("elab", "why really elaborate explain details how"),
    ("recall", "remember repeat recap summarize discuss covered talked"),
    ("progress", "level xp stats streak title score progress dex discovered unlocked"),
];

fn lexicon() -> &'static HashMap<&'static str, Vec<&'static str>> {
    static L: OnceLock<HashMap<&'static str, Vec<&'static str>>> = OnceLock::new();
    L.get_or_init(|| {
        let mut m: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
        for (class, ws) in LEXICON {
            for w in ws.split_whitespace() {
                m.entry(w).or_default().push(class);
            }
        }
        m
    })
}

/// fastText-style hashed features: word, bigram, char 3/4-grams.
pub fn features(text: &str) -> Vec<usize> {
    let ws = norm_words(text);
    let mut f = Vec::with_capacity(ws.len() * 8);
    let mut prev = String::from("^");
    for (i, w) in ws.iter().enumerate() {
        f.push(hash(&format!("w:{w}")));
        f.push(hash(&format!("b:{prev} {w}")));
        if let Some(classes) = lexicon().get(w.as_str()) {
            // "not great" must not look like "great": a negator within two words flips the class
            let negated = ws[i.saturating_sub(2)..i].iter().any(|p| NEGATORS.contains(&p.as_str()));
            for c in classes {
                f.push(hash(&format!("l:{}{c}", if negated { "NOT_" } else { "" })));
            }
        }
        if w != "<num>" {
            let padded: Vec<char> = format!("<{w}>").chars().collect();
            for n in 3..=4 {
                for win in padded.windows(n) {
                    let g: String = win.iter().collect();
                    f.push(CHAR_BASE + hash(&format!("c:{g}")));
                }
            }
        }
        prev = w.clone();
    }
    if let Some(last) = ws.last() {
        f.push(hash(&format!("b:{last} $")));
    }
    f
}

/// Random word drops and typos so the model survives messy input.
pub fn noise(text: &str, rng: &mut Rng) -> String {
    let mut ws: Vec<String> = text.split_whitespace().map(String::from).collect();
    if ws.len() > 1 && rng.f32() < 0.3 {
        let i = rng.below(ws.len());
        ws.remove(i);
    }
    for w in ws.iter_mut() {
        let mut cs: Vec<char> = w.chars().collect();
        if cs.len() > 3 && rng.f32() < 0.3 {
            match rng.below(3) {
                0 => {
                    let i = rng.below(cs.len());
                    cs.remove(i);
                }
                1 => {
                    let i = rng.below(cs.len() - 1);
                    cs.swap(i, i + 1);
                }
                _ => {
                    let i = rng.below(cs.len());
                    let c = cs[i];
                    cs.insert(i, c);
                }
            }
        }
        *w = cs.into_iter().collect();
    }
    ws.join(" ")
}
