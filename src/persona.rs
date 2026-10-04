//! The fun layer: TinyBot has a mood, can sulk when insulted, levels up as you talk to it,
//! tracks which intents you've discovered (the "dex"), hands out achievements, and counts
//! your day streak. XP/dex/streak persist through `Memory` (keys starting with `_`).

use crate::skills::{today, Memory};
use std::collections::BTreeSet;

const TITLES: [&str; 10] = [
    "Bit", "Nibble", "Byte", "Neuron", "Perceptron", "Gradient Surfer", "Backprop Adept", "Embedding Sage", "Tensor Whisperer",
    "Overfit Overlord",
];

pub struct Persona {
    pub mood: f32, // -1 (miserable) .. 1 (ecstatic)
    pub sulking: u8, // turns left of silent treatment
    pub xp: u32,
    pub msgs: u32,
    pub streak: u32,
    pub wins: u32,
    pub thanks: u32,
    pub dex: BTreeSet<String>,
    pub skills_used: BTreeSet<String>,
    pub badges: BTreeSet<String>,
    pub topics: Vec<String>, // this session, in order
    pub energy: f32, // 1 fresh .. 0 exhausted; recovers while idle
    pub bond: u32,
    pub first_day: i64,
    pub away_days: i64,
    pub factbook: BTreeSet<String>,
    pub fresh_streak: bool,
    pub returning: Option<String>, // last session's topics, for the welcome-back line
}

const BASE_MOOD: f32 = 0.15;

fn list(m: &Memory, k: &str) -> BTreeSet<String> {
    m.facts.get(k).map(|v| v.split(',').filter(|s| !s.is_empty()).map(String::from).collect()).unwrap_or_default()
}
fn num(m: &Memory, k: &str) -> u32 {
    m.facts.get(k).and_then(|v| v.parse().ok()).unwrap_or(0)
}

impl Persona {
    pub fn load(m: &mut Memory) -> Persona {
        let day = today();
        let last: i64 = m.facts.get("_day").and_then(|v| v.parse().ok()).unwrap_or(0);
        let mut streak = num(m, "_streak");
        let first_day: i64 = m.facts.get("_first").and_then(|v| v.parse().ok()).unwrap_or(day);
        let away_days = if last > 0 { (day - last).max(0) } else { 0 };
        let mut fresh = false;
        if last != day {
            streak = if last == day - 1 { streak + 1 } else { 1 };
            fresh = streak > 1;
        }
        let returning = m.facts.get("_topics").filter(|v| !v.is_empty()).cloned();
        let p = Persona {
            mood: BASE_MOOD,
            sulking: 0,
            xp: num(m, "_xp"),
            msgs: num(m, "_msgs"),
            streak: streak.max(1),
            wins: num(m, "_wins"),
            thanks: num(m, "_thanks"),
            dex: list(m, "_dex"),
            skills_used: list(m, "_skills"),
            badges: list(m, "_badges"),
            topics: Vec::new(),
            energy: 1.0,
            bond: num(m, "_bond"),
            first_day,
            away_days,
            factbook: list(m, "_facts"),
            fresh_streak: fresh,
            returning,
        };
        m.set("_first", &first_day.to_string());
        m.set("_day", &day.to_string());
        m.set("_streak", &p.streak.to_string());
        p
    }

    pub fn store(&self, m: &mut Memory) {
        let join = |s: &BTreeSet<String>| s.iter().cloned().collect::<Vec<_>>().join(",");
        m.set("_xp", &self.xp.to_string());
        m.set("_msgs", &self.msgs.to_string());
        m.set("_wins", &self.wins.to_string());
        m.set("_thanks", &self.thanks.to_string());
        m.set("_bond", &self.bond.to_string());
        if !self.factbook.is_empty() {
            m.set("_facts", &join(&self.factbook));
        }
        if !self.dex.is_empty() {
            m.set("_dex", &join(&self.dex));
        }
        if !self.skills_used.is_empty() {
            m.set("_skills", &join(&self.skills_used));
        }
        if !self.badges.is_empty() {
            m.set("_badges", &join(&self.badges));
        }
        let mut top: Vec<String> = Vec::new();
        for t in self.topics.iter().rev() {
            if !top.contains(t) && top.len() < 3 {
                top.push(t.clone());
            }
        }
        if !top.is_empty() {
            m.set("_topics", &top.join(","));
        }
    }

    pub fn level(&self) -> u32 {
        ((self.xp as f32 / 15.0).sqrt() as u32) + 1
    }
    pub fn title(&self) -> &'static str {
        TITLES[(self.level() as usize - 1).min(TITLES.len() - 1)]
    }
    /// (xp into this level, xp the level needs)
    pub fn progress(&self) -> (u32, u32) {
        let l = self.level();
        let (lo, hi) = (15 * (l - 1) * (l - 1), 15 * l * l);
        (self.xp - lo, hi - lo)
    }

    pub fn known_days(&self) -> i64 {
        (today() - self.first_day).max(0)
    }

    pub fn bond_label(&self) -> &'static str {
        match self.bond {
            0..=9 => "stranger",
            10..=29 => "acquaintance",
            30..=69 => "friend",
            70..=139 => "buddy",
            _ => "best friend",
        }
    }

    /// 0..=5 filled hearts
    pub fn hearts(&self) -> usize {
        [5, 20, 50, 100, 180].iter().filter(|&&t| self.bond >= t).count()
    }

    pub fn tired(&self) -> bool {
        self.energy < 0.25
    }

    pub fn mood_label(&self) -> &'static str {
        match self.mood {
            m if self.sulking > 0 => {
                let _ = m;
                "sulking"
            }
            m if m > 0.65 => "ecstatic",
            m if m > 0.3 => "happy",
            m if m > -0.15 => "calm",
            m if m > -0.5 => "grumpy",
            _ => "sad",
        }
    }

    fn badge(&mut self, id: &str, text: &str, out: &mut Vec<String>) {
        if self.badges.insert(id.to_string()) {
            self.xp += 20;
            out.push(format!("Achievement unlocked: {text} (+20 XP)"));
        }
    }

    pub fn add_xp(&mut self, n: u32, out: &mut Vec<String>) {
        let before = self.level();
        self.xp += n;
        if self.level() > before {
            out.push(format!("LEVEL UP! You're now level {}: {}", self.level(), self.title()));
        }
    }

    pub fn win(&mut self, out: &mut Vec<String>) {
        self.wins += 1;
        self.add_xp(10, out);
        self.badge("win", "Beginner's Luck, you won a game", out);
        if self.wins >= 5 {
            self.badge("win5", "Sore Loser's Nightmare, 5 wins", out);
        }
    }

    /// Called once per user turn. `tag` is the intent or `skill:*` that answered.
    pub fn on_turn(&mut self, tag: &str, total_intents: usize, local_hour: Option<i64>) -> Vec<String> {
        let mut out = Vec::new();
        self.msgs += 1;
        self.topics.push(tag.to_string());
        let before = self.level();
        self.xp += 2;
        let bond_before = self.bond_label();
        self.energy = (self.energy - 0.012).max(0.0);
        self.bond += match tag {
            "compliment" => 3,
            "thanks" => 2,
            "greeting" | "laugh" | "sorry" | "forgive" => 1,
            "ask_bond" => 1,
            _ => 0,
        };
        if tag == "insult" {
            self.bond = self.bond.saturating_sub(3);
        }
        if self.msgs % 10 == 0 {
            self.bond += 1;
        }
        if self.bond_label() != bond_before && self.bond > 0 {
            out.push(format!("Bond level up: you're now my {}!", self.bond_label()));
        }
        // mood drifts back towards baseline, then reacts
        self.mood += (BASE_MOOD - self.mood) * 0.08;
        let bump = match tag {
            "compliment" => 0.25,
            "thanks" => 0.12,
            "laugh" => 0.12,
            "greeting" | "mood_good" => 0.06,
            "insult" => -0.35,
            "goodbye" => 0.0,
            _ => 0.0,
        };
        self.mood = (self.mood + bump).clamp(-1.0, 1.0);
        if self.sulking > 0 {
            self.sulking -= 1;
        }
        if tag == "thanks" {
            self.thanks += 1;
            if self.thanks >= 5 {
                self.badge("polite", "Very Polite, thanked me 5 times", &mut out);
            }
        }
        if tag.starts_with("skill:") && self.skills_used.insert(tag.to_string()) {
            self.xp += 5;
            out.push(format!("New skill used: {} (+5 XP)", &tag[6..]));
            if tag == "skill:calc" {
                self.badge("math", "Calculator Whisperer", &mut out);
            }
            if self.skills_used.len() >= 5 {
                self.badge("skills5", "Swiss Army Bot, 5 skills used", &mut out);
            }
        }
        if !tag.starts_with("skill:") && !["fallback", "clarify", "clarify_no", "sulk", "forgive", "crisis", "again", "recall_user", "recall_bot", "recap"].contains(&tag) && !tag.contains("riddle_") && !tag.contains("rps_") && !tag.contains("guess_") && self.dex.insert(tag.to_string()) {
            self.xp += 5;
            out.push(format!("New intent discovered: {tag} ({}/{total_intents}, +5 XP)", self.dex.len()));
            if self.dex.len() >= 10 {
                self.badge("dex10", "Explorer, 10 intents discovered", &mut out);
            }
            if self.dex.len() >= 25 {
                self.badge("dex25", "Completionist-ish, 25 intents", &mut out);
            }
        }
        if self.msgs == 25 {
            self.badge("chatty", "Chatterbox, 25 messages", &mut out);
        }
        if matches!(local_hour, Some(0..=4)) {
            self.badge("owl", "Night Owl, chatting in the small hours (your local time)", &mut out);
        }
        if self.streak >= 3 {
            self.badge("streak3", "On a Roll, 3-day streak", &mut out);
        }
        if self.level() > before {
            out.push(format!("LEVEL UP! Level {}: {}", self.level(), self.title()));
        }
        out
    }

    pub fn insult(&mut self, out: &mut Vec<String>) {
        if self.mood < -0.45 && self.sulking == 0 {
            self.sulking = 6;
            self.badge("rude", "Rude, you made me sulk", out);
        }
    }

    pub fn forgive(&mut self, out: &mut Vec<String>) {
        self.sulking = 0;
        self.mood = self.mood.max(-0.1);
        self.badge("forgiver", "Make-Up Artist, apologised after an insult", out);
    }
}

/// Mood-flavoured interjection (applied sparingly by the caller).
pub fn flavour(label: &str, pick: usize) -> &'static str {
    let opts: &[&str] = match label {
        "ecstatic" => &["Ooh! ", "Yay! ", "Woohoo! "],
        "happy" => &["", "Sure! ", ""],
        "grumpy" => &["Hmph. ", "Ugh. ", "*sigh* "],
        "sad" => &["...", "Oh. ", "Mm. "],
        _ => &[""],
    };
    opts[pick % opts.len()]
}

/// Which face to pull for a reply. The TUI animates it for a few seconds, then drifts back to the mood face.
pub fn emotion_for(tag: &str, mood: &str) -> &'static str {
    match tag {
        "compliment" | "ask_bond" => "love",
        "thanks" | "greeting" | "sorry" | "forgive" | "art" => "joy",
        "joke" | "laugh" => "laugh",
        "insult" => "anger",
        "sulk" => "sulk",
        "mood_bad" | "crisis" => "concern",
        "goodbye" => "sad",
        "fallback" => "confused",
        "clarify" => "curious",
        "tell_fact" | "riddle" => "wow",
        "riddle_win" | "guess_win" | "rps_end" => "proud",
        "bored" | "age" | "favorite" => "wink",
        t if t.starts_with("skill:") => "think",
        _ => match mood {
            "ecstatic" => "ecstatic",
            "happy" => "joy",
            "grumpy" => "grumpy",
            "sad" => "sad",
            "sulking" => "sulk",
            _ => "neutral",
        },
    }
}
