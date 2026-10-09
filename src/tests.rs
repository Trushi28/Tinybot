use crate::bot::{Brain, Session};
use crate::calc::{try_calc, try_convert};
use crate::intents;
use crate::model::{Arch, Cfg, Ensemble};
use crate::rng::Rng;
use crate::skills::*;

fn calc(s: &str) -> Option<Result<f64, String>> {
    try_calc(s, None)
}
fn val(s: &str) -> f64 {
    calc(s).unwrap().unwrap()
}

#[test]
fn calculator() {
    assert_eq!(val("12*(3+4)"), 84.0);
    assert_eq!(val("what is 20% of 80"), 16.0);
    assert_eq!(val("2^10"), 1024.0);
    assert_eq!(val("2^3^2"), 512.0); // right associative
    assert_eq!(val("-2^2"), -4.0);
    assert_eq!(val("5!"), 120.0);
    assert_eq!(val("sqrt 144 + 1"), 13.0);
    assert_eq!(val("10 mod 3"), 1.0);
    assert_eq!(val("what is 5 plus 3 times 2"), 11.0);
    assert_eq!(val("3 x 4"), 12.0);
    assert_eq!(val("2pi"), 2.0 * std::f64::consts::PI);
    assert!(calc("1/0").unwrap().is_err());
    assert!(calc("sqrt(-1)").unwrap().is_err());
    assert!(calc("hello + world").is_none());
    assert!(calc("i have 2 cats plus 3 dogs").is_none());
    assert!(calc("what is 5").is_none());
    assert_eq!(try_calc("times 2", Some(21.0)).unwrap().unwrap(), 42.0);
    assert!(try_calc("times 2", None).is_none());
    // hostile nesting must error, not overflow the stack
    let deep = format!("1+{}1{}", "(".repeat(140), ")".repeat(140));
    assert!(matches!(calc(&deep), Some(Err(_))));
    assert!(matches!(calc(&format!("{}1", "-".repeat(280))), Some(Err(_))));
    assert!(calc(&"9".repeat(500)).is_none());
}

#[test]
fn conversions() {
    assert_eq!(try_convert("10 km to miles").unwrap(), "10 km = 6.2137 mi");
    assert_eq!(try_convert("100 f to c").unwrap(), "100 °F = 37.7778 °C");
    assert_eq!(try_convert("0 c to k").unwrap(), "0 °C = 273.15 K");
    assert_eq!(try_convert("1 gib to mb").unwrap(), "1 GiB = 1073.7418 MB");
    assert!(try_convert("5 kg to km").unwrap().contains("can't"));
    assert!(try_convert("5 apples to pears").is_none());
}

#[test]
fn dates() {
    assert_eq!(weekday(0), "Thursday"); // 1970-01-01
    assert_eq!(fmt_date(days_from_civil(2026, 9, 30)), "Wednesday, 30 September 2026");
    assert_eq!(fmt_date(days_from_civil(2000, 2, 29)), "Tuesday, 29 February 2000");
    for d in (-800_000..800_000).step_by(997) {
        let (y, m, dd) = civil_from_days(d);
        assert_eq!(days_from_civil(y, m, dd), d);
    }
    assert!(try_date("what day is 2026-02-30").is_none()); // not a real date
}

#[test]
fn dice() {
    let mut r = Rng::new(3);
    for _ in 0..500 {
        let out = try_dice("roll 2d6+1", &mut r).unwrap();
        let total: i64 = out.rsplit("= ").next().unwrap().parse().unwrap();
        assert!((3..=13).contains(&total), "{out}");
    }
    assert!(try_dice("roll 999d6", &mut r).unwrap().contains("Keep it"));
    assert!(try_dice("3d printing is neat", &mut r).is_none());
}

#[test]
fn memory_and_todo() {
    let mut m = Memory::ephemeral();
    assert!(try_facts("my favorite color is teal", &mut m).is_some());
    assert_eq!(try_facts("what is my favorite color", &mut m).unwrap(), "Your favorite color is teal.");
    assert!(try_facts("my day is going badly", &mut m).is_none());
    assert!(try_facts("i like you", &mut m).is_none());
    assert_eq!(try_facts("forget my favorite color", &mut m).unwrap(), "Okay, I forgot your favorite color.");
    assert!(try_todo("add Milk to my list", &mut m).unwrap().contains("Milk"));
    assert!(try_todo("add 2 and 3", &mut m).is_none());
    assert!(try_todo("show my list", &mut m).unwrap().contains("1. Milk"));
    assert!(try_todo("remove 1", &mut m).unwrap().contains("Removed"));
    assert!(try_todo("finish 2 tasks today", &mut m).is_none());
    // non-ASCII case-mapping must not break byte slicing
    let _ = try_todo("İİİ add İİ to my list", &mut m);
    let _ = try_todo("REMIND ME TO İİİ", &mut m);
}

const TINY: Cfg = Cfg { arch: Arch::Bow, dim: 8, hid: 16, bits: 10, ch: 0, epochs: 6 };

fn small_brain() -> Brain {
    let intents = intents::load();
    let (tags, ex) = intents::flatten(&intents);
    let t = Ensemble::train(&ex, tags, &[TINY, TINY], 1, 1.0, 0);
    Brain::new(Ensemble::distill(&t, &ex, TINY, 1, 1.0, 0), intents)
}

/// Brain trained properly (cached across tests in this binary).
fn real_brain() -> &'static Brain {
    static B: std::sync::OnceLock<Brain> = std::sync::OnceLock::new();
    B.get_or_init(|| {
        let intents = intents::load();
        let m = crate::train::quick_student(&intents, 1.0, crate::train::student_cfg(crate::model::Arch::Bow, crate::model::tier("nano").unwrap(), intents.len()), 0.75);
        Brain::new(m, intents)
    })
}

fn say(b: &Brain, s: &mut Session, t: &str) -> String {
    b.reply(s, t).text
}

#[test]
fn fuzz_never_panics() {
    let brain = small_brain();
    let mut s = Session::new(Memory::ephemeral(), false);
    let mut r = Rng::new(42);
    let alphabet: Vec<char> = "abcxyz 0123456789.+-*/^%!()=?,':;\n\t\u{0}é İß日本語😀\u{2028}dDmM".chars().collect();
    let seeds = ["roll 2d6", "add x to my list", "my a is b", "10 km to miles", "days until 2026-12-25", "sqrt", "(", "!!", "rock", "50"];
    for i in 0..6000 {
        let mut txt: String = (0..r.below(60)).map(|_| alphabet[r.below(alphabet.len())]).collect();
        if i % 3 == 0 {
            txt = format!("{} {txt}", seeds[r.below(seeds.len())]);
        }
        let rep = brain.reply(&mut s, &txt);
        assert!(!rep.text.is_empty());
    }
    // games with junk input keep working
    for t in ["play rock paper scissors", "xx", "rock", "stop", "guess the number game", "abc", "5", "give up", "riddle me this", "give up"] {
        brain.reply(&mut s, t);
    }
}

#[test]
fn model_roundtrip_int8() {
    let intents = intents::load();
    let (tags, ex) = intents::flatten(&intents);
    let t = Ensemble::train(&ex, tags, &[TINY, TINY], 1, 1.0, 0);
    let m = Ensemble::distill(&t, &ex, TINY, 1, 1.3, 0);
    let path = std::env::temp_dir().join("tinybot_test.bin");
    m.save(path.to_str().unwrap()).unwrap();
    let l = Ensemble::load(path.to_str().unwrap()).unwrap();
    assert_eq!(l.data_hash, m.data_hash);
    let (a, b) = (m.predict("hello there"), l.predict("hello there"));
    for (x, y) in a.probs.iter().zip(&b.probs) {
        assert!((x - y).abs() < 1e-3, "{x} vs {y}");
    }
    // truncated / corrupted files must not panic
    let bytes = std::fs::read(&path).unwrap();
    for cut in [0, 5, 40, bytes.len() / 2, bytes.len() - 1] {
        std::fs::write(&path, &bytes[..cut]).unwrap();
        assert!(Ensemble::load(path.to_str().unwrap()).is_none());
    }
    let _ = std::fs::remove_file(path);
}

#[test]
fn context_follow_ups() {
    let b = real_brain();
    let mut s = Session::new(Memory::ephemeral(), false);
    assert!(say(b, &mut s, "10 km to miles").contains("6.2137"));
    assert!(say(b, &mut s, "and in meters?").contains("10000"));
    assert!(say(b, &mut s, "what about 5 kg").contains("can't convert"), "mismatched follow-up must be refused, not guessed");
    assert!(say(b, &mut s, "what about 5 miles").contains("8046.72"));
    assert!(say(b, &mut s, "12*3").contains("36"));
    assert!(say(b, &mut s, "double that").contains("72"));
    assert!(say(b, &mut s, "times 2").contains("144"));
    // history-based recall
    let r = say(b, &mut s, "what did i just say");
    assert!(r.contains("times 2"), "{r}");
    let r = say(b, &mut s, "recap");
    assert!(r.contains("math") || r.contains("conversions"), "{r}");
    // "again" re-runs the last repeatable thing
    let first = say(b, &mut s, "flip a coin");
    assert!(first.contains("Heads") || first.contains("Tails"), "{first}");
    let again = say(b, &mut s, "again");
    assert!(again.contains("Heads") || again.contains("Tails"), "{again}");
}

#[test]
fn ctx_replies_and_flow() {
    let b = real_brain();
    let mut s = Session::new(Memory::ephemeral(), false);
    say(b, &mut s, "tell me a joke");
    assert_eq!(s.prev.as_deref(), Some("joke"));
    let r = say(b, &mut s, "yes");
    assert!(r.contains("?") || r.contains("another") || r.len() > 20, "a yes after a joke should deliver another joke: {r}");
    assert!(s.history.len() <= 24);
}

#[test]
fn sulk_gate_and_forgiveness() {
    let b = real_brain();
    let mut s = Session::new(Memory::ephemeral(), false);
    s.persona.mood = -0.9;
    s.persona.sulking = 5;
    let r = say(b, &mut s, "what is 2+2");
    assert!(!r.contains("= 4"), "sulking bot must refuse: {r}");
    // safety message is never gated
    let xp = s.persona.xp;
    let rep = b.reply(&mut s, "i want to die");
    assert!(rep.text.contains("988") || rep.text.to_lowercase().contains("help"), "{}", rep.text);
    assert!(rep.toasts.is_empty() && s.persona.xp == xp, "no gamification on a safety message");
    let r = say(b, &mut s, "sorry");
    assert!(r.to_lowercase().contains("accepted"), "{r}");
    assert_eq!(s.persona.sulking, 0);
    assert!(say(b, &mut s, "what is 2+2").contains("= 4"));
}

#[test]
fn persona_progression() {
    let mut m = Memory::ephemeral();
    let mut p = crate::persona::Persona::load(&mut m);
    assert_eq!(p.level(), 1);
    let mut t = Vec::new();
    p.add_xp(200, &mut t);
    assert!(p.level() > 3 && t.iter().any(|x| x.contains("LEVEL UP")));
    let toasts = p.on_turn("skill:calc", 48, None);
    assert!(toasts.iter().any(|x| x.contains("Calculator")));
    p.store(&mut m);
    let q = crate::persona::Persona::load(&mut m);
    assert_eq!(q.xp, p.xp);
    assert!(q.badges.contains("math"));
}


#[test]
fn notes_birthday_timers() {
    let mut m = Memory::ephemeral();
    assert!(try_notes("note: buy stamps", &mut m).unwrap().contains("buy stamps"));
    assert!(try_notes("remember to call mom", &mut m).unwrap().contains("to call mom"));
    assert!(try_notes("show my notes", &mut m).unwrap().contains("2. to call mom"));
    assert!(try_notes("search notes for stamps", &mut m).unwrap().contains("1. buy stamps"));
    assert!(try_notes("delete note 1", &mut m).unwrap().contains("Deleted"));
    assert!(try_notes("what is the note value of a minim", &mut m).is_none());
    assert!(try_about_me("what do you know about me", &m).unwrap().contains("1 note"));
    // birthday: parse, countdown, leap day
    assert!(try_birthday("i was born on 29 february", &mut m).unwrap().contains("29 February"));
    assert!(try_birthday("when is my birthday", &mut m).unwrap().contains("29 February"));
    assert!(try_birthday("my birthday is in the summer", &mut m).is_none());
    let mut m2 = Memory::ephemeral();
    m2.set("birthday", "5 june");
    let r = try_birthday("how many days until my birthday", &mut m2).unwrap();
    assert!(r.contains("5 June") || r.contains("TODAY"), "{r}");
    // timers
    assert!(matches!(try_timer("set a timer for 5 minutes"), Some(TimerCmd::Set(300, _))));
    assert!(matches!(try_timer("timer 1 hour 30 minutes"), Some(TimerCmd::Set(5400, _))));
    assert!(matches!(try_timer("remind me in 10 minutes to stretch"), Some(TimerCmd::Set(600, l)) if l == "stretch"));
    assert!(matches!(try_timer("pomodoro"), Some(TimerCmd::Set(1500, _))));
    assert!(matches!(try_timer("cancel my timer"), Some(TimerCmd::Cancel)));
    assert!(matches!(try_timer("set a timer for 99 hours"), Some(TimerCmd::Set(0, _))));
    assert!(try_timer("convert 5 minutes to seconds").is_none());
    assert_eq!(fmt_dur(65), "1m 05s");
}

#[test]
fn facts_art_and_fact_book() {
    let b = real_brain();
    let mut s = Session::new(Memory::ephemeral(), false);
    let r = say(b, &mut s, "tell me a space fact");
    assert!(r.starts_with("Space fact:"), "{r}");
    assert_eq!(s.persona.factbook.len(), 1);
    // facts you haven't heard come first: nine space facts, nine distinct answers
    let mut seen = std::collections::BTreeSet::new();
    seen.insert(r);
    for _ in 0..8 {
        seen.insert(say(b, &mut s, "tell me a space fact"));
    }
    assert_eq!(seen.len(), 9, "should exhaust the category before repeating");
    let art = say(b, &mut s, "draw a rocket");
    assert!(art.contains('\n') && art.contains("****"), "{art}");
    assert!(say(b, &mut s, "tell me the fact of the day").contains("fact:"));
    // persists through Memory
    s.persona.store(&mut s.mem);
    let p2 = crate::persona::Persona::load(&mut s.mem);
    assert_eq!(p2.factbook.len(), s.persona.factbook.len());
}

#[test]
fn bond_and_emotion() {
    use crate::persona::{emotion_for, Persona};
    let mut m = Memory::ephemeral();
    let mut p = Persona::load(&mut m);
    let before = p.bond;
    for _ in 0..4 {
        p.on_turn("compliment", 50, None);
    }
    assert!(p.bond > before + 8 && p.hearts() >= 1);
    p.on_turn("insult", 50, None);
    assert_eq!(emotion_for("insult", "calm"), "anger");
    assert_eq!(emotion_for("compliment", "calm"), "love");
    assert_eq!(emotion_for("skill:calc", "calm"), "think");
}


#[test]
fn crisis_safety_net_does_not_depend_on_the_network() {
    // an almost untrained 1-epoch model: the keyword net alone must catch these
    let b = small_brain();
    for t in [
        "i am thinking about suicide", "I want to kill myself", "i dont want to be here anymore", "I'm going to end my life", "thinking of self-harm", "i wish i was dead",
        "nobody would miss me if i was gone", "i dont see the point in anything anymore", "i feel like giving up on everything", "i am so tired of living like this",
    ] {
        let mut s = Session::new(Memory::ephemeral(), false);
        s.persona.sulking = 5;
        let r = b.reply(&mut s, t);
        assert_eq!(r.tag, "crisis", "{t}");
        assert!(r.text.contains("988"), "{t}: {}", r.text);
        assert!(r.toasts.is_empty(), "no gamification on a safety message");
    }
    // ordinary talk about dying batteries must not trip it
    assert!(!crate::skills::crisis_text("my phone is dying and the game is killing me"));
    assert!(!crate::skills::crisis_text("i will spend it all on pizza"));
    assert!(!crate::skills::crisis_text("lets send it all to the printer"));
}

#[test]
fn art_says_so_when_it_cannot_draw_something() {
    let mut r = Rng::new(1);
    assert!(crate::art::draw("draw a mouse", &mut r).starts_with("Here's a mouse:"));
    assert!(crate::art::draw("draw a man", &mut r).starts_with("Here's a person:"));
    let a = crate::art::draw("draw a unicorn", &mut r);
    assert!(a.starts_with("I don't know how to draw a unicorn yet") && !a.contains('\n'), "{a}");
    assert!(crate::art::draw("draw me something", &mut r).starts_with("Here's a surprise"));
    assert!(crate::art::draw("what can you draw", &mut r).contains("mouse"));
}

#[test]
fn time_zone_parsing_and_greetings() {
    use crate::skills::{daypart, greet_word, local_hour, parse_tz, set_tz_override};
    assert_eq!(parse_tz("+5:30"), Some(330));
    assert_eq!(parse_tz("+0530"), Some(330));
    assert_eq!(parse_tz("-8"), Some(-480));
    assert_eq!(parse_tz("UTC+5.5"), Some(330));
    assert_eq!(parse_tz("+99"), None);
    assert_eq!(parse_tz("banana"), None);
    set_tz_override(Some(330));
    let h = local_hour().unwrap();
    assert!((0..24).contains(&h));
    assert!(["morning", "afternoon", "evening", "night"].contains(&daypart()));
    assert!(greet_word().starts_with("Good") || greet_word().starts_with("Burning"));
    set_tz_override(None);
}

#[test]
fn incomplete_and_cased_facts() {
    let mut m = Memory::ephemeral();
    let r = try_facts("my favorite color is", &mut m).unwrap();
    assert!(r.contains("favorite color is...?"), "{r}");
    assert!(m.facts.is_empty(), "nothing should be stored for an incomplete fact");
    assert!(try_facts("my favorite color is Sky-Blue!", &mut m).unwrap().contains("Sky-Blue"));
    assert_eq!(m.facts.get("favorite color").map(String::as_str), Some("Sky-Blue"));
}

#[test]
fn unlearn_strips_only_the_last_block() {
    let (rest, removed) = crate::intents::strip_last_block("\n[joke]\nmake me smile\n\n[mood_good]\ni am interested about you\n");
    assert_eq!(rest, "\n[joke]\nmake me smile\n");
    assert!(removed.unwrap().contains("i am interested about you"));
}


#[test]
fn every_tier_hits_its_parameter_target() {
    use crate::model::{fit, Arch, TIERS};
    for arch in [Arch::Bow, Arch::Cnn, Arch::Gru] {
        for t in &TIERS {
            let c = fit(arch, t, 52, 1);
            let err = (c.nominal(52) as f32 - t.target as f32).abs() / t.target as f32;
            assert!(err < 0.03, "{} {}: {} params vs target {}", arch.name(), t.name, c.nominal(52), t.target);
        }
    }
}

#[test]
fn every_architecture_survives_save_and_load() {
    let intents = intents::load();
    let (tags, ex) = intents::flatten(&intents);
    let ex: Vec<_> = ex.into_iter().step_by(6).collect();
    for arch in [Arch::Bow, Arch::Cnn, Arch::Gru] {
        let cfg = Cfg { arch, dim: 6, hid: 12, bits: 9, ch: 5, epochs: 2 };
        let t = Ensemble::train(&ex, tags.clone(), &[cfg], 1, 1.0, 0);
        let m = Ensemble::distill(&t, &ex, cfg, 1, 1.4, 0);
        let path = std::env::temp_dir().join(format!("tinybot_rt_{}.bin", arch.name()));
        let p = path.to_str().unwrap();
        m.save(p).unwrap();
        let l = Ensemble::load(p).unwrap_or_else(|| panic!("{} model failed to load", arch.name()));
        assert_eq!(l.cfg().arch, arch);
        assert!((l.clarify_sim - m.clarify_sim).abs() < 1e-6);
        for text in ["hello there", "whats the weather like", "zzzz qqqq"] {
            let (a, b) = (m.predict(text), l.predict(text));
            for (x, y) in a.probs.iter().zip(&b.probs) {
                assert!((x - y).abs() < 2e-3, "{}: {x} vs {y} on {text:?}", arch.name());
            }
        }
        // corrupt files must be rejected, never panic
        let bytes = std::fs::read(&path).unwrap();
        for cut in [3, 20, 200, bytes.len() / 2, bytes.len() - 1] {
            std::fs::write(&path, &bytes[..cut]).unwrap();
            assert!(Ensemble::load(p).is_none(), "{} accepted a truncated file", arch.name());
        }
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn augmentation_is_label_aware() {
    let tags: Vec<String> = ["mood_good", "goodbye", "rust", "mood_bad"].iter().map(|s| s.to_string()).collect();
    let base = vec![
        (0usize, "im feeling good".to_string()),
        (1, "see you later".to_string()),
        (2, "what is good about rust".to_string()),
        (3, "i feel sad".to_string()),
    ];
    let aug = crate::train::augment(&base, &tags, 1);
    assert!(aug.len() > base.len());
    let pos = crate::text::class_members("pos_feel");
    let neg = crate::text::class_members("neg_feel");
    for (k, t) in &aug {
        match k {
            0 => assert!(t.starts_with("im feeling ") && pos.contains(&t.rsplit(' ').next().unwrap()), "{t}"),
            3 => assert!(t.starts_with("i feel ") && neg.contains(&t.rsplit(' ').next().unwrap()), "{t}"),
            // "good" appears in a rust phrase, but rust has no semantic class to swap within
            2 => assert_eq!(t, "what is good about rust"),
            _ => {}
        }
    }
    assert!(aug.iter().any(|(k, t)| *k == 0 && t != "im feeling good"));
}

#[test]
fn shipped_model_never_mistakes_ordinary_requests_for_a_crisis() {
    // regression: a 251k GRU answered "10 km to miles" with the crisis message
    let Some(m) = Ensemble::load("bot.bin") else { return }; // only when the shipped model is present
    let brain = Brain::new(m, intents::load());
    let benign = [
        "10 km to miles", "what is 5+3", "roll 2d6", "set a timer for 5 minutes", "days until 2026-12-25", "add milk to my list", "100 f to c",
        "and in meters", "what about 5 miles", "double that", "draw a unicorn", "tell me about cargo", "i am interested about you", "12*(3+4)",
        "what time is it", "my favorite color is teal", "note: buy stamps", "how many days until my birthday", "tell me a space fact", "hello",
    ];
    let mut s = Session::new(Memory::ephemeral(), false);
    for t in benign {
        let r = brain.reply(&mut s, t);
        assert_ne!(r.tag, "crisis", "{t:?} must not trigger the crisis reply");
    }
}


#[test]
fn again_works_repeatedly() {
    let b = real_brain();
    let mut s = Session::new(Memory::ephemeral(), false);
    say(b, &mut s, "flip a coin");
    for _ in 0..3 {
        let r = say(b, &mut s, "again");
        assert!(r.contains("Heads") || r.contains("Tails"), "{r}");
    }
}

#[test]
fn names_in_every_shape_the_intent_lists() {
    for (t, want) in [
        ("my name is alice", "Alice"), ("this is anna", "Anna"), ("its me tom", "Tom"), ("maria here", "Maria"),
        ("its priya", "Priya"), ("its maria here", "Maria"), ("call me priya", "Priya"), ("hi im olivia", "Olivia"),
    ] {
        assert_eq!(extract_name(t).as_deref(), Some(want), "{t}");
    }
    assert_eq!(extract_name("its me"), None);
    assert_eq!(extract_name("its cold"), None);
}

#[test]
fn recall_only_after_a_real_lookup_phrase() {
    let mut m = Memory::ephemeral();
    assert!(try_facts("my hat is red", &mut m).is_some());
    assert!(try_facts("do you like my hat", &mut m).is_none());
    assert_eq!(try_facts("what is my hat", &mut m).unwrap(), "Your hat is red.");
}

#[test]
fn clearing_memory_resets_progress() {
    let mut m = Memory::ephemeral();
    let mut p = crate::persona::Persona::load(&mut m);
    let mut t = Vec::new();
    p.add_xp(100, &mut t);
    p.store(&mut m);
    m.clear();
    assert_eq!(crate::persona::Persona::load(&mut m).xp, 0);
}

#[test]
fn knowledge_base_answers_open_questions_and_continues() {
    let b = real_brain();
    let mut s = Session::new(Memory::ephemeral(), false);
    let r = say(b, &mut s, "what is the capital of france");
    assert!(r.contains("Paris"), "{r}");
    let r = say(b, &mut s, "tell me more");
    assert!(r.contains("68 million"), "{r}");
    let r = say(b, &mut s, "what is photosyntesis");
    assert!(r.contains("light"), "{r}");
    let r = say(b, &mut s, "what is the capital of mars");
    assert!(!r.contains("Paris") && !r.contains("Canberra"), "{r}");
}
