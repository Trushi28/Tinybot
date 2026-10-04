//! A tiny gallery of ASCII art (plain ASCII so column widths are never ambiguous).
//! Ask for something that isn't here and it says so instead of drawing something random.

use crate::rng::Rng;

const CAT: &str = r"
 /\_/\
( o.o )
 > ^ <
";
const HEART: &str = r"
  .:::.   .:::.
 :::::::.:::::::
 :::::::::::::::
  ':::::::::::'
    ':::::::'
      ':::'
        '
";
const ROCKET: &str = r"
     /\
    /  \
   | () |
   |    |
  /|    |\
 / |    | \
/__|____|__\
    /__\
    ****
";
const CRAB: &str = r"
   _~^~^~_
\) /  o o  \ (/
  '_   -   _'
  / '-----' \
";
const ROBOT: &str = r"
   [^_^]
  /|___|\
   |   |
   d   b
";
const TREE: &str = r"
     &&& &&
   && &&& &&&
  &&& &&&& &&
     \\ //
      ||||
      ||||
";
const MOUSE: &str = r"
    ___
 __/o o\__
(__  v  __)~~~
   \_^_/
";
const BUNNY: &str = r#"
 (\_/)
 (='.'=)
 (")_(")
"#;
const PERSON: &str = r"
   O
  /|\
  / \
";
const DOG: &str = r"
  __      _
o'')}____//
 `_/      )
 (_(_/-(_/
";
const FISH: &str = r"
  o
   O    ><(((o>
  o
";
const HOUSE: &str = r"
    /\
   /  \
  /____\
  | [] |
  |_||_|
";
const FLOWER: &str = r"
   _(_)_
  (_)@(_)
    (_)\
      |
";
const STAR: &str = r"
     *
    ***
 *********
   *****
  **   **
";
const OWL: &str = r"
  ,___,
  (O,O)
  /)__)
  -'-'-
";
const SUN: &str = r"
   \   |   /
     .-.
 -- (   ) --
     `-'
   /   |   \
";

/// (name, keywords, picture)
static GALLERY: &[(&str, &[&str], &str)] = &[
    ("cat", &["cat", "kitten", "kitty"], CAT),
    ("heart", &["heart", "love"], HEART),
    ("rocket", &["rocket", "space", "ship"], ROCKET),
    ("crab", &["crab", "ferris", "rust"], CRAB),
    ("robot", &["robot", "bot", "tinybot"], ROBOT),
    ("tree", &["tree", "forest", "plant"], TREE),
    ("mouse", &["mouse", "mice", "rat"], MOUSE),
    ("bunny", &["bunny", "rabbit"], BUNNY),
    ("person", &["man", "woman", "person", "human", "guy", "girl", "boy", "stickman", "people"], PERSON),
    ("dog", &["dog", "puppy", "doggo"], DOG),
    ("fish", &["fish"], FISH),
    ("house", &["house", "home", "building"], HOUSE),
    ("flower", &["flower", "rose", "daisy"], FLOWER),
    ("star", &["star", "stars"], STAR),
    ("owl", &["owl", "bird"], OWL),
    ("sun", &["sun", "sunny"], SUN),
];

pub fn names() -> String {
    GALLERY.iter().map(|g| g.0).collect::<Vec<_>>().join(", ")
}

/// The thing being asked for: the first meaningful word after "draw"/"sketch".
fn subject(words: &[&str]) -> Option<String> {
    const SKIP: [&str; 20] = [
        "me", "a", "an", "the", "some", "my", "for", "something", "anything", "picture", "pic", "ascii", "art", "please", "can", "you", "drawing", "of",
        "surprise", "random",
    ];
    let at = words.iter().position(|w| matches!(*w, "draw" | "sketch" | "drawing"))?;
    words[at + 1..].iter().find(|w| !SKIP.contains(w)).map(|w| w.to_string())
}

pub fn draw(text: &str, rng: &mut Rng) -> String {
    let low = text.to_lowercase();
    let ws: Vec<&str> = low.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    if low.contains("what can you draw") || low.contains("what do you draw") || low.contains("what can you sketch") || ws.contains(&"gallery") {
        return format!("I can draw: {}. Try \"draw a dog\".", names());
    }
    if let Some(g) = GALLERY.iter().find(|(_, kws, _)| kws.iter().any(|k| ws.contains(k))) {
        return format!("Here's a {}:\n{}", g.0, g.2.trim_matches('\n'));
    }
    if let Some(sub) = subject(&ws) {
        return format!("I don't know how to draw a {sub} yet. I can draw: {}.", names());
    }
    let g = &GALLERY[rng.below(GALLERY.len())];
    format!("Here's a surprise, a {}:\n{}", g.0, g.2.trim_matches('\n'))
}
