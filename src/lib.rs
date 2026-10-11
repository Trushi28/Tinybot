//! tinybot: a small intent-routing chatbot in pure Rust, zero dependencies.
//!
//! Engine   : 5-net teacher ensemble distilled into ONE compact int8 student (68k to 4M params)
//! Knowledge: BM25 retrieval over kb.txt, kb_user.txt (/know) and facts.txt for open questions
//! Context  : 24-turn memory, `>@prev` replies, learned intent-flow prior, follow-up resolution
//!            ("and in meters?", "double that", "again", "what did I just say")
//! Skills   : calculator, units, dates, dice, memory, todo list (all from scratch)
//! Fun      : mood + sulking, XP/levels/titles, intent dex, badges, streaks, games
//! UI       : animated terminal UI with a live view into the network (`--plain` for a bare REPL)

pub mod art;
pub mod bot;
pub mod calc;
pub mod clock;
pub mod cmds;
pub mod facts;
pub mod intents;
pub mod kb;
pub mod model;
pub mod persona;
pub mod rng;
pub mod seqnet;
pub mod skills;
pub mod text;
#[cfg(test)]
mod tests;
pub mod train;
#[cfg(not(target_arch = "wasm32"))]
pub mod tui;
pub mod web;
