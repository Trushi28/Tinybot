//! tinybot: a small intent-routing chatbot in pure Rust, zero dependencies.
//!
//! Engine   : 5-net teacher ensemble distilled into ONE compact int8 student (~64k params)
//! Context  : 24-turn memory, `>@prev` replies, learned intent-flow prior, follow-up resolution
//!            ("and in meters?", "double that", "again", "what did I just say")
//! Skills   : calculator, units, dates, dice, memory, todo list (all from scratch)
//! Fun      : mood + sulking, XP/levels/titles, intent dex, badges, streaks, games
//! UI       : animated terminal UI with a live view into the network (`--plain` for a bare REPL)

mod art;
mod bot;
mod calc;
mod cmds;
mod facts;
mod intents;
mod model;
mod persona;
mod rng;
mod skills;
mod text;
#[cfg(test)]
mod tests;
mod train;
mod tui;

use bot::{Brain, Session};
use cmds::App;
use model::Ensemble;
use skills::Memory;
use std::io::{self, BufRead, IsTerminal, Write};
use std::time::Instant;
use std::{env, process};
use train::MODEL_PATH;

const MEMORY_PATH: &str = "memory.txt";
const FLOW_PATH: &str = "flow.txt";

fn load_or_train() -> Brain {
    let intents = intents::load();
    let (tags, ex) = intents::flatten(&intents);
    let want = model::data_hash(&tags, &ex, 0);
    let m = match Ensemble::load(MODEL_PATH) {
        Some(m) if m.data_hash == want && m.tags == tags => m,
        _ => {
            eprintln!("(model missing or out of date with intents, training first)");
            train::train_quiet()
        }
    };
    Brain::new(m, intents)
}

fn new_app() -> App {
    let brain = load_or_train();
    let mut s = Session::new(Memory::load(MEMORY_PATH), true);
    s.load_flow(FLOW_PATH);
    App::new(brain, s)
}

fn cmd_plain() {
    let mut app = new_app();
    println!("TinyBot ready ({} params). /help for commands.", train::params_label(&app.brain.model));
    let stdin = io::stdin();
    loop {
        print!("you> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('/') {
            let r = cmds::run(&mut app, &line);
            for l in r.lines {
                println!("  {l}");
            }
            if let Some(m) = app.wait_retrain() {
                println!("  ({m})");
            }
            if r.quit {
                break;
            }
            continue;
        }
        let t = Instant::now();
        let r = app.brain.reply(&mut app.s, &line);
        println!("bot> {}", r.text);
        for toast in &r.toasts {
            println!("  * {toast}");
        }
        if app.s.debug {
            println!("  [{} {}  {:.2?}]", r.tag, r.debug, t.elapsed());
        }
        if let Some((k, phrase)) = r.learn {
            let _ = intents::append_learned(&app.brain.model.tags[k].clone(), &phrase, None);
            app.start_retrain();
            if let Some(m) = app.wait_retrain() {
                println!("  (learned {phrase:?}: {m})");
            }
        }
        app.s.persona.store(&mut app.s.mem);
        app.s.mem.save();
        app.s.save_flow(FLOW_PATH);
        if r.done {
            break;
        }
    }
}

fn cmd_tui() {
    let mut app = new_app();
    tui::run(&mut app);
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let plain = args.iter().any(|a| a == "--plain");
    match args.iter().find(|a| !a.starts_with("--")).map(String::as_str) {
        Some("train") => {
            train::cmd_train();
        }
        Some("eval") => train::cmd_eval(false),
        Some("sweep") => train::cmd_eval(true),
        Some("ctx") => train::cmd_ctx_eval(),
        Some("bench") => {
            let b = load_or_train();
            let (us, qps) = train::bench(&b.model);
            println!("{us:.1} us/query, {qps:.0} queries/s, {} params, {:.0} KB", train::params_label(&b.model), train::model_kb(&b.model));
        }
        Some("chat") | None => {
            if plain || !io::stdout().is_terminal() || !io::stdin().is_terminal() {
                cmd_plain()
            } else {
                cmd_tui()
            }
        }
        _ => {
            eprintln!("usage: tinybot [chat [--plain] | train | eval | sweep | ctx | bench]");
            process::exit(2);
        }
    }
}
