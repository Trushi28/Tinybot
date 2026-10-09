//! tinybot: a small intent-routing chatbot in pure Rust, zero dependencies.
//!
//! Engine   : 5-net teacher ensemble distilled into ONE compact int8 student (68k to 4M params)
//! Knowledge: BM25 retrieval over kb.txt, kb_user.txt (/know) and facts.txt for open questions
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
mod kb;
mod model;
mod persona;
mod rng;
mod seqnet;
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

const MEMORY_PATH: &str = "memory.txt";
const FLOW_PATH: &str = "flow.txt";

fn load_or_train(o: &train::Opts) -> Brain {
    let intents = intents::load();
    let (tags, ex) = intents::flatten(&intents);
    let scfg = train::student_cfg(o.arch, o.tier, intents.len());
    let used = if o.aug { train::augment(&ex, &tags, train::SEED) } else { ex.clone() };
    let want = model::data_hash(&tags, &used, model::cfg_salt(&scfg));
    let path = o.path();
    let m = match Ensemble::load(&path) {
        Some(m) if m.data_hash == want && m.tags == tags => m,
        _ => {
            eprintln!("({path} missing or out of date with the intents, training first)");
            train::train_quiet(o)
        }
    };
    Brain::new(m, intents)
}

fn new_app(o: &train::Opts) -> App {
    let brain = load_or_train(o);
    let mut s = Session::new(Memory::load(MEMORY_PATH), true);
    s.load_flow(FLOW_PATH);
    let mut app = App::new(brain, s);
    app.model_path = o.path();
    app
}

fn cmd_plain(o: &train::Opts) {
    let mut app = new_app(o);
    println!("TinyBot ready ({}, {} params). /help for commands.", train::model_name(&app.brain.model), train::params_label(&app.brain.model));
    let stdin = io::stdin();
    loop {
        print!("you> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let line: String = line.chars().filter(|c| !c.is_control()).collect::<String>().trim().to_string();
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

fn cmd_tui(o: &train::Opts) {
    let mut app = new_app(o);
    tui::run(&mut app);
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let plain = args.iter().any(|a| a == "--plain");
    let o = train::Opts::parse(&args);
    let cmd = args.iter().find(|a| !a.starts_with("--") && !matches!(a.as_str(), "nano" | "small" | "base" | "large" | "xl" | "max" | "bow" | "cnn" | "gru" | "het") && a.parse::<usize>().is_err() && !a.contains(','));
    match cmd.map(String::as_str) {
        Some("train") if args.iter().any(|a| a == "--all") => train::cmd_train_all(&o),
        Some("train") => {
            train::cmd_train(&o);
        }
        Some("eval") => train::cmd_eval(&o),
        Some("sweep") => train::cmd_sweep(&o),
        Some("models") => train::cmd_models(intents::load().len()),
        Some("speed") => train::cmd_speed(intents::load().len()),
        Some("ctx") => train::cmd_ctx_eval(),
        Some("bench") => {
            let b = load_or_train(&o);
            let (us, qps) = train::bench(&b.model);
            println!("{}: {us:.1} us/query, {qps:.0} queries/s, {} params, {:.0} KB", train::model_name(&b.model), train::params_label(&b.model), train::model_kb(&b.model));
        }
        Some("chat") | None => {
            if plain || !io::stdout().is_terminal() || !io::stdin().is_terminal() {
                cmd_plain(&o)
            } else {
                cmd_tui(&o)
            }
        }
        _ => {
            eprintln!("usage: tinybot [chat [--plain] | train | eval | sweep | models | speed | ctx | bench]\n       options: --size nano|small|base|large|xl|max  --arch bow|cnn|gru  --teacher bow|het  --no-aug  --folds 0,3\n       train --all [--archs bow,gru] [--tiers nano,base]   builds many sizes from one teacher into models/");
            process::exit(2);
        }
    }
}
