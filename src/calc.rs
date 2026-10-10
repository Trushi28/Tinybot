//! Calculator + unit conversion. No panics on hostile input: length and nesting
//! depth are capped, and every numeric edge case returns an Err instead.

const MAX_LEN: usize = 300;
const MAX_DEPTH: u32 = 40;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Num(f64),
    Word(String),
    Sym(char),
}

pub fn lex(s: &str) -> Option<Vec<Tok>> {
    if s.chars().count() > MAX_LEN {
        return None;
    }
    let cs: Vec<char> = s.to_lowercase().chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || (c == '.' && cs.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let st = i;
            let mut dot = false;
            while i < cs.len() {
                if cs[i].is_ascii_digit() {
                    i += 1;
                } else if cs[i] == '.' && !dot && cs.get(i + 1).is_some_and(|d| d.is_ascii_digit()) {
                    dot = true;
                    i += 1;
                } else {
                    break;
                }
            }
            let n: String = cs[st..i].iter().collect();
            out.push(Tok::Num(n.parse().ok()?));
        } else if c.is_alphabetic() {
            let st = i;
            while i < cs.len() && (cs[i].is_alphanumeric() || cs[i] == '\'') {
                i += 1;
            }
            out.push(Tok::Word(cs[st..i].iter().filter(|&&c| c != '\'').collect()));
        } else {
            out.push(Tok::Sym(c));
            i += 1;
        }
    }
    Some(out)
}

pub fn fmt_num(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    if v.fract() == 0.0 && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    if v.abs() >= 1e15 || v.abs() < 1e-6 {
        return format!("{v:.6e}");
    }
    fmt_round(v, 10)
}

pub fn fmt_round(v: f64, dp: usize) -> String {
    let s = format!("{v:.dp$}");
    let s = if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
    if s == "-0" { "0".into() } else { s }
}

// ---------------- calculator ----------------

const FILLER: [&str; 22] = [
    "what", "is", "whats", "calculate", "calc", "compute", "please", "tell", "me", "the", "result", "how", "much",
    "does", "do", "equal", "equals", "solve", "evaluate", "answer", "of", "s",
];
const TRIGGERS: [&str; 5] = ["calculate", "calc", "compute", "evaluate", "solve"];
const FUNCS: [&str; 17] = [
    "sqrt", "cbrt", "sin", "cos", "tan", "asin", "acos", "atan", "ln", "log", "log2", "log10", "exp", "abs", "floor",
    "ceil", "round",
];
const CONSTS: [&str; 4] = ["pi", "e", "tau", "ans"];
const CONTINUE: [&str; 12] = [
    "times", "plus", "minus", "divided by", "multiplied by", "over", "mod", "modulo", "to the power", "squared",
    "cubed", "then",
];

/// None = not a math request. Some(Err) = math request that failed.
pub fn try_calc(text: &str, ans: Option<f64>) -> Option<Result<f64, String>> {
    let low = text.to_lowercase();
    let mut s = format!(" {} ", low.replace(['?', '=', '\''], " "));
    for (a, b) in [
        (" to the power of ", " ^ "),
        (" raised to the power of ", " ^ "),
        (" raised to ", " ^ "),
        (" multiplied by ", " * "),
        (" divided by ", " / "),
        (" square root of ", " sqrt "),
        (" percent of ", " pct "),
        ("% of ", " pct "),
        (" times ", " * "),
        (" plus ", " + "),
        (" minus ", " - "),
        (" over ", " / "),
        (" modulo ", " % "),
        (" mod ", " % "),
        (" squared ", " ^ 2 "),
        (" cubed ", " ^ 3 "),
        (" factorial ", " ! "),
    ] {
        s = s.replace(a, b);
    }
    let explicit = TRIGGERS.iter().any(|t| low.split_whitespace().any(|w| w == *t));
    let trimmed = low.trim();
    let continues = ans.is_some()
        && (CONTINUE.iter().any(|c| trimmed.starts_with(c))
            || trimmed.starts_with(['*', '/', '^', '%']));

    let mut toks: Vec<Tok> = lex(&s)?;
    toks.retain(|t| !matches!(t, Tok::Word(w) if FILLER.contains(&w.as_str())));
    // "3 x 4" -> multiplication, but only between numbers
    for i in 1..toks.len().saturating_sub(1) {
        if toks[i] == Tok::Word("x".into())
            && matches!(toks[i - 1], Tok::Num(_) | Tok::Sym(')'))
            && matches!(toks[i + 1], Tok::Num(_) | Tok::Sym('('))
        {
            toks[i] = Tok::Sym('*');
        }
    }
    for t in toks.iter_mut() {
        if *t == Tok::Word("pct".into()) {
            *t = Tok::Sym('p');
        }
        if *t == Tok::Word("then".into()) {
            *t = Tok::Word("".into());
        }
    }
    toks.retain(|t| *t != Tok::Word("".into()));
    // A lone "-5", or a tight "555-1234" / "1990-2000" with no other operator, is a negative number,
    // a phone number or a range, not a subtraction. "5 - 3", "5 minus 3" and "what is 10-3" still calculate.
    if !explicit && !continues {
        let only_minus = toks.iter().all(|t| matches!(t, Tok::Num(_) | Tok::Sym('-'))) && toks.iter().filter(|t| **t == Tok::Sym('-')).count() == 1;
        if only_minus {
            let lone_sign = toks.len() == 2 && toks[0] == Tok::Sym('-');
            let cs: Vec<char> = low.chars().collect();
            let tight = cs.windows(3).any(|w| w[0].is_ascii_digit() && w[1] == '-' && w[2].is_ascii_digit());
            let asked = low.split_whitespace().next().is_some_and(|w| matches!(w, "what" | "whats" | "how" | "what's"));
            if lone_sign || (tight && !asked) {
                return None;
            }
        }
    }
    if continues {
        // strip a leading "then"-style word already handled; prefix ans
        toks.insert(0, Tok::Word("ans".into()));
    }
    // every token must be something the parser understands
    for t in &toks {
        let ok = match t {
            Tok::Num(_) => true,
            Tok::Sym(c) => "+-*/^%()!p".contains(*c),
            Tok::Word(w) => FUNCS.contains(&w.as_str()) || CONSTS.contains(&w.as_str()),
        };
        if !ok {
            return None;
        }
    }
    let has_operand = toks.iter().any(|t| matches!(t, Tok::Num(_)) || matches!(t, Tok::Word(w) if CONSTS.contains(&w.as_str())));
    let has_op = toks
        .iter()
        .any(|t| matches!(t, Tok::Sym(c) if "+-*/^%!p".contains(*c)) || matches!(t, Tok::Word(w) if FUNCS.contains(&w.as_str())));
    let implicit = toks.iter().any(|t| matches!(t, Tok::Num(_))) && toks.iter().any(|t| matches!(t, Tok::Word(w) if w == "pi" || w == "tau"));
    if !has_operand || (!has_op && !explicit && !implicit) {
        return None;
    }
    if toks.iter().any(|t| *t == Tok::Word("ans".into())) && ans.is_none() {
        return Some(Err("I don't have a previous result to use yet.".into()));
    }
    let mut p = Parser { t: &toks, i: 0, depth: 0, ans: ans.unwrap_or(0.0) };
    let res = p.expr().and_then(|v| {
        if p.i < toks.len() {
            Err("parse".to_string())
        } else {
            Ok(v)
        }
    });
    match res {
        Ok(v) if v.is_finite() => Some(Ok(v)),
        Ok(_) => Some(Err("That result is out of range.".into())),
        Err(e) if e == "parse" => {
            if explicit {
                Some(Err("I couldn't parse that expression.".into()))
            } else {
                None
            }
        }
        Err(e) => Some(Err(e)),
    }
}

struct Parser<'a> {
    t: &'a [Tok],
    i: usize,
    depth: u32,
    ans: f64,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&'a Tok> {
        self.t.get(self.i)
    }
    fn sym(&self, c: char) -> bool {
        matches!(self.peek(), Some(Tok::Sym(x)) if *x == c)
    }
    fn enter(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            Err("That expression is nested too deeply.".into())
        } else {
            Ok(())
        }
    }

    fn expr(&mut self) -> Result<f64, String> {
        let mut v = self.term()?;
        loop {
            if self.sym('+') {
                self.i += 1;
                v += self.term()?;
            } else if self.sym('-') {
                self.i += 1;
                v -= self.term()?;
            } else {
                return Ok(v);
            }
        }
    }

    fn term(&mut self) -> Result<f64, String> {
        let mut v = self.unary()?;
        loop {
            if self.sym('*') {
                self.i += 1;
                v *= self.unary()?;
            } else if self.sym('/') {
                self.i += 1;
                let d = self.unary()?;
                if d == 0.0 {
                    return Err("Can't divide by zero.".into());
                }
                v /= d;
            } else if self.sym('%') {
                self.i += 1;
                let d = self.unary()?;
                if d == 0.0 {
                    return Err("Can't take a remainder by zero.".into());
                }
                v %= d;
            } else if self.sym('p') {
                self.i += 1;
                v = v / 100.0 * self.unary()?;
            } else if matches!(self.peek(), Some(Tok::Sym('('))) || matches!(self.peek(), Some(Tok::Word(_))) {
                v *= self.unary()?; // implicit multiplication: 2pi, 3(4+1)
            } else {
                return Ok(v);
            }
        }
    }

    fn unary(&mut self) -> Result<f64, String> {
        self.enter()?;
        let r = if self.sym('-') {
            self.i += 1;
            self.unary().map(|v| -v)
        } else if self.sym('+') {
            self.i += 1;
            self.unary()
        } else {
            self.power()
        };
        self.depth -= 1;
        r
    }

    fn power(&mut self) -> Result<f64, String> {
        let b = self.postfix()?;
        if self.sym('^') {
            self.i += 1;
            let e = self.unary()?;
            return Ok(b.powf(e));
        }
        Ok(b)
    }

    fn postfix(&mut self) -> Result<f64, String> {
        let mut v = self.atom()?;
        while self.sym('!') {
            self.i += 1;
            if v < 0.0 || v.fract() != 0.0 || v > 170.0 {
                return Err("Factorial needs a whole number from 0 to 170.".into());
            }
            v = (1..=v as u64).fold(1.0, |a, k| a * k as f64);
        }
        Ok(v)
    }

    fn atom(&mut self) -> Result<f64, String> {
        self.enter()?;
        let r = match self.peek().cloned() {
            Some(Tok::Num(n)) => {
                self.i += 1;
                Ok(n)
            }
            Some(Tok::Sym('(')) => {
                self.i += 1;
                let v = self.expr();
                if v.is_ok() {
                    if self.sym(')') {
                        self.i += 1;
                    } else {
                        self.depth -= 1;
                        return Err("parse".into());
                    }
                }
                v
            }
            Some(Tok::Word(w)) => {
                self.i += 1;
                match w.as_str() {
                    "pi" => Ok(std::f64::consts::PI),
                    "e" => Ok(std::f64::consts::E),
                    "tau" => Ok(std::f64::consts::TAU),
                    "ans" => Ok(self.ans),
                    f => {
                        let arg = if self.sym('(') { self.atom() } else { self.unary_tight() };
                        arg.and_then(|a| apply(f, a))
                    }
                }
            }
            _ => Err("parse".into()),
        };
        self.depth -= 1;
        r
    }

    /// operand of a function written without parentheses: "sqrt 16", "sin -1"
    fn unary_tight(&mut self) -> Result<f64, String> {
        if self.sym('-') {
            self.i += 1;
            return self.unary_tight().map(|v| -v);
        }
        self.postfix()
    }
}

fn apply(f: &str, a: f64) -> Result<f64, String> {
    let dom = || Err(format!("{f} isn't defined for {}.", fmt_num(a)));
    Ok(match f {
        "sqrt" if a < 0.0 => return dom(),
        "sqrt" => a.sqrt(),
        "cbrt" => a.cbrt(),
        "sin" => a.sin(),
        "cos" => a.cos(),
        "tan" => a.tan(),
        "asin" | "acos" if !(-1.0..=1.0).contains(&a) => return dom(),
        "asin" => a.asin(),
        "acos" => a.acos(),
        "atan" => a.atan(),
        "ln" | "log" | "log2" | "log10" if a <= 0.0 => return dom(),
        "ln" => a.ln(),
        "log" | "log10" => a.log10(),
        "log2" => a.log2(),
        "exp" => a.exp(),
        "abs" => a.abs(),
        "floor" => a.floor(),
        "ceil" => a.ceil(),
        "round" => a.round(),
        _ => return Err("parse".into()),
    })
}

// ---------------- unit conversion ----------------

#[derive(PartialEq, Clone, Copy)]
enum Cat {
    Length,
    Mass,
    Volume,
    Speed,
    Time,
    Data,
    Temp,
}

struct Unit {
    names: &'static [&'static str],
    show: &'static str,
    cat: Cat,
    to_base: f64,
}

const fn u(names: &'static [&'static str], show: &'static str, cat: Cat, to_base: f64) -> Unit {
    Unit { names, show, cat, to_base }
}

static UNITS: &[Unit] = &[
    u(&["m", "meter", "meters", "metre", "metres"], "m", Cat::Length, 1.0),
    u(&["km", "kilometer", "kilometers", "kilometre", "kilometres"], "km", Cat::Length, 1000.0),
    u(&["cm", "centimeter", "centimeters", "centimetre", "centimetres"], "cm", Cat::Length, 0.01),
    u(&["mm", "millimeter", "millimeters", "millimetre", "millimetres"], "mm", Cat::Length, 0.001),
    u(&["mi", "mile", "miles"], "mi", Cat::Length, 1609.344),
    u(&["ft", "foot", "feet"], "ft", Cat::Length, 0.3048),
    u(&["in", "inch", "inches"], "in", Cat::Length, 0.0254),
    u(&["yd", "yard", "yards"], "yd", Cat::Length, 0.9144),
    u(&["kg", "kilogram", "kilograms", "kilo", "kilos"], "kg", Cat::Mass, 1.0),
    u(&["g", "gram", "grams"], "g", Cat::Mass, 0.001),
    u(&["mg", "milligram", "milligrams"], "mg", Cat::Mass, 1e-6),
    u(&["lb", "lbs", "pound", "pounds"], "lb", Cat::Mass, 0.45359237),
    u(&["oz", "ounce", "ounces"], "oz", Cat::Mass, 0.028349523125),
    u(&["l", "liter", "liters", "litre", "litres"], "l", Cat::Volume, 1.0),
    u(&["ml", "milliliter", "milliliters", "millilitre", "millilitres"], "ml", Cat::Volume, 0.001),
    u(&["gal", "gallon", "gallons"], "gal (US)", Cat::Volume, 3.785411784),
    u(&["qt", "quart", "quarts"], "qt (US)", Cat::Volume, 0.946352946),
    u(&["cup", "cups"], "cup (US)", Cat::Volume, 0.2365882365),
    u(&["tbsp", "tablespoon", "tablespoons"], "tbsp", Cat::Volume, 0.01478676478),
    u(&["tsp", "teaspoon", "teaspoons"], "tsp", Cat::Volume, 0.00492892159),
    u(&["kmh", "kph", "kmph"], "km/h", Cat::Speed, 1.0 / 3.6),
    u(&["mph"], "mph", Cat::Speed, 0.44704),
    u(&["knot", "knots", "kt", "kts"], "kn", Cat::Speed, 0.514444),
    u(&["ms", "millisecond", "milliseconds"], "ms", Cat::Time, 0.001),
    u(&["s", "sec", "secs", "second", "seconds"], "s", Cat::Time, 1.0),
    u(&["min", "mins", "minute", "minutes"], "min", Cat::Time, 60.0),
    u(&["h", "hr", "hrs", "hour", "hours"], "h", Cat::Time, 3600.0),
    u(&["d", "day", "days"], "days", Cat::Time, 86400.0),
    u(&["week", "weeks", "wk"], "weeks", Cat::Time, 604800.0),
    u(&["year", "years", "yr", "yrs"], "years", Cat::Time, 31557600.0),
    u(&["bit", "bits"], "bit", Cat::Data, 0.125),
    u(&["b", "byte", "bytes"], "B", Cat::Data, 1.0),
    u(&["kb", "kilobyte", "kilobytes"], "KB", Cat::Data, 1e3),
    u(&["mb", "megabyte", "megabytes"], "MB", Cat::Data, 1e6),
    u(&["gb", "gigabyte", "gigabytes"], "GB", Cat::Data, 1e9),
    u(&["tb", "terabyte", "terabytes"], "TB", Cat::Data, 1e12),
    u(&["kib", "kibibyte", "kibibytes"], "KiB", Cat::Data, 1024.0),
    u(&["mib", "mebibyte", "mebibytes"], "MiB", Cat::Data, 1048576.0),
    u(&["gib", "gibibyte", "gibibytes"], "GiB", Cat::Data, 1073741824.0),
    u(&["tib", "tebibyte", "tebibytes"], "TiB", Cat::Data, 1099511627776.0),
    u(&["c", "celsius", "centigrade"], "°C", Cat::Temp, 0.0),
    u(&["f", "fahrenheit"], "°F", Cat::Temp, 0.0),
    u(&["k", "kelvin"], "K", Cat::Temp, 0.0),
];

fn unit(w: &str) -> Option<&'static Unit> {
    UNITS.iter().find(|u| u.names.contains(&w))
}

fn to_celsius(v: f64, show: &str) -> f64 {
    match show {
        "°F" => (v - 32.0) * 5.0 / 9.0,
        "K" => v - 273.15,
        _ => v,
    }
}
fn from_celsius(c: f64, show: &str) -> f64 {
    match show {
        "°F" => c * 9.0 / 5.0 + 32.0,
        "K" => c + 273.15,
        _ => c,
    }
}

#[derive(Clone, Debug)]
pub struct Conv {
    pub val: f64,
    pub from: &'static str, // canonical alias of the unit
    pub out: f64,
    pub to: &'static str,
}

fn fmt_q(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        fmt_num(v)
    } else if v.abs() >= 1e15 || v.abs() < 1e-4 {
        format!("{v:.4e}")
    } else {
        fmt_round(v, 4)
    }
}

fn run_conv(val: f64, ua: &'static Unit, ub: &'static Unit) -> Result<(Conv, String), String> {
    if ua.cat != ub.cat {
        return Err(format!("I can't convert {} to {}, they measure different things.", ua.show, ub.show));
    }
    let out = if ua.cat == Cat::Temp { from_celsius(to_celsius(val, ua.show), ub.show) } else { val * ua.to_base / ub.to_base };
    if !out.is_finite() {
        return Err("That result is out of range.".into());
    }
    let line = format!("{} {} = {} {}", fmt_q(val), ua.show, fmt_q(out), ub.show);
    Ok((Conv { val, from: ua.names[0], out, to: ub.names[0] }, line))
}

/// "convert 10 km to miles", "72 f in c", "3.5 gb to mib" ...
pub fn convert(text: &str) -> Option<Result<(Conv, String), String>> {
    let t: Vec<Tok> = lex(text)?;
    let t: Vec<Tok> = t
        .into_iter()
        .filter(|x| !matches!(x, Tok::Word(w) if ["degrees", "degree", "deg"].contains(&w.as_str())) && *x != Tok::Sym('°'))
        .collect();
    for i in 0..t.len() {
        let Tok::Num(mut val) = t[i] else { continue };
        if i > 0 && t[i - 1] == Tok::Sym('-') {
            val = -val;
        }
        let (Some(Tok::Word(a)), Some(Tok::Word(conn)), Some(Tok::Word(b))) = (t.get(i + 1), t.get(i + 2), t.get(i + 3)) else {
            continue;
        };
        if !["to", "in", "into", "as", "equals"].contains(&conn.as_str()) {
            continue;
        }
        let (Some(ua), Some(ub)) = (unit(a), unit(b)) else { continue };
        return Some(run_conv(val, ua, ub));
    }
    None
}

#[cfg(test)]
pub fn try_convert(text: &str) -> Option<String> {
    convert(text).map(|r| r.map(|x| x.1).unwrap_or_else(|e| e))
}

const CONV_FILLER: [&str; 17] = [
    "and", "what", "about", "how", "convert", "it", "in", "to", "into", "as", "is", "then", "whats", "now", "also", "bout", "please",
];

/// Follow-ups that lean on the previous conversion:
///   "and in meters?"      -> same quantity, new unit
///   "what about 5 kg?"    -> new quantity, previous target unit
///   "convert that to mi"  -> previous *result*, new unit
pub fn convert_followup(text: &str, last: &Conv) -> Option<Result<(Conv, String), String>> {
    let toks: Vec<Tok> = lex(text)?.into_iter().filter(|t| !matches!(t, Tok::Word(w) if CONV_FILLER.contains(&w.as_str()) || w == "?") && *t != Tok::Sym('?')).collect();
    let that = lex(text)?.iter().any(|t| matches!(t, Tok::Word(w) if w == "that"));
    // a bare unit word ("hours", "s", "m") is only a follow-up when it comes with a lead-in like "and in" / "to"
    let lead = lex(text)?.iter().any(|t| matches!(t, Tok::Word(w) if ["and", "in", "to", "into", "as", "about", "bout", "convert"].contains(&w.as_str())));
    let toks: Vec<Tok> = toks.into_iter().filter(|t| !matches!(t, Tok::Word(w) if w == "that")).collect();
    let run = |v: f64, from: &str, to: &str| Some(run_conv(v, unit(from)?, unit(to)?));
    match toks.as_slice() {
        [Tok::Word(u)] if lead && unit(u).is_some() => {
            if that { run(last.out, last.to, u) } else { run(last.val, last.from, u) }
        }
        [Tok::Num(n), Tok::Word(u)] if unit(u).is_some() => run(*n, u, last.to),
        _ => None,
    }
}
