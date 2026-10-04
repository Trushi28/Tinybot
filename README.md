# TinyBot

A chatbot with a face, a memory and a mood, written from scratch in Rust with **zero dependencies**.

The brain is a tiny neural network (**66k parameters, 109 KB on disk**) that routes what you say to an
intent. Around it sit hand-written skills (math, units, dates, timers, notes), a 24-turn conversation memory,
and a terminal UI where you can watch the network think.

```
╭ TinyBot ─ Lv 3 Byte ─ ▓▓▓▓▓▓░░░░ 41/75 XP ─ mood happy ─ ♥♥♡♡♡ ─ ⏱ 04:32 stretch ─────────────╮
│                 ✦                  │ you  how are you                                         │
│               ✧ │      ♥           │ bot  Running at full speed, thanks for asking. You?      │
│           ╭───────────╮            │                                                          │
│           │  ♥     ♥  │            │ you  fine, and in meters?                                │
│           │░  ╰───╯  ░│            │ bot  10 km = 10000 m                                     │
│           ╰───────────╯            │                                                          │
│           happy · Byte             │  ★   Fact book: 4/53 collected                           │
│                                    │                                                          │
│ ─ BRAIN ────────────────────────── │ ─ CONSOLE · art ──────────────────────────────────────── │
│ skill:convert  ██████████ 100%     │  /\_/\                                                   │
│ ─ NEURONS ──────────────────────── │ ( o.o )                                                  │
│   ▓▓ ░░ ██ ▒▒ ░░ ▓▓ ██ ░░         │  > ^ <                                                   │
│ ─ STATUS ───────────────────────── │                                                          │
│ ♥♥♡♡♡ friend (34 pts)              │                                                          │
╰────────────────────────────────────┴──────────────────────────────────────────────────────────╯
 you ▸
```

*(Schematic. The real thing is colour, animated, and the neuron grid is a live heatmap.)*

## Quick start

```bash
cargo run --release            # animated terminal UI
cargo run --release -- chat --plain   # bare REPL (also used automatically when piped)
```

The first run trains the model (about 30 s on one core, parallel across cores) and saves it to `bot.bin`.
After that it loads instantly. Edit `intents.txt` or `facts.txt` and it retrains by itself.

You need a terminal with 256-colour support and at least 84×30. Resize before launching; the layout is
fixed at start. I tested on Linux through a pseudo-terminal; macOS terminals should behave the same, and
Windows Terminal should work but is untested (use `--plain` if anything misrenders).

## What it can do

| | |
|---|---|
| **Chat** | 52 intents: greetings, moods, jokes, riddles, identity, games, small talk. Greets by time of day about half the time |
| **Math** | `12*(3+4)`, `20% of 80`, `sqrt 144`, `2^10`, `5!`, `sin(pi/2)` |
| **Follow-ups** | `times 2`, `double that`, `and in meters?`, `what about 5 miles`, `again`, `why` |
| **Units** | length, mass, volume, speed, time, data (KB vs KiB), temperature |
| **Dates & time** | `days until 2026-12-25`, `in 10 days`; clock and dates use your local time zone (auto-detected, or `/tz +5:30`) |
| **Timers** | `timer for 5 minutes`, `remind me in 10 minutes to stretch`, `pomodoro`. Rings in real time |
| **Memory** | `my favorite color is teal`, `i live in Berlin`, `i like pizza`, `i hate mondays`, `what do you know about me` |
| **Notes** | `note: buy stamps`, `remember to call mom`, `show my notes`, `search notes for stamps` |
| **Birthday** | `my birthday is 12 march`, `how many days until my birthday` (it greets you on the day) |
| **List** | `add milk to my list`, `show my list`, `remove 1` |
| **Fact book** | 53 facts in 7 categories: `tell me a space fact`, `animal fact`, `fact of the day`. Unheard ones come first |
| **ASCII art** | 16 pieces (cat, dog, mouse, bunny, person, fish, owl, house, flower, star, sun, tree, heart, rocket, crab, robot). Ask for something it can't draw and it tells you, instead of drawing something random |
| **Games** | riddles, rock-paper-scissors (the bot learns your habits), number guessing |

## It has feelings, sort of

- **18 faces** driven by what's happening: love (hearts float up), anger (steam), sad (tears), laughing, wow,
  confused, curious, thinking (spinning eyes, thought bubble), proud, wink, sulking (storm cloud), sleepy and asleep (z Z z).
  Preview any with `/emote love`.
- **Mood** drifts with how you treat it. Compliments lift it, insults sink it. Push it far enough and it
  **sulks**: it refuses to help until you say sorry (or pay it a compliment). It forgives on its own after a few turns.
- **Bond** grows as you talk (stranger → acquaintance → friend → buddy → best friend), shown as hearts.
  It's remembered between sessions, so it notices when you've been away for days.
- **Energy** drains as you chat and recovers while idle. Tired bots yawn. Leave it alone and it dozes off.
- **XP, levels, titles** (Bit → Nibble → Byte → Neuron → … → Overfit Overlord), **achievements**, day **streaks**,
  and an **intent dex** (`/dex`): `???` until you discover each thing it can do.
- **Safety net.** A deterministic keyword check runs before the neural net, games, sulking and every skill.
  A message about suicide or self-harm always gets a plain, serious answer pointing to real help, with no XP,
  badges or mood effects. The tests prove it works even with an almost untrained model, because a 66k-parameter
  classifier should never be the only thing standing between someone and that answer.
- **Night Owl** badge: chatting between midnight and 5am *your* local time (needs a known time zone).

## Commands

```
/dex  /facts  /memory  /notes  /timers   what you've discovered, heard, told it
/stats      level, XP, badges, model size and speed
/history    its short-term memory of this chat
/why        which words drove the last decision
/fix <tag>  its last answer was wrong, it meant <tag>   (learns in the background)
/teach a => b   create a new intent: when you say a, it replies b
/unlearn    undo the last thing it learned
/tz +5:30   set your time zone (or /tz auto)
/intents  /bench  /emote <x>  /anim  /clear  /forget  /help  /quit

Learning runs on a background thread (10 to 30 s on one core), so you can keep chatting; a spinner shows in
the header and the new brain swaps in when it's done.
```

Command output lands in the **console pane under the chat**, so the conversation stays readable.
Lists, notes and ASCII art go there too. Press Enter on an empty line to clear it.

## How it works

```
 your text
    │
    ├─► skills first (high precision, no training data): todo · timers · birthday · units ·
    │   follow-ups · dice · random · math · facts · notes
    │
    └─► neural router ──► intent ──► reply (context-aware, mood-flavoured)
          hashed features ─► 2 pooled embeddings ─► ReLU layer ─► softmax
          └ a learned "what usually follows what" prior breaks ties on short replies
```

**Features.** Words, bigrams and character 3/4-grams are hashed into an embedding table (fastText-style),
plus a small hand-written semantic lexicon (so "lonely" leans on what "sad" taught it) with negation
("not great" ≠ "great"). Numbers collapse to one token. Word-level and char-level features are pooled
separately so one decisive word isn't averaged away.

**Training.** Hand-written backprop and SGD, label smoothing, typo and word-drop augmentation.

**Parameter efficiency: distillation.** A 5-net teacher ensemble (3.6M parameters) is trained first.
It then labels thousands of synthetic phrases (typos, dropped words, spliced phrases) and **one small student
net learns to reproduce its soft predictions**. Only the student ships, with int8 embedding rows and only the
rows that were ever trained stored.

5-fold cross-validation, 842 phrases, 52 intents (`cargo run --release -- eval`):

| | Teacher (5 nets) | Student (shipped) |
|---|---:|---:|
| Parameters | 3.61M | 67k |
| File size | 4,054 KB | 109 KB |
| Clean accuracy | 75.9% | 72.6% |
| Accuracy with typos | 67.9% | 62.1% |
| Auto-answered precision | 88.1% | 87.7% |
| Out-of-scope rejection | 86% | 89% |
| Inference | | about 5–15 µs per query |

**Size, concretely.** The shipped model is 67k parameters stored as int8 rows in a 110 KB file. The whole
running process, model and phrase index included, uses about 4.3 MB of RAM. The 3.6M-parameter teacher exists
only while training and is never saved.

The student is about 54× smaller for roughly 3 points of clean accuracy. **Typo robustness is the weak spot**
(about 6 points lost). More noisy distillation samples didn't fix it; more phrases in `intents.txt` will.
`cargo run --release -- sweep` trains students of 8 different sizes so you can pick your own trade-off.

**Context.** The last 24 turns are kept. Replies can depend on the previous intent (`>@prev` lines),
numbers and units carry over between turns, `again` re-runs the last repeatable thing, and "what did I just say"
and "recap" read the history. The transition prior (which intent tends to follow which) is seeded from
`intents.txt` and keeps learning in `flow.txt`. On a 36-case probe of short follow-ups, averaged over 4
independently trained students: 93.1% without it, 97.9% with it. That's a small test, so treat it as
"helps a little", not proof.

**Confidence.** The temperature of the output distribution is calibrated on held-out data, so "90% sure" means
something. High confidence answers, middle confidence asks *"did you mean something like …?"* (only when the
closest known phrase is genuinely close) and learns from your yes, low confidence admits it doesn't know.
If it learns something wrong, `/unlearn` removes it.

## Customising

`intents.txt` is plain text:

```
[greeting]                      # an intent
hello                           # example phrases, one per line
hey there
> Hey! What's on your mind?     # replies, picked at random
>@mood_bad I'm here for you.    # only used when the previous intent was mood_bad
> =joke                         # borrow another intent's replies
> I'm {mood}, level {level}.    # placeholders
```

Placeholders: `{name} {time} {date} {weekday} {coin} {dice} {rand} {mood} {level} {title} {xp} {streak}
{msgs} {dex} {dex_total} {known_days} {bond_label} {bond} {hearts} {energy} {greet} {daypart} {params} {tz}`.
`{greet}` is "Good morning"/"Good evening"…, `{daypart}` is morning/afternoon/evening/night; mix lines that use
them with lines that don't and the bot is only *sometimes* time-aware.
Riddles are `question | answer/alt answer`. Add the phrases first, run `eval`, then trust the numbers.

`art.rs` holds the ASCII gallery (add a `const` and a `GALLERY` row). `facts.txt` is `category: fact`, one per line. New categories need a keyword entry in `src/facts.rs`.

## Files it writes (all next to the binary, all plain text except the model)

| File | What |
|---|---|
| `bot.bin` | the trained student (int8) |
| `memory.txt` | your facts, list, notes, XP, dex, bond, fact book |
| `flow.txt` | learned intent-to-intent transitions |
| `learned.txt` | phrases and intents from `/fix`, `/teach` and "yes" to a clarification |

Delete any of them to reset that part. `/forget` wipes `memory.txt`. Writes to `learned.txt` and `memory.txt` are
sanitised so input can't break the line-based formats. Nothing touches the network.

## Source map

| | |
|---|---|
| `model.rs` | nets, backprop, ensemble, distillation, int8 storage, calibration |
| `text.rs` | tokenising, hashed features, lexicon, negation, typo noise |
| `bot.rs` | routing, dialogue state, follow-ups, games, context prior |
| `skills.rs` `calc.rs` | dates, dice, memory, notes, birthday, timers · calculator, units |
| `persona.rs` | mood, sulking, bond, energy, XP, badges, emotion mapping |
| `facts.rs` `art.rs` | the fact book · ASCII gallery |
| `tui.rs` | the terminal UI: face canvas, particles, brain panel, console |
| `cmds.rs` `train.rs` | slash commands · training pipeline, eval, benchmarks |
| `tests.rs` | 21 tests: calculator edge cases and hostile nesting, conversions, dates, memory, notes, timers, facts, sulking, a fuzz run |

```bash
cargo test --release     # includes a 6,000-input fuzz run; the first run trains a model
cargo run --release -- eval      # teacher vs student cross-validation
cargo run --release -- ctx       # does the context prior help?
cargo run --release -- bench     # inference speed
```

## Honest limitations

- It's an intent router, not a language model. It can't answer questions outside what's in `intents.txt`,
  `facts.txt` and the skills. When it doesn't know, it says so.
- About 74% of held-out phrases are classified correctly, and it asks or declines instead of guessing when
  unsure. More phrases per intent is the single biggest lever.
- Time zone comes from the OS (`date +%z`, Unix only) or `/tz`. If neither is available it falls back to UTC and
  switches time-of-day greetings off instead of guessing.
- The terminal UI needs Unix-style terminal size detection (`stty`) to fit the window; elsewhere it falls back to
  `LINES`/`COLUMNS` or 112×36. Ctrl-C quits abruptly: your progress is saved after every message.
