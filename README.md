# TinyBot

A chatbot with a face, a memory and a mood, written from scratch in Rust with **zero dependencies**.

The brain is a small neural network you can size from **68k to 4M parameters** and build as a bag-of-features
net, a temporal CNN or a bidirectional GRU (default: a **251k-parameter GRU, about 310 KB on disk**) that routes
what you say to an intent. Around it sit hand-written skills (math, units, dates, timers, notes), a 24-turn conversation memory,
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

No trained models are checked in (`bot.bin` and `models/` are in `.gitignore`), so the **first run trains the one it
needs**: about 2 minutes on one core for the default GRU, much less on several. After that it loads instantly. If you
edit `intents.txt` the saved model is out of date and it retrains by itself. Teaching it with `/teach`, `/fix` or a
"yes" to a clarification updates the saved model in the background, and that model is accepted at the next launch
without another full retrain.

Pick another brain with `--size` and `--arch`, for example `cargo run --release -- --size large --arch bow`.
See [Model sizes](#model-sizes-and-architectures).

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
| **Knowledge** | about 170 short topics in `kb.txt` (countries, India, space, science, computing, history, sport): `what is the capital of turkey`, `how fast is a cheetah`, then `tell me more`. Teach it more instantly with `/know topic => text` (saved in `kb_user.txt`, no retraining). Questions it only half understands ("capital of mars") are declined |
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
- **Safety net.** A deterministic check of about 50 phrases and paraphrases ("want to die", "nobody would miss
  me", "tired of living", …) runs before the neural net, games, sulking and every skill. A message about suicide
  or self-harm always gets a plain, serious answer pointing to real help, with no XP, badges or mood effects.
  The tests prove it works even with an almost untrained model, because a small classifier should never be the
  only thing standing between someone and that answer. The network adds a second opinion for wordings the list
  misses, but it ignores inputs with digits: a 251k GRU once answered "10 km to miles" with the crisis message,
  so there is now a regression test that trains its own small GRU and checks ordinary requests against it. It's a keyword list, so it will still miss unusual
  phrasings; treat it as a safety net, not a guarantee.
- **Night Owl** badge: chatting between midnight and 5am *your* local time (needs a known time zone).

## Commands

```
/dex  /facts  /memory  /notes  /timers   what you've discovered, heard, told it
/stats      level, XP, badges, model size and speed
/model      which brain is running (architecture, size) and which others are built
/confidence 0.7   how sure it must be to answer (higher = fewer mistakes, more questions)
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

## Model sizes and architectures

Three architectures, all hand-written, six sizes each (`cargo run --release -- models` prints this table):

| Size | Parameters | Bag-of-features | Temporal CNN | Bidirectional GRU |
|---|---:|---|---|---|
| `nano` | 68k | `bow-nano` | `cnn-nano` | `gru-nano` |
| `small` | 100k | `bow-small` | `cnn-small` | `gru-small` |
| `base` | 250k | `bow-base` | `cnn-base` | **`gru-base` (default)** |
| `large` | 500k | `bow-large` | `cnn-large` | `gru-large` |
| `xl` | 1M | `bow-xl` | `cnn-xl` | `gru-xl` |
| `max` | 4M | `bow-max` | `cnn-max` | `gru-max` |

Nothing is prebuilt: each model is built the first time you ask for it, or all at once with `train --all`.
Each size is fitted so every architecture lands within 3% of the same parameter budget, so the comparison is fair.
"Parameters" means allocated parameters (what the model costs in memory); the file on disk is smaller because
only trained embedding rows are stored, as int8.

```bash
cargo run --release -- --size large --arch gru      # run that brain (trains it on first use)
cargo run --release -- train --all --archs bow,gru  # build many sizes into models/ from ONE shared teacher
cargo run --release -- sweep --folds 0,3            # compare every architecture and size (slow: use several cores)
cargo run --release -- speed                        # training cost per step for each one
```

- **Bag-of-features** (fastText-style): hashed word, bigram and character n-gram embeddings, mean-pooled in two
  channels, then a ReLU layer. The fastest: 6 to 19 µs per query from 68k to 4M parameters.
- **Temporal CNN**: the same hashed token embeddings, then 1D convolutions of width 1, 2 and 3, max-over-time
  pooling and a dense head. Keeps local word order. 13 to 31 µs per query.
- **Bidirectional GRU**: the same token embeddings run through a GRU in both directions with hand-written
  backprop through time, mean-pooled. Keeps order and negation ("not at all", "so so"). 19 µs per query at 68k,
  46 µs at 250k, 332 µs at 4M.
- All three are trained with typo and word-drop augmentation, label smoothing and temperature calibration. The
  sequence models use Adam with gradient clipping, and every gradient is checked against finite differences in the tests.
- LSTM is not implemented (the GRU covers the same ground with fewer parameters).

### What the measurements say

Same data, same teacher, same folds. Clean accuracy, 2 folds (0 and 3), 347 held-out phrases, so about ±2 points of
noise. Every student is distilled from the same teacher (3 bag-of-features nets + 1 CNN + 1 GRU, 78.4% clean).

| Size | Bag-of-features | CNN | GRU |
|---|---:|---:|---:|
| 68k | 75.8% | 72.9% | **77.2%** |
| 100k | 75.2% | 74.9% | **77.2%** |
| 250k | 74.9% | 75.5% | **78.4%** |
| 500k | 77.8% | 76.4% | **79.0%** |
| 1M | 78.1% | **79.0%** | 78.7% |
| 4M | 79.3% | not measured | not measured |

- **The GRU wins at small and medium sizes**, and at 68k a GRU already matches the 5-net bag-of-features teacher
  (78.7%). That's why a 250k GRU is the default.
- **The CNN never clearly beats the bag-of-features net.**
- **Going from 250k to 4M adds about 1 point, which is inside the noise.** With about a thousand training
  phrases the data is the ceiling, not the model. Bigger models mostly buy typo robustness (60.8% at 68k vs 68.9% at 4M for
  the bag-of-features net), not accuracy.
- The 4M CNN and GRU were not benchmarked: the sweep process was killed partway through in my sandbox, and I
  didn't rerun those two rows. `sweep --archs cnn,gru --tiers max` will measure them if you have the cores.

## How it works

```
 your text
    │
    ├─► safety net first (keyword check for self-harm), then skills (high precision, no training data):
    │   todo · timers · birthday · units · follow-ups · dice · random · math · facts · notes
    │
    └─► neural router ──► intent ──► reply (context-aware, mood-flavoured)
          hashed token features ─► GRU / CNN / bag-of-features ─► dense head ─► softmax
          └ a learned "what usually follows what" prior breaks ties on short replies
```

**Features.** Words, bigrams and character 3/4-grams are hashed into an embedding table, plus a small
hand-written semantic lexicon (so "lonely" leans on what "sad" taught it) with negation ("not great" ≠ "great").
Numbers collapse to one token.

**Training.** Hand-written backprop, label smoothing, typo and word-drop augmentation, plus **label-aware synonym
augmentation**: inside an intent like `mood_good`, a word is swapped for another member of its semantic class
("im feeling good" becomes "im feeling great"). It is only ever applied to training data, never to held-out phrases.

**Parameter efficiency: distillation.** A teacher of five nets (3 bag-of-features, 1 CNN, 1 GRU; 3.2M parameters
in total) is trained first. It then labels thousands of synthetic phrases (typos, dropped words, spliced
phrases) and **one small student learns to reproduce its soft predictions**. Only the student ships.

5-fold cross-validation of the default (`cargo run --release -- eval`), 996 phrases, 52 intents:

| | Teacher (5 nets) | Student (GRU, 251k) |
|---|---:|---:|
| Parameters | 3.22M | 251k |
| File size | 3,230 KB | 308 KB |
| Clean accuracy | 83.0% | 80.9% |
| Accuracy with typos | 74.8% | 71.1% |
| Auto-answered precision | 91.3% | 90.6% |
| Share auto-answered | 84.1% | 82.6% |
| Out-of-scope rejection | 77% | 89% |

Caveat: I added paraphrases to `intents.txt` for the intents that were missing in an earlier run, so part of the gain
over earlier versions is data, not the model. The controlled table above is the fair architecture comparison.

**Precision is a dial.** `/confidence 0.7` (or `tinybot eval`'s table) trades coverage for precision. Below the
threshold the bot asks "did you mean …?" or admits it doesn't know instead of guessing:

| Answer threshold | Answered | Precision | Wrong answers |
|---:|---:|---:|---:|
| 0.40 | 89.2% | 87.5% | 11.1% of inputs |
| 0.50 | 84.5% | 89.4% | 8.9% |
| **0.55 (default)** | 82.6% | 90.6% | 7.7% |
| 0.60 | 80.0% | 91.3% | 6.9% |
| 0.70 | 74.0% | 92.7% | 5.4% |
| 0.80 | 67.4% | 95.2% | 3.2% |
| 0.90 | 56.3% | 97.7% | 1.3% |

**Context.** The last 24 turns are kept. Replies can depend on the previous intent (`>@prev` lines),
numbers and units carry over between turns, `again` re-runs the last repeatable thing, and "what did I just say"
and "recap" read the history. The transition prior (which intent tends to follow which) is seeded from
`intents.txt` and keeps learning in `flow.txt`. On a small 36-case probe of short follow-ups it scored higher with
the prior than without, but that's too small a test to prove much.

**Confidence.** The temperature of the output distribution is calibrated on held-out data, so "90% sure" means
something. The "did you mean …?" similarity threshold is calibrated per model too (each architecture's embeddings
have their own scale): it sits above what out-of-scope phrases reach and below what related phrases reach. If it
learns something wrong, `/unlearn` removes it.

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
| `bot.bin` | the default student (GRU 250k, int8), trained on first run |
| `models/<arch>-<size>.bin` | every other size and architecture you build |
| `memory.txt` | your facts, list, notes, XP, dex, bond, fact book |
| `flow.txt` | learned intent-to-intent transitions |
| `learned.txt` | phrases and intents from `/fix`, `/teach` and "yes" to a clarification |
| `kb_user.txt` | topics you added with `/know` |

Delete any of them to reset that part. `/forget` wipes `memory.txt`. Writes to `learned.txt` and `memory.txt` are
sanitised so input can't break the line-based formats. Nothing touches the network.

## Source map

| | |
|---|---|
| `model.rs` | architectures and tiers, bag-of-features net, ensembles, distillation, int8 storage, calibration |
| `seqnet.rs` | the temporal CNN and bidirectional GRU, with backprop through time and Adam |
| `text.rs` | tokenising, hashed features, lexicon, negation, typo noise |
| `bot.rs` | routing, dialogue state, follow-ups, games, context prior |
| `skills.rs` `calc.rs` | dates, dice, memory, notes, birthday, timers · calculator, units |
| `persona.rs` | mood, sulking, bond, energy, XP, badges, emotion mapping |
| `facts.rs` `art.rs` | the fact book · ASCII gallery |
| `tui.rs` | the terminal UI: face canvas, particles, brain panel, console |
| `cmds.rs` `train.rs` | slash commands · training pipeline, eval, benchmarks |
| `tests.rs` + `seqnet.rs` | 55 tests, including finite-difference gradient checks for the CNN and GRU: calculator edge cases and hostile nesting, conversions, dates, memory, notes, timers, facts, games, corrupt model files, the terminal UI's column widths, sulking, a fuzz run |

```bash
cargo test --release     # includes a 6,000-input fuzz run; trains several small models, about a minute
cargo run --release -- eval      # teacher vs student cross-validation
cargo run --release -- ctx       # does the context prior help?
cargo run --release -- bench     # inference speed
```

## Honest limitations

- It's an intent router, not a language model. It can't answer questions outside what's in `intents.txt`,
  `facts.txt` and the skills. When it doesn't know, it says so.
- About 81% of held-out phrases are classified correctly, and it asks or declines instead of guessing when
  unsure. More phrases per intent is the single biggest lever, bigger than any architecture or size.
- On one CPU core the first build of the default GRU takes about 2 minutes, and the 4M GRU takes much longer.
  Training uses one thread per ensemble member, so more cores help.
- Time zone comes from the OS (`date +%z`, Unix only) or `/tz`. If neither is available it falls back to UTC and
  switches time-of-day greetings off instead of guessing.
- The terminal UI needs Unix-style terminal size detection (`stty`) to fit the window; elsewhere it falls back to
  `LINES`/`COLUMNS` or 112×36. Ctrl-C quits abruptly: your progress is saved after every message.
