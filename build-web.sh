#!/usr/bin/env bash
# Builds everything the web page needs into web/:  tinybot.wasm (the engine) and bot.bin (the trained model).
# Run it after changing any .rs file or intents.txt / facts.txt / kb.txt, then commit web/tinybot.wasm and web/bot.bin.
#
#   ./build-web.sh            build both
#   ./build-web.sh --serve    build, then serve web/ on http://localhost:8000
#
# Needs: Rust with the wasm target (rustup target add wasm32-unknown-unknown). Optional: wasm-opt (binaryen) to shrink the .wasm.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

if ! rustc --print target-list | grep -qx wasm32-unknown-unknown; then
  echo "this rustc has no wasm32 support" >&2; exit 1
fi
if ! rustup target list --installed 2>/dev/null | grep -qx wasm32-unknown-unknown; then
  echo "missing the wasm target. Run:  rustup target add wasm32-unknown-unknown" >&2; exit 1
fi

echo "== 1/3  engine -> web/tinybot.wasm"
cargo rustc --lib --release --target wasm32-unknown-unknown --crate-type cdylib
cp target/wasm32-unknown-unknown/release/tinybot.wasm web/tinybot.wasm
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -O3 web/tinybot.wasm -o web/tinybot.wasm
else
  echo "   (wasm-opt not found, skipping the optional size pass)"
fi

echo "== 2/3  model -> web/bot.bin"
# The browser build only knows the intents baked into the binary, so train in an empty directory:
# no intents.txt, learned.txt or facts.txt can leak in and make the tags disagree with the wasm.
cargo build --release
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
( cd "$WORK" && "$ROOT/target/release/tinybot" train --arch gru --size base )
cp "$WORK/bot.bin" web/bot.bin

echo "== 3/3  check"
# the Rust test refuses to pass if web/bot.bin does not match the built-in intents
cargo test --release --lib shipped_model_matches_the_intents -- --quiet
ls -l web/tinybot.wasm web/bot.bin

if [ "${1:-}" = "--serve" ]; then
  echo "serving http://localhost:8000  (Ctrl-C to stop)"
  cd web && exec python3 -m http.server 8000
fi
