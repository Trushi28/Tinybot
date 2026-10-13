#!/usr/bin/env bash
# Builds everything the web page needs into web/:  tinybot.wasm (the engine) and bot.bin (the trained model).
# Run it after changing any .rs file or intents.txt / facts.txt / kb.txt, then commit web/tinybot.wasm and web/bot.bin.
#
#   ./build-web.sh                  build everything
#   ./build-web.sh --serve          build, then serve web/ on http://localhost:8000
#   ./build-web.sh --skip-models    keep the models already in web/models/ (saves the slowest step; only safe if
#                                   intents.txt and the model code did not change, the Rust test below checks that)
#
# web/models/ holds the other architectures and sizes for the page's model picker (bow, cnn, gru x nano..xl); bot.bin is
# the default (gru base). Set TB_TIERS to ship fewer sizes, e.g. TB_TIERS=nano,small,base ./build-web.sh
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

SKIP_MODELS=0; SERVE=0
for a in "$@"; do
  case "$a" in
    --skip-models) SKIP_MODELS=1 ;;
    --serve) SERVE=1 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

echo "== 1/4  engine -> web/tinybot.wasm"
cargo rustc --lib --release --target wasm32-unknown-unknown --crate-type cdylib
cp target/wasm32-unknown-unknown/release/tinybot.wasm web/tinybot.wasm
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -O3 web/tinybot.wasm -o web/tinybot.wasm
else
  echo "   (wasm-opt not found, skipping the optional size pass)"
fi

echo "== 2/4  model -> web/bot.bin"
# The browser build only knows the intents baked into the binary, so train in an empty directory:
# no intents.txt, learned.txt or facts.txt can leak in and make the tags disagree with the wasm.
cargo build --release
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
( cd "$WORK" && "$ROOT/target/release/tinybot" train --arch gru --size base )
cp "$WORK/bot.bin" web/bot.bin

echo "== 3/4  other architectures and sizes -> web/models/"
TIERS="${TB_TIERS:-nano,small,base,large}"
if [ "$SKIP_MODELS" = 1 ]; then
  echo "   (--skip-models: keeping what is in web/models/)"
else
  # one shared teacher builds every student; this is the slow step (several minutes, more cores help)
  ( cd "$WORK" && "$ROOT/target/release/tinybot" train --all --archs bow,cnn,gru --tiers "$TIERS" )
  rm -rf web/models
  mkdir -p web/models
  for f in "$WORK"/models/*.bin; do
    n="$(basename "$f" .bin)"
    [ "$n" = "gru-base" ] && continue   # that one is bot.bin, built above
    cp "$f" "web/models/$n.bin"
  done
  # the list the page reads; ids are <arch>-<size>
  {
    printf '{"default":"gru-base","models":[\n'
    printf '{"id":"gru-base","arch":"gru","size":"base","file":"bot.bin","bytes":%s}' "$(wc -c < web/bot.bin | tr -d ' ')"
    for f in web/models/*.bin; do
      n="$(basename "$f" .bin)"
      printf ',\n{"id":"%s","arch":"%s","size":"%s","file":"models/%s.bin","bytes":%s}' "$n" "${n%%-*}" "${n#*-}" "$n" "$(wc -c < "$f" | tr -d ' ')"
    done
    printf '\n]}\n'
  } > web/models/index.json
fi

echo "== 4/4  check"
# the Rust tests refuse to pass if any shipped model does not match the built-in intents
cargo test --release --lib shipped_model -- --quiet
ls -l web/tinybot.wasm web/bot.bin
du -sh web/models 2>/dev/null || true

if [ "$SERVE" = 1 ]; then
  echo "serving http://localhost:8000  (Ctrl-C to stop)"
  cd web && exec python3 -m http.server 8000
fi
