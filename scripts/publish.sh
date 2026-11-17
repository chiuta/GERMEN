#!/usr/bin/env bash
# Publishes GERMEN to GitHub as chiuta/GERMEN.
#
# Run it from Git Bash, from the unpacked project folder:
#     bash scripts/publish.sh            # creates the repo, pushes, adds topics and 10 starter issues
#     bash scripts/publish.sh --no-issues
#
# Needs: git and the GitHub CLI (gh), already signed in (check with: gh auth status).
# Commits are made as Alexio <alexio@trom.tf>, like in the other repos.
set -euo pipefail

OWNER="chiuta"
REPO="GERMEN"
GIT_NAME="Alexio"
GIT_EMAIL="alexio@trom.tf"
DESCRIPTION="Train a small language model from scratch in pure Rust, on an ordinary CPU (Burn, Flex backend). No Python, no CUDA."
WITH_ISSUES=1
[ "${1:-}" = "--no-issues" ] && WITH_ISSUES=0

cd "$(dirname "$0")/.."

command -v git >/dev/null || { echo "git nu este instalat"; exit 1; }
command -v gh  >/dev/null || { echo "gh (GitHub CLI) nu este instalat"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "gh nu este autentificat: rulați 'gh auth login'"; exit 1; }

(
  if [ ! -d .git ]; then
    git init -b main
  fi
  git config user.name  "$GIT_NAME"
  git config user.email "$GIT_EMAIL"
  git add -A
  if ! git diff --cached --quiet; then
    git commit -m "GERMEN 0.1.0: train a small language model from scratch in pure Rust"
  fi
  git branch -M main

  if gh repo view "$OWNER/$REPO" >/dev/null 2>&1; then
    echo "Depozitul $OWNER/$REPO există deja; îl folosesc."
    git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$OWNER/$REPO.git"
    git push -u origin main
  else
    gh repo create "$OWNER/$REPO" --public --description "$DESCRIPTION" --source . --remote origin --push
  fi

  gh repo edit "$OWNER/$REPO" \
    --add-topic rust --add-topic llm --add-topic language-model --add-topic transformer \
    --add-topic burn --add-topic machine-learning --add-topic cpu --add-topic trade-free \
    >/dev/null || echo "(nu am putut adăuga subiectele; nu e grav)"
)

if [ "$WITH_ISSUES" = "1" ]; then
  issue() { gh issue create --repo "$OWNER/$REPO" --title "$1" --body "$2" >/dev/null && echo "  problemă creată: $1"; }

  issue "BPE tokenizer (trainable, with byte fallback)" \
"The byte-level vocabulary is simple but makes sequences 3-4x longer than a subword vocabulary, which wastes compute. Add a trainable BPE tokenizer (merges learned from the corpus, 256 byte tokens as fallback so nothing is out-of-vocabulary), store it in the checkpoint, and make \`germen train\`/\`generate\` use it. Keep the byte tokenizer as the default for tiny corpora. Needs a decision on the vocabulary size (suggested: 2k-8k for models up to 25M parameters)."

  issue "Restore AdamW moment estimates on --resume" \
"\`--resume\` currently restores weights and the step counter only; the optimizer restarts with zeroed moments, which causes a small loss bump. Burn's \`ModuleOptimizer\` has \`to_record()\`/\`load_record()\`. Save the optimizer record next to \`weights.bpk\` (atomically) and load it on resume. Add a test that two half-runs match one full run."

  issue "KV cache for fast generation" \
"Generation re-runs the whole context window for every new byte. Add a key/value cache so each new token costs one position instead of the full window, and benchmark the speed-up. Probably needs a separate inference forward path next to the training one."

  issue "Gradient accumulation" \
"Let users train with an effective batch larger than memory allows: add \`--accum N\`, accumulate N micro-batch gradients before one optimizer step, scale the loss, and keep logging per optimizer step. Test that accum=2 with batch=B matches accum=1 with batch=2B up to float error."

  issue "Global-norm gradient clipping" \
"Burn's built-in \`GradientClippingConfig::Norm\` clips each parameter tensor separately. Standard transformer training clips the global norm over all parameters. Implement a global-norm clip on \`GradientsParams\` (or contribute it upstream to Burn) and compare training curves."

  issue "Export weights to safetensors (Hugging Face compatible)" \
"Export a trained model to \`.safetensors\` with tensor names that match a common GPT/Llama layout so it can be loaded by other tools, plus a \`config.json\`. Document the name mapping, and add an import path. Note that the byte-level vocabulary needs a matching tokenizer file."

  issue "Graphical interface (egui): live loss curve, sample viewer, start/stop" \
"A desktop window around the library: pick a text file and a model size, press Start, watch the loss curve, validation and samples update live, stop and resume. \`germen_core::train::train\` already reports progress through \`Event\` callbacks, so the GUI needs no changes in the core."

  issue "Translate the command-line messages (i18n) and the README" \
"All CLI messages are in Romanian. Introduce a small message catalogue (no heavy dependency), add English first, then the remaining languages used across the project family, and add a README per language."

  issue "Optional GPU backend behind a Cargo feature" \
"The default build is CPU-only on purpose (no drivers, no toolchains). Add an opt-in \`wgpu\` feature (and \`--device\` flag) so people with a GPU can train bigger models, while the default binary stays dependency-free. Measure with \`germen bench\` and record the numbers."

  issue "Community benchmarks: collect \`germen bench --json\` from many CPUs" \
"Create \`docs/BENCHMARKS.md\` with a table of results (CPU model, threads, SIMD, tokens/s for nano/micro/mini/small) and invite contributions as pull requests that add a JSON file under \`benchmarks/\`. A small script can render the table from those files."
fi

echo
echo "Gata: https://github.com/$OWNER/$REPO"
echo "Pentru o versiune: git tag v0.1.0-alpha.1 && git push --tags  (CI-ul construiește binarele)."
