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

# The version comes from Cargo.toml, so the commit message and the tag hint never go stale.
VERSION=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')

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
    git commit -m "GERMEN $VERSION: train a small language model from scratch in pure Rust"
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

if [ "$WITH_ISSUES" = "1" ] && [ -n "$(gh issue list --repo "$OWNER/$REPO" --state all --limit 1 --json number --jq '.[].number' 2>/dev/null)" ]; then
  echo "Depozitul are deja probleme (issues); nu mai creez altele."
  WITH_ISSUES=0
fi

if [ "$WITH_ISSUES" = "1" ]; then
  issue() { gh issue create --repo "$OWNER/$REPO" --title "$1" --body "$2" >/dev/null && echo "  problemă creată: $1"; }

  issue "Use more cores in the non-matmul part of the training step" \
"Matrix products scale well with threads, but RMSNorm, rotary embeddings and the AdamW update hardly do (see docs/PLAN.md, section 4). On a machine with many cores they become the bottleneck. Ideas: a parallel AdamW update over parameter tensors, fused RMSNorm / rotary kernels, fewer tensor copies. Start from \`germen bench --scaling --json\` and the \`profile_*\` examples, and show the curve before and after."

  issue "Memory-mapped corpora larger than RAM" \
"The whole corpus is read into memory as 16-bit token ids. Add a memory-mapped, tokenised corpus on disk so a model can be trained on text larger than the available RAM; keep the train/validation split and the BPE-from-the-training-part rule."

  issue "Export weights to safetensors (Hugging Face compatible)" \
"Export a trained model to \`.safetensors\` with tensor names that match a common GPT/Llama layout so it can be loaded by other tools, plus a \`config.json\`. Document the name mapping, and add an import path. Note that the tokenizer (bytes or BPE, stored in \`tokenizer.json\`) needs a matching file."

  issue "Graphical interface (egui): live loss curve, sample viewer, start/stop" \
"A desktop window around the library: pick a text file and a model size, press Start, watch the loss curve, validation and samples update live, stop and resume. \`germen_core::train::train\` already reports progress through \`Event\` callbacks, so the GUI needs no changes in the core."

  issue "Translate the command-line messages (i18n)" \
"All CLI messages are in Romanian (the READMEs already exist in eight languages). Introduce a small message catalogue (no heavy dependency), add English first, then the remaining languages used across the project family."

  issue "Optional GPU backend behind a Cargo feature" \
"The default build is CPU-only on purpose (no drivers, no toolchains). Add an opt-in \`wgpu\` feature (and \`--device\` flag) so people with a GPU can train bigger models, while the default binary stays dependency-free. Measure with \`germen bench\` and record the numbers."

  issue "Community benchmarks: collect \`germen bench --scaling --json\` from many CPUs" \
"Create \`docs/BENCHMARKS.md\` with a table of results (CPU model, threads, SIMD, tokens/s for nano/micro/mini/small, and the speed-up curve from \`--scaling\`) and invite contributions as pull requests that add a JSON file under \`benchmarks/\`. A small script can render the table from those files."
fi

echo
echo "Gata: https://github.com/$OWNER/$REPO"
echo "Pentru o versiune: git tag v$VERSION-alpha.1 && git push --tags  (CI-ul construiește binarele)."
