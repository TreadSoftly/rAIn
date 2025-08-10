#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET="$HERE"
case ":$PATH:" in *":$TARGET:"*) echo "Already on PATH."; exit 0;; esac

read -rp "Add '$TARGET' to PATH in your shell profile? [Y/n] " a
a="${a:-Y}"; a="${a,,}"
if [[ "$a" != y && "$a" != yes ]]; then
  echo "Skipped."
  exit 0
fi

# Pick a profile
PROFILE=""
if [[ -f "$HOME/.zshrc" ]]; then PROFILE="$HOME/.zshrc"
elif [[ -f "$HOME/.bashrc" ]]; then PROFILE="$HOME/.bashrc"
else PROFILE="$HOME/.profile"; fi

echo '' >> "$PROFILE"
echo '# rAIn installers on PATH' >> "$PROFILE"
echo "export PATH=\"\$PATH:$TARGET\"" >> "$PROFILE"

echo "✅ Added to PATH in $PROFILE"
echo "→ Open a new terminal (or 'source' your profile) to use:  run / argos"
