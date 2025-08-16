#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$HERE"

case ":$PATH:" in *":$TARGET:"*) echo "Already on PATH."; exit 0;; esac

ANS="${1-}"
if [[ "$ANS" == "-y" || -n "${CI-}" ]]; then
  REPLY="y"
else
  read -rp "Add '$TARGET' to PATH in your shell profile? [y/n] " REPLY
  REPLY="${REPLY:-Y}"
fi
REPLY="${REPLY,,}"
if [[ "$REPLY" != "y" && "$REPLY" != "yes" ]]; then
  echo "Skipped."
  exit 0
fi

# Session update (current shell)
export PATH="$PATH:$TARGET"

# Profile update
PROFILE=""
if [[ -f "$HOME/.zshrc" ]]; then PROFILE="$HOME/.zshrc"
elif [[ -f "$HOME/.bashrc" ]]; then PROFILE="$HOME/.bashrc"
else PROFILE="$HOME/.profile"; fi

if ! grep -Fq '# rAIn installers on PATH' "$PROFILE" 2>/dev/null; then
  {
    echo ''
    echo '# rAIn installers on PATH'
    echo 'export PATH="$PATH:'"$TARGET"'"'
  } >> "$PROFILE"
fi

echo " Added to PATH in $PROFILE"
echo " Open a new terminal (or 'source' your profile) to use: build, all, argos, d/detect, hm/heatmap, gj/geojson, lv/livevideo"