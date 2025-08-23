#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$HERE"

case ":$PATH:" in *":$TARGET:"*) : ;; *)
  export PATH="$TARGET:$PATH"
esac

ANS="${1-}"
if [[ "$ANS" == "-y" || -n "${CI-}" ]]; then
  REPLY="y"
else
  read -rp "Add '$TARGET' to PATH in your shell profile (prepended)? [y/n] " REPLY
  REPLY="${REPLY:-Y}"
fi

REPLY_LC="$(printf '%s' "$REPLY" | tr '[:upper:]' '[:lower:]')"
if [[ "$REPLY_LC" != "y" && "$REPLY_LC" != "yes" ]]; then
  echo "Skipped."
  exit 0
fi

# Session already updated above. Now persist in profile (prepend).
PROFILE=""
if [[ -f "$HOME/.zshrc" ]]; then PROFILE="$HOME/.zshrc"
elif [[ -f "$HOME/.bashrc" ]]; then PROFILE="$HOME/.bashrc"
else PROFILE="$HOME/.profile"; fi

if ! grep -Fq '# rAIn installers on PATH (prepend)' "$PROFILE" 2>/dev/null; then
  {
    echo ''
    echo '# rAIn installers on PATH (prepend)'
    echo 'export PATH="'"$TARGET"':$PATH"'
  } >> "$PROFILE"
fi

echo " Added to PATH in $PROFILE (prepended)."
echo " Open a new terminal or 'source' your profile to use:"
echo "   • build, all, argos"
echo "   • d/detect, hm/heatmap, gj/geojson, lv/livevideo"
echo "   • classify/clf, pose/pse, obb/object"
