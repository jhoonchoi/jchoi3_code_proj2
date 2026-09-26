#!/usr/bin/env bash
# Verify this code folder is ready to zip for Canvas.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

fail=0

if [[ -f main.py ]]; then
  echo "OK: main.py present"
else
  echo "FAIL: main.py not found — the course guide requires a main.py entry point"
  fail=1
fi

if [[ -f README.md ]]; then
  echo "OK: README.md present"
else
  echo "FAIL: README.md not found"
  fail=1
fi

junk=$(find . \( -name "__pycache__" -o -name "*.pyc" -o -name ".ipynb_checkpoints" -o -name ".DS_Store" \) 2>/dev/null)
if [[ -n "$junk" ]]; then
  echo "NOTE: found generated files package.sh will exclude from the zip:"
  echo "$junk"
fi

echo
if [[ $fail -eq 0 ]]; then
  echo "Ready to zip — run ./package.sh"
else
  echo "Fix the issues above first."
fi
exit $fail
