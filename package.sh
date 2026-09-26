#!/usr/bin/env bash
# Zip this code folder for Canvas, matching the course's
# andrewid_code_projX.zip convention (folder name == zip basename).
# Written two levels up (outside projN/, not just outside this folder) so
# the zip never ends up inside the webpage folder that gets uploaded to AFS.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDER_NAME="$(basename "$SCRIPT_DIR")"
DEST="$SCRIPT_DIR/../../${FOLDER_NAME}.zip"

rm -f "$DEST"
cd "$SCRIPT_DIR/.."
zip -rq "$DEST" "$FOLDER_NAME" \
  -x "*/__pycache__/*" "*.pyc" "*/.ipynb_checkpoints/*" "*/.DS_Store"

size_mb=$(du -m "$DEST" | cut -f1)
echo "Created $DEST (${size_mb} MB)"
if (( size_mb > 100 )); then
  echo "Warning: that's ${size_mb} MB — make sure you're not zipping checkpoints/datasets that don't belong in the Canvas submission."
fi
