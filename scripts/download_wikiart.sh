#!/usr/bin/env bash
# Download + unzip the Kaggle WikiArt dataset into a (shared) directory.
# Usage: bash scripts/download_wikiart.sh /path/to/public/datasets
# Needs a Kaggle API token in ~/.kaggle/kaggle.json (see README).
set -euo pipefail
DEST="${1:?usage: bash scripts/download_wikiart.sh /path/to/public/dir}"
mkdir -p "$DEST"
command -v kaggle >/dev/null || pip install --user kaggle
export PATH="$HOME/.local/bin:$PATH"

echo ">> Downloading steubk/wikiart to $DEST (large: tens of GB, be patient)"
kaggle datasets download -d steubk/wikiart -p "$DEST"

ZIP="$(ls "$DEST"/*.zip | head -n1)"
echo ">> Unzipping $ZIP -> $DEST/wikiart"
mkdir -p "$DEST/wikiart"
unzip -q -o "$ZIP" -d "$DEST/wikiart"
rm -f "$ZIP"   # comment this out to keep the zip

# make it readable by others (remove if you don't want to share it)
chmod -R a+rX "$DEST/wikiart" || true
echo ">> Done. Style folders:"
ls "$DEST/wikiart" | head -n 30
