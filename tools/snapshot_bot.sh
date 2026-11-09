#!/bin/sh
# Copy bot/ to DEST with its package renamed, so an old version of the bot can
# be loaded next to the current one (tools/ab_code.py).  usage: snapshot_bot.sh DEST NAME
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$1"; NAME="$2"
rm -rf "$DEST"; mkdir -p "$DEST"
rsync -a --exclude __pycache__ "$ROOT/bot/" "$DEST/"
mv "$DEST/pb" "$DEST/$NAME"
sed -i '' "s/from pb import/from $NAME import/; s/import pb\b/import $NAME/" "$DEST/agent.py"
echo "snapshot at $DEST (package $NAME)"
