#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/ckpt"

for f in piad_seen.pt piad_unseen.pt laso_seen.pt laso_unseen.pt; do
  dst="$ROOT/ckpt/$f"
  if [[ -s "$dst" ]]; then
    echo "[skip] $dst"
    continue
  fi
  echo "[download] $f"
  curl -L --fail --retry 5 --retry-delay 2 -C - \
    "https://huggingface.co/datasets/dylanorange/geal/resolve/main/$f" \
    -o "$dst"
done

ls -lh "$ROOT/ckpt"
