#!/usr/bin/env bash
# Rebuild the original large files from their split parts.
# Run this once after cloning:  ./reassemble.sh
#
# Files larger than GitHub's 50MB limit are committed as 45MB chunks named
# <file>.part_aa, <file>.part_ab, ...  This script concatenates each set of
# parts back into the original file. It is safe to run repeatedly.
set -euo pipefail
cd "$(dirname "$0")"

shopt -s nullglob 2>/dev/null || true
found=0
while IFS= read -r first; do
  found=1
  base="${first%.part_aa}"
  printf 'reassembling %s ... ' "$base"
  cat "${base}".part_* > "${base}"
  printf 'done (%s)\n' "$(du -h "${base}" | cut -f1)"
done < <(find . -name '*.part_aa' | sort)

if [ "$found" -eq 0 ]; then
  echo "No .part_aa files found — nothing to reassemble."
fi
echo "All originals rebuilt. They are git-ignored, so 'git status' stays clean."
