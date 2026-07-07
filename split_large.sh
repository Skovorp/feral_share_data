#!/usr/bin/env bash
# Maintainer helper: (re)split any file over the size limit into 45MB parts.
# Run this after updating/adding a large data file, before committing:
#   ./split_large.sh
#
# It scans the repo for files > 50MB, splits each into <file>.part_aa,
# <file>.part_ab, ... and removes stale parts. The originals are git-ignored
# (see .gitignore); commit the resulting *.part_* files.
set -euo pipefail
cd "$(dirname "$0")"

LIMIT=$((50 * 1024 * 1024))   # 50 MB — GitHub's recommended max
CHUNK=45m                     # part size, safely under the limit

# Find candidate files, skipping .git and existing parts.
while IFS= read -r f; do
  sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$sz" -gt "$LIMIT" ]; then
    echo "splitting $f ($((sz/1048576))MB)"
    rm -f "$f".part_*
    split -b "$CHUNK" "$f" "$f".part_
  fi
done < <(find . -type f -not -path './.git/*' -not -name '*.part_*')

echo "Done. Review with 'git status', then commit the *.part_* files."
echo "Make sure each large original is listed in .gitignore."
