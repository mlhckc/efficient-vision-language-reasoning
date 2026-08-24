#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root_dir=$(CDPATH= cd -- "$script_dir/.." && pwd -P)

chapters=(
  "chapters/introduction.tex"
  "chapters/theory.tex"
  "chapters/methodology.tex"
  "chapters/technical.tex"
  "chapters/technical2.tex"
  "chapters/slm.tex"
  "chapters/efficiency.tex"
  "chapters/discussion.tex"
  "chapters/conclusions.tex"
)

if [[ ! -x /usr/bin/detex ]]; then
  printf 'error: /usr/bin/detex is missing or not executable\n' >&2
  exit 1
fi
if [[ ! -x /usr/bin/wc ]]; then
  printf 'error: /usr/bin/wc is missing or not executable\n' >&2
  exit 1
fi

for relative_path in "${chapters[@]}"; do
  chapter_path="$root_dir/$relative_path"
  if [[ ! -f "$chapter_path" || ! -r "$chapter_path" ]]; then
    printf 'error: missing or unreadable chapter: %s\n' "$chapter_path" >&2
    exit 1
  fi
done

total=0
for relative_path in "${chapters[@]}"; do
  chapter_path="$root_dir/$relative_path"
  count=$(/usr/bin/detex "$chapter_path" | /usr/bin/wc -w)
  if [[ ! $count =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]]; then
    printf 'error: invalid word count for %s: %s\n' "$relative_path" "$count" >&2
    exit 1
  fi
  count="${count//[[:space:]]/}"
  total=$((total + count))
  printf '%s\t%s\n' "$relative_path" "$count"
done

printf 'TOTAL\t%s\n' "$total"
