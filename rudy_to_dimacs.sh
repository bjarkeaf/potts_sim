#!/usr/bin/env bash
set -euo pipefail

# Usage check
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <rudy-files-directory>"
  exit 1
fi

INPUT_DIR="$1"

# Ensure the input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Error: Directory '$INPUT_DIR' does not exist."
  exit 2
fi

shopt -s nullglob
for rudy_file in "$INPUT_DIR"/*.rudy; do
  filename=$(basename "$rudy_file" .rudy)
  output_file="${filename}.col"

  # Transform:
  #  - first line: prepend 'p '
  #  - all other lines: prepend 'e '
  sed '1s/^/p /; 2,$s/^/e /' "$rudy_file" > "$output_file"

  echo "Converted '$rudy_file' → '$output_file'"
done
shopt -u nullglob

