#!/usr/bin/env bash
set -euo pipefail

# Directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for DIR in results logs; do
    TARGET="$SCRIPT_DIR/$DIR"
    if [[ -d "$TARGET" ]]; then
        echo "Clearing contents of $TARGET"
        # Remove all files (including hidden) but keep the directory
        find "$TARGET" -mindepth 1 -delete
    else
        echo "Warning: $TARGET does not exist"
    fi
done