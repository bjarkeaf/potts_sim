#!/bin/bash
# osync.sh - Interactive sync: LOCAL → REMOTE
# Pushes changes from local directory to gbar:~/potts_sim/

source "$(dirname "$0")/sync_common.sh"

do_sync "push" "$(pwd)"
