#!/bin/bash
# isync.sh - Interactive sync: REMOTE → LOCAL
# Pulls changes from gbar:~/potts_sim/ to local directory

source "$(dirname "$0")/sync_common.sh"

do_sync "pull" "$(pwd)"
