#!/bin/bash

PASSFILE="$HOME/.ssh/gbarpass"
RSYNC_RSH="sshpass -f \"$PASSFILE\" ssh -i ~/.ssh/gbar"

# Sync remote -> local 
rsync -avz --exclude-from="$(dirname "$0")/sync_exclude" -e "$RSYNC_RSH" "gbar:~/potts_sim/" "$(pwd)/"
