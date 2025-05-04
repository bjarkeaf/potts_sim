#!/bin/bash

PASSFILE="$HOME/.ssh/gbarpass"
RSYNC_RSH="sshpass -f \"$PASSFILE\" ssh -i ~/.ssh/gbar"

# Sync local -> remote
rsync -avz --exclude={'osync.sh','isync.sh'} -e "$RSYNC_RSH" "$(pwd)/" "gbar:~/potts_sim/"
