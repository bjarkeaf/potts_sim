#!/bin/bash

PASSFILE="$HOME/.ssh/gbarpass"
RSYNC_RSH="sshpass -f \"$PASSFILE\" ssh -i ~/.ssh/gbar"

# Sync remote -> local 
rsync -avz --exclude={'osync.sh','isync.sh'} -e "$RSYNC_RSH" "gbar:~/potts_sim/" "$(pwd)/"
