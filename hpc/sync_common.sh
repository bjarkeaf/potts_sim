#!/bin/bash
# sync_common.sh - Shared functions for isync.sh and osync.sh

set -euo pipefail

# Configuration
PASSFILE="$HOME/.ssh/gbarpass"
RSYNC_RSH="sshpass -f \"$PASSFILE\" ssh -i ~/.ssh/gbar"
REMOTE_HOST="gbar"
REMOTE_PATH="~/potts_sim/"
BACKUP_PREFIX="BACKUP."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where the scripts live (for finding sync_exclude)
get_script_dir() {
    echo "$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
}

# Build rsync base command
build_rsync_cmd() {
    local script_dir="$1"
    echo "rsync -avz --exclude-from=\"${script_dir}/sync_exclude\" -e \"$RSYNC_RSH\""
}

# Run rsync dry-run and get itemized changes
get_sync_preview() {
    local direction="$1"  # "pull" or "push"
    local script_dir="$2"
    local local_path="$3"
    
    local rsync_base
    rsync_base=$(build_rsync_cmd "$script_dir")
    
    local cmd
    if [[ "$direction" == "pull" ]]; then
        cmd="$rsync_base --dry-run --itemize-changes \"${REMOTE_HOST}:${REMOTE_PATH}\" \"${local_path}/\""
    else
        cmd="$rsync_base --dry-run --itemize-changes \"${local_path}/\" \"${REMOTE_HOST}:${REMOTE_PATH}\""
    fi
    
    eval "$cmd" 2>/dev/null || true
}

# Parse rsync itemize-changes output into new files and overwrites
# Format: YXcstpoguax path
#   Y = update type: < sent, > received, c local change, h hard link, . no update, * message
#   X = file type: f file, d directory, L symlink, etc.
#   c = checksum differs, s = size differs, t = time differs, etc.
#   +++++++++ means new file
parse_changes() {
    local -n new_ref=$1
    local -n overwrite_ref=$2
    local -n deleted_ref=$3
    
    while IFS= read -r line; do
        # Skip empty lines and directory entries
        [[ -z "$line" ]] && continue
        
        # Extract the itemize string and filename
        local itemize="${line:0:11}"
        local filename="${line:12}"
        
        # Skip directories (we only care about files)
        [[ "${itemize:1:1}" == "d" ]] && continue
        
        # Skip if no filename
        [[ -z "$filename" ]] && continue
        
        # Check if it's a new file (has +++++++++ pattern)
        if [[ "$itemize" =~ \+\+\+\+\+\+\+ ]]; then
            new_ref+=("$filename")
        # Check if it's a deletion (starts with *deleting)
        elif [[ "$line" =~ ^\*deleting ]]; then
            deleted_ref+=("${line#*deleting   }")
        # Otherwise it's an update/overwrite
        elif [[ "${itemize:0:1}" =~ [\<\>c] ]]; then
            overwrite_ref+=("$filename")
        fi
    done
}

# Display the preview of changes
display_preview() {
    local direction="$1"
    shift
    local -n new_display=$1
    shift
    local -n overwrite_display=$1
    shift
    local -n deleted_display=$1
    
    local dir_label
    if [[ "$direction" == "pull" ]]; then
        dir_label="REMOTE → LOCAL"
    else
        dir_label="LOCAL → REMOTE"
    fi
    
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Sync Preview: ${dir_label}${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo ""
    
    if [[ ${#new_display[@]} -eq 0 && ${#overwrite_display[@]} -eq 0 && ${#deleted_display[@]} -eq 0 ]]; then
        echo -e "${GREEN}✓ Everything is in sync. No changes needed.${NC}"
        return 1
    fi
    
    if [[ ${#new_display[@]} -gt 0 ]]; then
        echo -e "${GREEN}NEW FILES (will be created):${NC}"
        for f in "${new_display[@]}"; do
            echo -e "  ${GREEN}+${NC} $f"
        done
        echo ""
    fi
    
    if [[ ${#overwrite_display[@]} -gt 0 ]]; then
        echo -e "${YELLOW}OVERWRITES (will replace existing):${NC}"
        for f in "${overwrite_display[@]}"; do
            echo -e "  ${YELLOW}~${NC} $f"
        done
        echo ""
    fi
    
    if [[ ${#deleted_display[@]} -gt 0 ]]; then
        echo -e "${RED}DELETIONS (will be removed):${NC}"
        for f in "${deleted_display[@]}"; do
            echo -e "  ${RED}-${NC} $f"
        done
        echo ""
    fi
    
    echo -e "${BLUE}───────────────────────────────────────────────────────${NC}"
    echo -e "  New: ${GREEN}${#new_display[@]}${NC}  |  Overwrite: ${YELLOW}${#overwrite_display[@]}${NC}  |  Delete: ${RED}${#deleted_display[@]}${NC}"
    echo -e "${BLUE}───────────────────────────────────────────────────────${NC}"
    echo ""
    
    return 0
}

# Prompt for action (sets global SYNC_ACTION variable)
prompt_action() {
    local has_overwrites=$1
    local response
    
    if [[ "$has_overwrites" == "true" ]]; then
        printf "Proceed? [${GREEN}y${NC}]es / [${YELLOW}b${NC}]ackup first / [${RED}n${NC}]o: "
    else
        printf "Proceed? [${GREEN}y${NC}]es / [${RED}n${NC}]o: "
    fi
    
    read -r response
    SYNC_ACTION="${response,,}"  # lowercase, stored in global
}

# Create backups of files that will be overwritten
# Outputs backup paths to stdout (one per line) for capture
# Progress messages go to stderr so they display immediately
create_backups() {
    local direction="$1"
    local local_path="$2"
    shift 2
    local files=("$@")
    
    local backup_list=()
    
    echo -e "${YELLOW}Creating backups...${NC}" >&2
    
    for f in "${files[@]}"; do
        local target_path
        if [[ "$direction" == "pull" ]]; then
            target_path="${local_path}/${f}"
        else
            # For push, we need to backup on remote
            target_path="$f"
        fi
        
        if [[ "$direction" == "pull" ]]; then
            # Local backup
            if [[ -f "$target_path" ]]; then
                local dir_part=$(dirname "$target_path")
                local file_part=$(basename "$target_path")
                local backup_path="${dir_part}/${BACKUP_PREFIX}${file_part}"
                cp "$target_path" "$backup_path"
                backup_list+=("$backup_path")
                echo -e "  ${YELLOW}↳${NC} $f → ${BACKUP_PREFIX}${file_part}" >&2
            fi
        else
            # Remote backup via ssh
            local remote_file="${REMOTE_PATH}${f}"
            local dir_part=$(dirname "$f")
            local file_part=$(basename "$f")
            local backup_name="${BACKUP_PREFIX}${file_part}"
            
            # Build the remote backup command
            local ssh_cmd="sshpass -f \"$PASSFILE\" ssh -i ~/.ssh/gbar"
            eval "$ssh_cmd $REMOTE_HOST \"if [ -f '${remote_file}' ]; then cp '${remote_file}' '${REMOTE_PATH}${dir_part}/${backup_name}'; fi\"" 2>/dev/null || true
            backup_list+=("${dir_part}/${backup_name}")
            echo -e "  ${YELLOW}↳${NC} (remote) $f → ${backup_name}" >&2
        fi
    done
    
    echo "" >&2
    
    # Return backup list as newline-separated string (to stdout for capture)
    printf '%s\n' "${backup_list[@]}"
}

# Execute the actual sync
execute_sync() {
    local direction="$1"
    local script_dir="$2"
    local local_path="$3"
    
    local rsync_base
    rsync_base=$(build_rsync_cmd "$script_dir")
    
    echo -e "${BLUE}Syncing...${NC}"
    echo ""
    
    local cmd
    if [[ "$direction" == "pull" ]]; then
        cmd="$rsync_base \"${REMOTE_HOST}:${REMOTE_PATH}\" \"${local_path}/\""
    else
        cmd="$rsync_base \"${local_path}/\" \"${REMOTE_HOST}:${REMOTE_PATH}\""
    fi
    
    eval "$cmd"
    
    echo ""
    echo -e "${GREEN}✓ Sync complete!${NC}"
}

# Prompt to delete backup files
prompt_cleanup_backups() {
    local direction="$1"
    local local_path="$2"
    shift 2
    local backup_files=("$@")
    
    if [[ ${#backup_files[@]} -eq 0 ]]; then
        return
    fi
    
    echo ""
    printf "Delete backup files? [${GREEN}y${NC}]es / [${RED}n${NC}]o: "
    read -r response
    
    if [[ "${response,,}" == "y" ]]; then
        echo -e "${YELLOW}Deleting backups...${NC}"
        for f in "${backup_files[@]}"; do
            if [[ "$direction" == "pull" ]]; then
                rm -f "$f" 2>/dev/null && echo -e "  ${RED}✗${NC} Deleted: $f"
            else
                # Remote deletion
                local ssh_cmd="sshpass -f \"$PASSFILE\" ssh -i ~/.ssh/gbar"
                eval "$ssh_cmd $REMOTE_HOST \"rm -f '${REMOTE_PATH}${f}'\"" 2>/dev/null || true
                echo -e "  ${RED}✗${NC} Deleted: (remote) $f"
            fi
        done
        echo -e "${GREEN}✓ Backups cleaned up.${NC}"
    else
        echo -e "${BLUE}ℹ Backups preserved.${NC}"
    fi
}

# Main orchestration function
do_sync() {
    local direction="$1"  # "pull" or "push"
    local local_path="${2:-$(pwd)}"
    
    local script_dir
    script_dir=$(get_script_dir)
    
    # Get preview
    echo -e "${BLUE}Analyzing changes...${NC}"
    local preview
    preview=$(get_sync_preview "$direction" "$script_dir" "$local_path")
    
    # Parse into categories
    local new_files=()
    local overwrite_files=()
    local deleted_files=()
    
    parse_changes new_files overwrite_files deleted_files <<< "$preview"
    
    # Display preview
    if ! display_preview "$direction" new_files overwrite_files deleted_files; then
        exit 0
    fi
    
    # Determine if we have overwrites (affects prompt options)
    local has_overwrites="false"
    if [[ ${#overwrite_files[@]} -gt 0 ]]; then
        has_overwrites="true"
    fi
    
    # Prompt for action
    prompt_action "$has_overwrites"
    local action="$SYNC_ACTION"
    
    case "$action" in
        y|yes)
            execute_sync "$direction" "$script_dir" "$local_path"
            ;;
        b|backup)
            if [[ "$has_overwrites" == "false" ]]; then
                echo -e "${YELLOW}No files to backup. Running sync...${NC}"
                execute_sync "$direction" "$script_dir" "$local_path"
            else
                # Create backups
                local backup_list
                backup_list=$(create_backups "$direction" "$local_path" "${overwrite_files[@]}")
                
                # Convert back to array
                local backup_array=()
                while IFS= read -r line; do
                    [[ -n "$line" ]] && backup_array+=("$line")
                done <<< "$backup_list"
                
                # Execute sync
                execute_sync "$direction" "$script_dir" "$local_path"
                
                # Prompt for cleanup
                prompt_cleanup_backups "$direction" "$local_path" "${backup_array[@]}"
            fi
            ;;
        n|no|*)
            echo -e "${RED}Sync cancelled.${NC}"
            exit 0
            ;;
    esac
}
