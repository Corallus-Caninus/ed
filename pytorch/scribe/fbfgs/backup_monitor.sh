#!/bin/bash

# File Backup Monitor Script
# Creates timestamped backups whenever the source file is modified

# Configuration
SOURCE_FILE="/path/to/your/file.txt"  # Change this to your file path
BACKUP_DIR="/path/to/backups"         # Change this to your backup directory
CHECK_INTERVAL=60                    # Check interval in seconds (for polling mode)

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Function to create backup
create_backup() {
    # Get current timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    # Create unique filename
    BACKUP_FILE="${BACKUP_DIR}/$(basename "$SOURCE_FILE")_${TIMESTAMP}"

    # Copy the file
    cp "$SOURCE_FILE" "$BACKUP_FILE"

    # Optional: Add a message to the backup
    echo "# Backup created at $(date)" >> "$BACKUP_FILE"

    echo "Created backup: $BACKUP_FILE"
}

# Check if inotify-tools is available
if command -v inotifywait &> /dev/null; then
    echo "Using inotify for real-time monitoring..."

    # Initial backup
    create_backup

    # Monitor the file for changes using inotifywait
    while true; do
        inotifywait -e modify "$SOURCE_FILE"
        create_backup
    done
else
    echo "inotify-tools not found, using polling mode (every ${CHECK_INTERVAL} seconds)..."

    # Store initial modification time
    LAST_MOD=$(stat -c %Y "$SOURCE_FILE")

    # Initial backup
    create_backup

    # Polling loop
    while true; do
        CURRENT_MOD=$(stat -c %Y "$SOURCE_FILE")

        if [ "$CURRENT_MOD" != "$LAST_MOD" ]; then
            create_backup
            LAST_MOD=$CURRENT_MOD
        fi

        sleep $CHECK_INTERVAL
    done
fi
