#!/bin/bash

# Input file
FILE="app.py"

# Create a temp file
TMPFILE=$(mktemp)

# Process the file to remove help parameters
awk '{
    # If the line contains help=", remove the parameter
    if ($0 ~ /help="/) {
        # Remove help="..." leaving the rest of the line intact
        gsub(/help="[^"]*"/, "", $0)
        
        # Remove trailing comma if it became the last parameter
        gsub(/,[ \t]*\)/, ")", $0)
        
        # Clean up any double spaces that may have been created
        gsub(/  +/, " ", $0)
    }
    print $0
}' "$FILE" > "$TMPFILE"

# Replace the original file
mv "$TMPFILE" "$FILE"

echo "Removed help tooltips from $FILE"
