#!/usr/bin/env bash
set -euo pipefail

# Usage: ./download_s3.sh [links_file]
# Default links_file is "links.txt" in the current directory.

# Path to file containing one S3 presigned URL per line
LINKS_FILE="${1:-links.txt}"

# Ensure the input file exists
if [[ ! -f "$LINKS_FILE" ]]; then
  echo "Error: links file '$LINKS_FILE' not found."
  exit 1
fi

# Create the target directory if necessary
TARGET_DIR="$HOME/data"
mkdir -p "$TARGET_DIR"

# Read each non-empty line and download
while IFS= read -r url; do
  # skip empty lines and comments
  [[ -z "$url" || "${url:0:1}" == "#" ]] && continue

  echo "Downloading: $url"
  wget --quiet --show-progress --directory-prefix="$TARGET_DIR" "$url"
done < "$LINKS_FILE"

echo "All downloads complete. Files are in $TARGET_DIR."
