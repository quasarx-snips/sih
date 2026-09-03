#!/bin/bash

# Set to 99M (GitHub's hard limit is 100M per file)
MAX_SIZE=${1:-"99M"}
COMMIT_MSG=${2:-"Commit files up to GitHub's size limit ($MAX_SIZE)"}

echo "Scanning for files larger than $MAX_SIZE..."

# Find files exceeding the size limit, excluding the .git directory
find . -type f -not -path '*/.git*' -size +$MAX_SIZE | while read -r file; do
    # Strip leading './' for cleaner .gitignore entries
    file_path="${file#./}"

    # Append to .gitignore if not already listed
    if ! grep -qxF "$file_path" .gitignore 2>/dev/null; then
        echo "$file_path" >> .gitignore
        echo "Ignored: $file_path"
    fi

    # Remove from Git tracking if it was previously committed
    if git ls-files --error-match --silent "$file_path" 2>/dev/null; then
        git rm --cached "$file_path"
        echo "Untracked: $file_path"
    fi
done

echo "Staging remaining files..."
git add .

echo "Committing changes..."
git commit -m "$COMMIT_MSG"

echo "Pushing to GitHub (https://github.com/quasarx-snips/sih)..."
git push origin main