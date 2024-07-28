#!/bin/bash

FOLDER=$1
cd $FOLDER
## clear the input
echo "# Labs" > "labs.md"

# Find all .ipynb files excluding those in the ./_build/* path, remove leading "./",
# then sort them to ensure they are grouped by directory.
find . -name "*.ipynb" ! -path "./_build/*" | sed 's|^\./||' | sort | \
while read -r line; do
    # Extract the directory and filename without extension
    dir=$(dirname "$line")
    filename=$(basename "$line" .ipynb)

    # Get only the first two levels of the directory
    # Assuming a leading "./" has already been removed, so we're working with paths like "level1/level2/..."
    dir_level=$(echo "$dir" | awk -F'/' '{print $1"/"$2}')

    # If the current directory level is different from the last one processed, print it as a heading
    if [[ "$prev_dir_level" != "$dir_level" ]]; then
        # Convert directory path to a Markdown header. Ensure to handle cases where there might not be a second level.
        if [[ "$dir_level" == */* ]]; then
            echo -e "\n## $dir_level\n" >> labs.md
        else
            echo -e "\n## $dir_level/\n">> labs.md
        fi
        prev_dir_level="$dir_level"
    fi

    # Print the file link in Markdown format
    # Adjust the printed path to match the actual location if necessary
    echo "* [$filename]($line)" >> labs.md
done
