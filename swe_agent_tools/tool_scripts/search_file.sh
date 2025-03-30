
# @yaml
# signature: search_file <search_term> [<file>]
# docstring: searches for search_term in file. If file is not provided, searches in the current open file
# arguments:
#   search_term:
#     type: string
#     description: the term to search for
#     required: true
#   file:
#     type: string
#     description: the file to search in (if not provided, searches in the current open file)
#     required: false
search_file() {
    # Check if the first argument is provided
    if [ -z "$1" ]; then
        echo "Usage: search_file <search_term> [<file>]"
        return
    fi
    # Check if the second argument is provided
    if [ -n "$2" ]; then
        # Check if the provided argument is a valid file
        if [ -f "$2" ]; then
            if [ ! -r "$2" ]; then
                echo "Error: Permission denied. You do not have read access to $2."
                return
            else
                local file="$2"  # Set file if valid
            fi
        else
            echo "Usage: search_file <search_term> [<file>]"
            echo "Error: File name $2 not found. Please provide a valid file name."
            return  # Exit if the file is not valid
        fi
    else
        # Check if a file is open
        if [ -z "$CURRENT_FILE" ]; then
            echo "No file open. Use the open command first."
            return  # Exit if no file is open
        fi
        local file="$CURRENT_FILE"  # Set file to the current open file
    fi
    local search_term="$1"
    file=$(realpath "$file")
    # Use grep to directly get the desired formatted output
    local matches=$(grep -nH -- "$search_term" "$file")
    # Check if no matches were found
    if [ -z "$matches" ]; then
        echo "No matches found for \"$search_term\" in $file"
        return
    fi
    # Calculate total number of matches
    local num_matches=$(echo "$matches" | wc -l | awk '{$1=$1; print $0}')

    # calculate total number of lines matched
    local num_lines=$(echo "$matches" | cut -d: -f1 | sort | uniq | wc -l | awk '{$1=$1; print $0}')
    # if num_lines is > 100, print an error
    if [ $num_lines -gt 100 ]; then
        echo "More than $num_lines lines matched for \"$search_term\" in $file. Please narrow your search."
        return
    fi

    # Print the total number of matches and the matches themselves
    echo "Found $num_matches matches for \"$search_term\" in $file:"
    echo "$matches" | cut -d: -f1-2 | sort -u -t: -k2,2n | while IFS=: read -r filename line_number; do
        echo "Line $line_number:$(sed -n "${line_number}p" "$file")"
    done
    echo "End of matches for \"$search_term\" in $file"
}