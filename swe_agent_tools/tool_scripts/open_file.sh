open() {
    if [ -z "$1" ]; then
        echo "Usage: open \"<file>\" [<line_number>]"
        return
    fi

    # Check if the file exists before proceeding
    if [ ! -f "$1" ]; then
        if [ -d "$1" ]; then
            echo "Error: $1 is a directory. You can only open files. Use cd or ls to navigate directories."
        else
            echo "File $1 not found"
        fi
        return
    fi

    if [ ! -r "$1" ]; then
        echo "Error: Permission denied. You do not have read access to $1."
        return
    fi

    if ! file "$1" | grep -q -E '(empty|text|very short file)'; then
        echo "Error: $1 is not a readable text file."
        return
    fi

    # Now it's safe to perform operations on the file
    if [ -n "$2" ]; then
        # Check if the provided argument is a valid number
        if ! [[ $2 =~ ^[0-9]+$ ]]; then
            echo "Usage: open \"<file>\" [<line_number>]"
            echo "Error: <line_number> must be a number"
            return  # Exit if the line number is not valid
        fi
        local max_line=$(awk 'END {print NR}' "$1")
        if [ $2 -gt $max_line ]; then
            echo "Warning: <line_number> ($2) is greater than the number of lines in the file ($max_line)"
            echo "Warning: Setting <line_number> to $max_line"
            local line_number=$(jq -n "$max_line")  # Set line number to max if greater than max
        elif [ $2 -lt 1 ]; then
            echo "Warning: <line_number> ($2) is less than 1"
            echo "Warning: Setting <line_number> to 1"
            local line_number=$(jq -n "1")  # Set line number to 1 if less than 1
        else
            local OFFSET=$(jq -n "$WINDOW/6" | jq 'floor')
            local line_number=$(jq -n "[$2 + $WINDOW/2 - $OFFSET, 1] | max | floor")
        fi
    else
        local line_number=$(jq -n "$WINDOW/2")  # Set default line number if not provided
    fi

    export CURRENT_FILE=$(realpath "$1")
    export CURRENT_LINE=$line_number
    _constrain_line
    _print
}
