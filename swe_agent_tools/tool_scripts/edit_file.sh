# @yaml
# signature: |-
#   edit <start_line>:<end_line>
#   <replacement_text>
#   end_of_edit
# docstring: replaces lines <start_line> through <end_line> (inclusive) with the given text in the open file. The replacement text is terminated by a line with only end_of_edit on it. All of the <replacement text> will be entered, so make sure your indentation is formatted properly. Python files will be checked for syntax errors after the edit. If the system detects a syntax error, the edit will not be executed. Simply try to edit the file again, but make sure to read the error message and modify the edit command you issue accordingly. Issuing the same command a second time will just lead to the same error message again.
# end_name: end_of_edit
# arguments:
#   start_line:
#     type: integer
#     description: the line number to start the edit at
#     required: true
#   end_line:
#     type: integer
#     description: the line number to end the edit at (inclusive)
#     required: true
#   replacement_text:
#     type: string
#     description: the text to replace the current selection with
#     required: true
edit() {
    if [ -z "$CURRENT_FILE" ]
    then
        echo 'No file open. Use the `open` command first.'
        return
    fi

    

    local start_line="$(echo $1: | cut -d: -f1)"
    local end_line="$(echo $1: | cut -d: -f2)"

    local re='^[0-9]+$'
    if ! [[ $start_line =~ $re ]]; then
        echo "Usage: edit <start_line>:<end_line>"
        echo "Error: start_line must be a number"
        return
    fi
    if ! [[ $end_line =~ $re ]]; then
        echo "Usage: edit <start_line>:<end_line>"
        echo "Error: end_line must be a number"
        return
    fi

    # Add this validation
    if [ "$start_line" -lt 1 ] || [ "$end_line" -lt 1 ]; then
        echo "Error: start_line and end_line must be at least 1"
        return
    fi

    # Check if the file does not exist
    if [ ! -f "$CURRENT_FILE" ]; then
        echo "Error: $CURRENT_FILE not found. Maybe it was delete or moved since you opened it?"
        return
    fi

    if [ ! -w "$CURRENT_FILE" ]; then
        echo "Error: Permission denied. You do not have write access to $CURRENT_FILE."
        return
    fi

    local linter_cmd="flake8 --isolated --select=F821,F822,F831,E111,E112,E113,E999,E902"
    local linter_before_edit=$($linter_cmd "$CURRENT_FILE" 2>&1)

    # Bash array starts at 0, so let's adjust
    local start_line=$((start_line - 1))
    local end_line=$((end_line))

    local line_count=0
    local replacement=()
    while IFS= read -r line
    do
        replacement+=("$line")
        ((line_count++))
    done

    # Create a backup of the current file
    cp "$CURRENT_FILE" "/tmp/$(basename "$CURRENT_FILE")_backup"

    # Read the file line by line into an array
    mapfile -t lines < "$CURRENT_FILE"
    local new_lines=("${lines[@]:0:$start_line}" "${replacement[@]}" "${lines[@]:$((end_line))}")
    # Write the new stuff directly back into the original file
    printf "%s\n" "${new_lines[@]}" >| "$CURRENT_FILE"

    # Run linter
    if [[ $CURRENT_FILE == *.py ]]; then
        _lint_output=$($linter_cmd "$CURRENT_FILE" 2>&1)
        lint_output=$({SPLIT_STRING_LOCATION_SANDBOX} "$_lint_output" "$linter_before_edit" "$((start_line+1))" "$end_line" "$line_count")
    else
        # do nothing
        lint_output=""
    fi

    # if there is no output, then the file is good
    if [ -z "$lint_output" ]; then
        export CURRENT_LINE=$start_line
        _constrain_line
        _print

        echo "File updated. Please review the changes and make sure they are correct (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
    else
        echo "Your proposed edit has introduced new syntax error(s). Please read this error message carefully and then retry editing the file."
        echo ""
        echo "ERRORS:"
        echo "$lint_output"
        echo ""

        # Save original values
        original_current_line=$CURRENT_LINE
        original_window=$WINDOW

        # Update values
        export CURRENT_LINE=$(( (line_count / 2) + start_line )) # Set to "center" of edit
        export WINDOW=$((line_count + 10)) # Show +/- 5 lines around edit

        echo "This is how your edit would have looked if applied"
        echo "-------------------------------------------------"
        _constrain_line
        _print
        echo "-------------------------------------------------"
        echo ""

        # Restoring CURRENT_FILE to original contents.
        cp "/tmp/$(basename "$CURRENT_FILE")_backup" "$CURRENT_FILE"

        export CURRENT_LINE=$(( ((end_line - start_line + 1) / 2) + start_line ))
        export WINDOW=$((end_line - start_line + 10))

        echo "This is the original code before your edit"
        echo "-------------------------------------------------"
        _constrain_line
        _print
        echo "-------------------------------------------------"

        # Restore original values
        export CURRENT_LINE=$original_current_line
        export WINDOW=$original_window

        echo "Your changes have NOT been applied. Please fix your edit command and try again."
        echo "You either need to 1) Specify the correct start/end line arguments or 2) Correct your edit code."
        echo "DO NOT re-run the same failed edit command. Running it again will lead to the same error."
    fi

    # Remove backup file
    rm -f "/tmp/$(basename "$CURRENT_FILE")_backup"
}