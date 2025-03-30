scroll_up() {
    if [ -z "$CURRENT_FILE" ]
    then
        echo "No file open. Use the open command first."
        return
    fi
    export CURRENT_LINE=$(jq -n "$CURRENT_LINE - $WINDOW + $OVERLAP")
    _constrain_line
    _print
    _scroll_warning_message
}