create() {

    echo 'create_start' > /tmp/create_start

    if [ -z "$1" ]; then
        echo "Usage: create <filename>"
        return
    fi

    # Check if the file already exists
    if [ -e "$1" ]; then
        echo "Error: File '$1' already exists."
		open "$1"
        return
    fi

    # Check the file isn't a directory
    if [ -d "$1" ]; then
        echo "Error: $1 is a directory already."
        return
    fi

    # Check the parent directory exists
    if [ ! -d "$(dirname "$1")" ]; then
        echo "Error: parent directory '$(dirname "$1")' does not exist."
        return
    fi

    # Create the file an empty new line
    if ! printf "\n" > "$1"; then
        echo "Error: Failed to create file '$1'."
        return
    fi

    # Use the existing open command to open the created file
    open "$1"

    echo 'create_end' > /tmp/create_end

}
