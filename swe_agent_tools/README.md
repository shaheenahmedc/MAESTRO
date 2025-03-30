# SWE Agent Tools

This is an import of the file editing tools from [the SWE-agent paper](https://arxiv.org/abs/2405.15793). These tools are defined as bash functions, which can be  individually in each of the ./tool_scripts/ folder - definitions which have been copied from the original paper. 

The tools each function by running a bash command which contains the definitions of all the functions, and then running whichever function is needed for that particular tool. See ./utils.py for the helper functions which allow this to happen, including the 'run_bash_command' function.

NOTE: there is already a "create_file" tool elsewhere, but it's not integrated with these tools. Use this directory's "create_new_file", which will open the
file and work with these tools.

## Tools added

- edit
- open
- goto
- search_dir (check this is recursive)
- create
- search_file
- scroll_down
- scroll_up
- find_file (find all files with the given name in dir, recursively)
