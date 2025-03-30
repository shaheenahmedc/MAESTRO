import shlex

from inspect_ai.tool import Tool, tool

from .utils import run_bash_for_swe_agent_tool


@tool(parallel=False)
def find_file() -> Tool:
    async def find_file(file_name: str, dir: str) -> str:
        """Finds all files with the given name in dir, recursively. If the special dir $ is provided, searches in the current directory

        Args:
            file_name (str): the name of the file to search for.
            dir (str): the directory to search in, or $ to search in the current directory

        Returns:
        str: the search results
        """
        if dir == "$":
            tool_command = f"find_file {shlex.quote(file_name)}"
        else:
            tool_command = f"find_file {shlex.quote(file_name)} {shlex.quote(dir)}"

        return await run_bash_for_swe_agent_tool(tool_command)

    return find_file
