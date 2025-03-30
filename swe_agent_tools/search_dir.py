import shlex

from inspect_ai.tool import Tool, tool

from .utils import run_bash_for_swe_agent_tool


@tool(parallel=False)
def search_dir() -> Tool:
    async def search_dir(search_term: str, dir: str) -> str:
        """Searches for search_term in all files in dir, recursively. If the special dir $ is provided, searches the current directory.

        Args:
            search_term (str): the term to search for.
            dir (str): the directory to search in, or $ to search the current directory.

        Returns:
        str: the search results
        """
        if dir == "$":
            tool_command = f"search_dir {shlex.quote(search_term)}"
        else:
            tool_command = f"search_dir {shlex.quote(search_term)} {shlex.quote(dir)}"

        return await run_bash_for_swe_agent_tool(tool_command)

    return search_dir
