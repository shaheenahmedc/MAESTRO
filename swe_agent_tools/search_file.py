import shlex

from inspect_ai.tool import Tool, tool

from .utils import run_bash_for_swe_agent_tool


@tool(parallel=False)
def search_file() -> Tool:
    # Note, Inspect doesn't support optional parameters, so I've added $ as a workaround
    async def search_file(search_term: str, file: str) -> str:
        """Searches for search_term in file. If the special filename $ is provided, search the current open file.

        Args:
            search_term (str): the term to search for.
            file (str): the file to search in, or $ to search the current open file.

        Returns:
        str: the search results
        """
        if file == "$":
            tool_command = f"search_file {shlex.quote(search_term)}"
        else:
            tool_command = f"search_file {shlex.quote(search_term)} {shlex.quote(file)}"

        return await run_bash_for_swe_agent_tool(tool_command)

    return search_file
