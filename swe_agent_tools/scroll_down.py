from inspect_ai.tool import Tool, tool

from .utils import run_bash_for_swe_agent_tool


@tool(parallel=False)
def scroll_down() -> Tool:
    async def scroll_down() -> str:
        """Moves the window down in the current file.

        Returns:
        str: the window into the file after scrolling down.
        """
        return await run_bash_for_swe_agent_tool("scroll_down")

    return scroll_down
