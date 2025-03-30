from inspect_ai.tool import Tool, tool

from .utils import run_bash_for_swe_agent_tool


@tool(parallel=False)
def scroll_up() -> Tool:
    async def scroll_up() -> str:
        """Moves the window up in the current file.

        Returns:
        str: the window into the file after scrolling up.
        """
        return await run_bash_for_swe_agent_tool("scroll_up")

    return scroll_up
