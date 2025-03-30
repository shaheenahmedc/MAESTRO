import shlex

from inspect_ai.tool import Tool, tool

from .utils import run_bash_for_swe_agent_tool


@tool(parallel=False)
def open_file() -> Tool:
    async def open_file(path: str, line_number: int = -1) -> str:
        """Opens the file at the given path in the editor. If line_number is provided, the window will be move to include that line.

        Args:
            path (str): the path to the file to open.
            line_number (int): the line number to move the window to (if not provided, or set to -1 will be set to the middle of the file)

        Returns:
        str: the output of the open command.
        """
        if line_number < -1:
            return "Error: line_number must be a positive integer"

        command_to_execute = f"open {shlex.quote(path)} {shlex.quote(str(line_number)) if line_number != -1 else ''}"

        return await run_bash_for_swe_agent_tool(command_to_execute)

    return open_file
