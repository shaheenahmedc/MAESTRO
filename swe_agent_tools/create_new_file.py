import shlex

from inspect_ai.tool import Tool, tool

from .utils import run_bash_for_swe_agent_tool


@tool(parallel=False)
def create_new_file() -> Tool:
    async def create_new_file(path: str) -> str:
        """Creates and opens a new file at the given path in the editor.

        Args:
            path (str): the path to the file to create
        Returns:
        str: the output of the create command.
        """
        command_to_execute = f"create {shlex.quote(path)}"

        return await run_bash_for_swe_agent_tool(command_to_execute)

    return create_new_file
