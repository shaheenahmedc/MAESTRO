import shlex

from inspect_ai.tool import Tool, tool

from .utils import run_bash_for_swe_agent_tool


@tool(parallel=False)
def edit_file() -> Tool:
    async def edit_file(start_line: int, end_line: int, replacement_text: str) -> str:
        """Replaces lines <start_line> through <end_line> (inclusive) with the given text in the open file. Python files will be checked for syntax errors after the edit. If the system detects a syntax error, the edit will not be executed. Simply try to edit the file again, but make sure to read the error message and modify the edit command you issue accordingly. Issuing the same command a second time will just lead to the same error message again.

        Args:
            start_line (int): the line number to start the edit at.
            end_line (int): the line number to end the edit at (inclusive).
            replacement_text (str): the text to replace the current selection with. This must have correct whitespace (be particularly careful with indentation).

        Returns:
        str: the output of the edit command.
        """
        tool_command = f"edit {shlex.quote(f'{start_line}:{end_line}')} << 'end_of_edit584' \n{replacement_text}\nend_of_edit584"

        return await run_bash_for_swe_agent_tool(tool_command)

    return edit_file
