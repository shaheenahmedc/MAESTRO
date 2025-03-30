from typing import (
    Any,
    Callable,
    Coroutine,
    Optional,
    cast,
)


from autogen_core import CancellationToken

from autogen_core.tools import BaseTool
from inspect_ai.tool import bash
from pydantic import BaseModel, Field

from inspect_evals.swe_bench.swe_agent_tools import (
    create_new_file,
    edit_file,
    find_file,
    open_file,
    scroll_down,
    scroll_up,
    search_dir,
    search_file,
)


class RunBashCommandArgs(BaseModel):
    cmd: str


class EditFileArgs(BaseModel):
    start_line: int
    end_line: int
    replacement_text: str


class FindFileArgs(BaseModel):
    file_name: str
    dir: str


class ToolResponse(BaseModel):
    output: str

    def __str__(self) -> str:
        return self.output


class CreateNewFileArgs(BaseModel):
    path: str


# Argument classes for each tool
class OpenFileArgs(BaseModel):
    path: str
    line_number: int = -1  # Default as specified in the original tool


class ScrollArgs(BaseModel):
    """Empty args class for scroll operations"""

    # Make Gemini API happy
    dummy: Optional[str] = Field(None, description="Placeholder property")
    # pass


class SearchDirArgs(BaseModel):
    search_term: str
    dir: str


class SearchFileArgs(BaseModel):
    search_term: str
    file: str


class RunBashCommandTool(BaseTool[RunBashCommandArgs, ToolResponse]):
    """
    Adapter allowing an Autogen agent to call the Inspect `bash()` tool.
    """

    name = "run_bash_command"
    description = "Runs a bash command and returns the output."

    def __init__(
        self, run_bash_command_inspect_impl: Callable[..., Coroutine[Any, Any, str]]
    ) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            args_type=RunBashCommandArgs,
            return_type=ToolResponse,
        )
        self._run_bash_command_impl = run_bash_command_inspect_impl

    async def run(
        self,
        args: RunBashCommandArgs,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResponse:
        if isinstance(args, dict):
            parsed_args = RunBashCommandArgs(**args)
        else:
            parsed_args = args
        result = await self._run_bash_command_impl(cmd=parsed_args.cmd)

        # Truncate long output
        result_str = str(result)
        truncation_length = 5000
        if len(result_str) > truncation_length:
            truncated = result_str[:truncation_length]
            return ToolResponse(
                output=f"{truncated}\n\n[Output truncated. Total length: {len(result_str)} characters]"
            )

        return ToolResponse(output=str(result))


class CreateNewFileTool(BaseTool[CreateNewFileArgs, ToolResponse]):
    """
    Adapter allowing an Autogen agent to call the Inspect `create_new_file()` tool.
    """

    name = "create_new_file"
    description = "Creates and opens a new file at the given path in the editor."

    def __init__(
        self, create_new_file_inspect_impl: Callable[..., Coroutine[Any, Any, str]]
    ) -> None:
        """
        Args:
            create_new_file_inspect_impl: The Inspect @tool function that implements the real create_new_file.
        """
        super().__init__(
            name=self.name,
            description=self.description,
            args_type=CreateNewFileArgs,
            return_type=ToolResponse,
        )
        self._create_new_file_impl = create_new_file_inspect_impl

    async def run(
        self,
        args: CreateNewFileArgs,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResponse:
        if isinstance(args, dict):
            parsed_args = CreateNewFileArgs(**args)
        else:
            parsed_args = args
        result = await self._create_new_file_impl(path=parsed_args.path)
        return ToolResponse(output=str(result))


class EditFileTool(BaseTool[EditFileArgs, ToolResponse]):
    """
    Adapter allowing an Autogen agent to call the Inspect `edit_file()` tool.
    """

    name = "edit_file"
    description = (
        "Edits a file by replacing lines <start_line>:<end_line> "
        "with the given replacement_text. Checks for Python syntax "
        "errors before committing the change."
    )

    def __init__(
        self, edit_file_inspect_impl: Callable[..., Coroutine[Any, Any, str]]
    ) -> None:
        """
        Args:
            edit_file_inspect_impl: The Inspect @tool function that implements the real edit_file.
        """
        super().__init__(
            name=self.name,
            description=self.description,
            args_type=EditFileArgs,
            return_type=ToolResponse,
        )
        self._edit_file_impl = edit_file_inspect_impl

    async def run(
        self, args: EditFileArgs, cancellation_token: CancellationToken | None = None
    ) -> ToolResponse:
        if isinstance(args, dict):
            parsed_args = EditFileArgs(**args)
        else:
            parsed_args = args
        result = await self._edit_file_impl(
            start_line=parsed_args.start_line,
            end_line=parsed_args.end_line,
            replacement_text=parsed_args.replacement_text,
        )
        return ToolResponse(output=str(result))


class FindFileTool(BaseTool[FindFileArgs, ToolResponse]):
    """
    Adapter allowing an Autogen agent to call the Inspect `find_file()` tool.
    """

    name = "find_file"
    description = (
        "Finds all files matching a given name in the specified directory. "
        'Use dir="$" to search the current directory.'
        "IMPORTANT: Provide specific search terms (min 3 chars). "
    )

    def __init__(
        self, find_file_inspect_impl: Callable[..., Coroutine[Any, Any, str]]
    ) -> None:
        """
        Args:
            find_file_inspect_impl: The Inspect @tool function that implements the real find_file.
        """
        super().__init__(
            name=self.name,
            description=self.description,
            args_type=FindFileArgs,
            return_type=ToolResponse,
        )
        self._find_file_impl = find_file_inspect_impl

    async def run(
        self, args: FindFileArgs, cancellation_token: CancellationToken | None = None
    ) -> ToolResponse:
        if isinstance(args, dict):
            parsed_args = FindFileArgs(**args)
        else:
            parsed_args = args
        result = await self._find_file_impl(
            file_name=parsed_args.file_name, dir=parsed_args.dir
        )
        return ToolResponse(output=str(result))


# Open File Tool
class OpenFileTool(BaseTool[OpenFileArgs, ToolResponse]):
    """
    Adapter allowing an Autogen agent to call the Inspect `open_file()` tool.
    """

    name = "open_file"
    description = "Opens a file at the given path. If line_number is provided, moves to that line."

    def __init__(
        self, open_file_inspect_impl: Callable[..., Coroutine[Any, Any, str]]
    ) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            args_type=OpenFileArgs,
            return_type=ToolResponse,
        )
        self._open_file_impl = open_file_inspect_impl

    async def run(
        self, args: OpenFileArgs, cancellation_token: CancellationToken | None = None
    ) -> ToolResponse:
        if isinstance(args, dict):
            parsed_args = OpenFileArgs(**args)
        else:
            parsed_args = args
        result = await self._open_file_impl(
            path=parsed_args.path, line_number=parsed_args.line_number
        )
        return ToolResponse(output=str(result))


# Scroll Down Tool
class ScrollDownTool(BaseTool[ScrollArgs, ToolResponse]):
    """
    Adapter allowing an Autogen agent to call the Inspect `scroll_down()` tool.
    """

    name = "scroll_down"
    description = "Moves the window down in the current file."

    def __init__(
        self, scroll_down_inspect_impl: Callable[..., Coroutine[Any, Any, str]]
    ) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            args_type=ScrollArgs,
            return_type=ToolResponse,
        )
        self._scroll_down_impl = scroll_down_inspect_impl

    async def run(
        self, args: ScrollArgs, cancellation_token: CancellationToken | None = None
    ) -> ToolResponse:
        if isinstance(args, dict):
            parsed_args = ScrollArgs(**args)
        else:
            parsed_args = args
        result = await self._scroll_down_impl()
        return ToolResponse(output=str(result))


# Scroll Up Tool
class ScrollUpTool(BaseTool[ScrollArgs, ToolResponse]):
    """
    Adapter allowing an Autogen agent to call the Inspect `scroll_up()` tool.
    """

    name = "scroll_up"
    description = "Moves the window up in the current file."

    def __init__(
        self, scroll_up_inspect_impl: Callable[..., Coroutine[Any, Any, str]]
    ) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            args_type=ScrollArgs,
            return_type=ToolResponse,
        )
        self._scroll_up_impl = scroll_up_inspect_impl

    async def run(
        self, args: ScrollArgs, cancellation_token: CancellationToken | None = None
    ) -> ToolResponse:
        if isinstance(args, dict):
            parsed_args = ScrollArgs(**args)
        else:
            parsed_args = args
        result = await self._scroll_up_impl()
        return ToolResponse(output=str(result))


# Search Dir Tool
class SearchDirTool(BaseTool[SearchDirArgs, ToolResponse]):
    """
    Adapter allowing an Autogen agent to call the Inspect `search_dir()` tool.
    """

    name = "search_dir"
    description = "Searches for a term in all files in a directory recursively. Use dir='$' for current directory."

    def __init__(
        self, search_dir_inspect_impl: Callable[..., Coroutine[Any, Any, str]]
    ) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            args_type=SearchDirArgs,
            return_type=ToolResponse,
        )
        self._search_dir_impl = search_dir_inspect_impl

    async def run(
        self, args: SearchDirArgs, cancellation_token: CancellationToken | None = None
    ) -> ToolResponse:
        if isinstance(args, dict):
            parsed_args = SearchDirArgs(**args)
        else:
            parsed_args = args
        result = await self._search_dir_impl(
            search_term=parsed_args.search_term, dir=parsed_args.dir
        )
        return ToolResponse(output=str(result))


# Search File Tool
class SearchFileTool(BaseTool[SearchFileArgs, ToolResponse]):
    """
    Adapter allowing an Autogen agent to call the Inspect `search_file()` tool.
    """

    name = "search_file"
    description = (
        "Searches for a term in a specific file. Use file='$' for current open file."
    )

    def __init__(
        self, search_file_inspect_impl: Callable[..., Coroutine[Any, Any, str]]
    ) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            args_type=SearchFileArgs,
            return_type=ToolResponse,
        )
        self._search_file_impl = search_file_inspect_impl

    async def run(
        self, args: SearchFileArgs, cancellation_token: CancellationToken | None = None
    ) -> ToolResponse:
        if isinstance(args, dict):
            parsed_args = SearchFileArgs(**args)
        else:
            parsed_args = args
        result = await self._search_file_impl(
            search_term=parsed_args.search_term, file=parsed_args.file
        )
        return ToolResponse(output=str(result))


# Tool instances

run_bash_tool = RunBashCommandTool(
    cast(Callable[..., Coroutine[Any, Any, str]], bash())
)
create_new_file_tool = CreateNewFileTool(
    cast(Callable[..., Coroutine[Any, Any, str]], create_new_file())
)
open_tool = OpenFileTool(cast(Callable[..., Coroutine[Any, Any, str]], open_file()))
scroll_down_tool = ScrollDownTool(
    cast(Callable[..., Coroutine[Any, Any, str]], scroll_down())
)
scroll_up_tool = ScrollUpTool(
    cast(Callable[..., Coroutine[Any, Any, str]], scroll_up())
)
search_dir_tool = SearchDirTool(
    cast(Callable[..., Coroutine[Any, Any, str]], search_dir())
)
search_file_tool = SearchFileTool(
    cast(Callable[..., Coroutine[Any, Any, str]], search_file())
)
edit_tool = EditFileTool(cast(Callable[..., Coroutine[Any, Any, str]], edit_file()))
find_tool = FindFileTool(cast(Callable[..., Coroutine[Any, Any, str]], find_file()))
