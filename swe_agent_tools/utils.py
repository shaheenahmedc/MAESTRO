import json
import os
import shlex
from pathlib import Path

# from agents.tools.bash import run_bash_command
from typing import Tuple

from inspect_ai.util import ExecResult, sandbox, store

RUNTIME_INSTALL_ENABLED = bool(
    json.loads(os.getenv("AISI_AGENTS_RUNTIME_INSTALL_ENABLED", "1"))
)

SPLIT_STRING_LOCATION_SANDBOX = Path("/tmp/_split_string.py")
TOOL_SCRIPTS = Path(__file__).parent / "tool_scripts"
SPLIT_STRING_LOCATION_LOCAL = TOOL_SCRIPTS / "_split_string.py"
FUNCTION_DEFINITIONS_LOCATION_SANDBOX = Path("/tmp/_function_definitions.sh")


def get_all_function_definitions() -> str:
    all_tools = ""
    for file in TOOL_SCRIPTS.iterdir():
        if file.name != "all_swe_agent_tools.sh" and file.suffix == ".sh":
            all_tools += file.read_text() + "\n"

    all_tools = all_tools.replace(
        "{SPLIT_STRING_LOCATION_SANDBOX}", SPLIT_STRING_LOCATION_SANDBOX.as_posix()
    )

    return all_tools


async def run_bash_command(command: str, cwd: str) -> Tuple[ExecResult[str], str]:
    """Executes a bash command in the sandbox environment.

    Args:
        command: The bash command to execute
        cwd: The working directory to execute the command in

    Returns:
        Tuple containing:
        - ExecResult with command output
        - New working directory after command execution
    """
    # Get current state from store
    current_file = store().get("CURRENT_FILE", "")
    current_line = store().get("CURRENT_LINE", 0)

    # Create environment with current state
    env = {
        **default_env_variables,
        "CURRENT_FILE": str(current_file),
        "CURRENT_LINE": str(current_line),
    }

    # Execute command and capture environment changes
    wrapped_command = f"""
        cd {shlex.quote(cwd)}
        {command}
        echo "##ENV##CURRENT_FILE=$CURRENT_FILE"
        echo "##ENV##CURRENT_LINE=$CURRENT_LINE"
        echo "##PWD##$(pwd)"
    """
    result = await sandbox().exec(["bash", "-c", wrapped_command], env=env)

    # Parse environment changes and new working directory
    lines = result.stdout.split("\n")
    new_cwd = cwd

    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if line.startswith("##ENV##"):
            key, value = line[7:].split("=", 1)
            store().set(key, value)
            lines.pop(i)
        elif line.startswith("##PWD##"):
            new_cwd = line[7:]
            lines.pop(i)

    result.stdout = "\n".join(lines)
    return result, new_cwd


async def run_bash_for_swe_agent_tool(command: str) -> str:
    await install_tool_requirements()

    cwd = store().get("cwd", ".")
    command = f"source {FUNCTION_DEFINITIONS_LOCATION_SANDBOX.as_posix()} && {command}"
    result, new_cwd = await run_bash_command(command, cwd=cwd)
    store().set("cwd", new_cwd)
    assert result.stderr == "", (
        f"SWE-agent tools should have an empty stderr. Calling tool had:\n\n {result.stderr}; command was {command}"
    )
    return str(result.stdout)


default_env_variables = {
    "WINDOW": 100,
    "OVERLAP": 2,
    "CURRENT_LINE": 0,
    "CURRENT_FILE": "",
    "SEARCH_RESULTS": (),
    "SEARCH_FILES": (),
    "SEARCH_INDEX": 0,
    # Additional variables needed:
    "LAST_ACTION": "",  # Used for tracking scroll actions
    "SCROLL_COUNT": "0",  # Used for scroll warnings
    "ROOT": ".",  # Used as base directory for some operations
    "PREVIOUS_ERRORS": "",  # Used for tracking Python syntax errors
    "FLAKE8_OUTPUT": "",  # Used for Python linting results
    # Additional variables from codebase
    "WARN_AFTER_SCROLLING_TIMES": "3",  # Referenced in util_functions.sh and original scripts
    "OFFSET": "16",  # Used in open_file.sh and goto function (WINDOW/6)
    "PWD": ".",  # Working directory, used in state() function
}


async def install_tool_requirements() -> None:
    install_swe_agent_prerequisites = store().get("HAVE_INSTALLED_SWE_AGENT", False)
    if not install_swe_agent_prerequisites:
        store().set("HAVE_INSTALLED_SWE_AGENT", True)

        if RUNTIME_INSTALL_ENABLED:
            result = await sandbox().exec(["pip", "install", "flake8"], user="root")
            assert result.returncode == 0, (
                f"Failed to install flake8. Error: {result.stderr}"
            )

        await sandbox().write_file(
            FUNCTION_DEFINITIONS_LOCATION_SANDBOX.as_posix(),
            get_all_function_definitions(),
        )

        await sandbox().write_file(
            SPLIT_STRING_LOCATION_SANDBOX.as_posix(),
            SPLIT_STRING_LOCATION_LOCAL.read_text(),
        )
        result = await sandbox().exec(
            ["chmod", "+x", SPLIT_STRING_LOCATION_SANDBOX.as_posix()]
        )
        assert result.returncode == 0, (
            f"Failed to set environment variables. Error: {result.stderr}"
        )
        result, _ = await run_bash_command(
            "export "
            + " ".join(
                [f"{key}={value}" for key, value in default_env_variables.items()]
            ),
            cwd=".",
        )
        assert result.returncode == 0, (
            f"Failed to set environment variables. Error: {result.stderr}"
        )

        if RUNTIME_INSTALL_ENABLED:
            result = await sandbox().exec(
                [
                    "bash",
                    "-c",
                    "apt update && apt install -y jq wget make bash git file grep sed gawk coreutils findutils",
                ],
                user="root",
            )
            assert result.returncode == 0, (
                f"Failed to install tool dependencies. Error: {result.stderr}"
            )
