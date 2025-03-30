from inspect_ai.tool import Tool, tool
from inspect_ai.util import subtask
from inspect_ai.util import store
from typing import Any, Callable, Coroutine, Dict

from .runtime import setup_debate_team


@subtask
async def consult_team_subtask(question: str) -> str:
    # Force install tools at subtask start - once for the entire subtask
    from inspect_evals.swe_bench.swe_agent_tools.utils import install_tool_requirements

    # Reset installation flag to ensure fresh installation
    store().set("HAVE_INSTALLED_SWE_AGENT", False)
    await install_tool_requirements()
    # Create properly formatted sample
    sample = {
        "input": [{"role": "user", "content": question}],
        "sample_id": "consultation",
        "epoch": 0,
        "metadata": {},
        "target": [""],
    }

    # First call multi_agent_consultancy_team() to get the run function
    team_function = await setup_debate_team()

    # Then call that function with your sample
    result = await team_function(sample)

    # Extract and return the output
    return str(result["output"])


@tool
def consult_multi_agent_team(
    timeout: int = 900,
) -> Callable[[str], Coroutine[Any, Any, str]]:
    async def execute(question: str) -> str:
        """
        Consults a team of expert agents about a coding problem.

        Args:
            question: The specific coding problem or question to analyze

        Returns:
            A summary of the team's discussion and recommendations
        """
        try:
            # Get a clean string result from the multi-agent team
            result = await consult_team_subtask(question)

            # Ensure we're only returning a clean string, not any message objects
            if result is None:
                return "No result was provided by the consultation team."

            # Make sure we return a plain string result
            return str(result)

        except Exception as e:
            print(f"Error during multi-agent consultation: {str(e)}")
            # Provide informative error info in case something goes wrong
            error_msg = f"Error during multi-agent consultation: {str(e)}"
            return error_msg

    return execute
