from datetime import datetime
from typing import Dict, List, Any, Awaitable, Callable, Coroutine
import os
from pathlib import Path

from autogen_core import (
    DefaultTopicId,
    SingleThreadedAgentRuntime,
    TypeSubscription,
    AgentId,
)

from .agents.consultant import CodeConsultant
from .agents.aggregator import CodeConsultantAggregator
from .data_models.messages import Question
from .models.token_usage import TokenUsage
from .models.client_factory import create_model_client
from .utils.logging import (
    get_agent_log_path,
    log_missing_question,
    log_question_processing,
    log_agent_registration,
    log_subscription_setup,
    log_debate_starting,
    log_question_published,
    log_waiting_for_idle,
    log_debate_complete,
    log_collecting_token_usage,
    log_token_usage,
    log_token_usage_error,
    log_final_result_retrieval,
)
import json

# Import necessary tools
from inspect_evals.swe_bench.autogen_team.tools import (
    run_bash_tool,
    open_tool,
    scroll_down_tool,
    scroll_up_tool,
    search_dir_tool,
    search_file_tool,
    find_tool,
)


async def setup_debate_team() -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
    """
    Creates and sets up the debate team infrastructure.

    Returns:
        Function to run the team with a given input
    """
    config_path = "src/inspect_evals/swe_bench/autogen_team/configs/exp_1_1.json"
    # Load the config file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract config values
    experiment_name = config.get("experiment_name", "default_experiment")
    log_base_path = config.get(
        "log_base_path", "/root/inspect_evals/src/inspect_evals/swe_bench"
    )
    max_reflection_steps = config.get("max_reflection_steps", 10)
    agent_configs = config.get("agents", {})

    async def run_team(sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the multi-agent debate system on a given input.

        Args:
            sample: Dictionary containing an "input" key with the question

        Returns:
            Dictionary with the output result
        """
        print("setup debate team started")
        # Initialize components
        team_token_usage = TokenUsage()
        runtime = SingleThreadedAgentRuntime()
        run_team_log_path = get_agent_log_path(
            "team_orchestration", log_base_path, experiment_name
        )

        # Process input
        question_text = None
        if sample and "input" in sample:
            user_input = sample["input"]
            for message in user_input:
                if message.get("role") == "user" and "content" in message:
                    question_text = message["content"]
                    break

        if not question_text:
            log_missing_question(run_team_log_path)
            return {"output": "No question provided in the input"}

        log_question_processing(run_team_log_path, question_text)

        # Setup tools and agents
        tools = _setup_tools()

        # Register agents with the runtime
        await _register_agents(
            runtime,
            tools,
            agent_configs,
            log_base_path,
            experiment_name,
            run_team_log_path,
        )

        # Set up subscriptions
        await _setup_subscriptions(runtime, run_team_log_path)

        # Run the debate
        print("run debate started")
        await _run_debate(runtime, question_text, run_team_log_path)
        print("run debate finished")
        # Collect token usage statistics
        team_token_usage = await _collect_token_usage(
            runtime, team_token_usage, run_team_log_path
        )

        # Get the final answer
        result = await _get_aggregator_result(runtime, run_team_log_path)

        return {"output": result}

    def _setup_tools() -> List[Any]:
        """Set up the tools needed by the consultant agents."""
        return [
            run_bash_tool,
            open_tool,
            scroll_down_tool,
            scroll_up_tool,
            search_dir_tool,
            search_file_tool,
            find_tool,
        ]

    async def _register_agents(
        runtime: SingleThreadedAgentRuntime,
        tools: List[Any],
        agent_configs: Dict[str, Any],
        log_base_path: str,
        experiment_name: str,
        run_team_log_path: Path,
    ) -> None:
        """Register all agent instances with the runtime."""
        log_agent_registration(run_team_log_path)

        # Register consultant agents
        await CodeConsultant.register(
            runtime,
            "CodeConsultantA",
            lambda: CodeConsultant(
                model_client=create_model_client(agent_configs["agent_A"]),
                topic_type="CodeConsultantA",
                num_neighbors=2,
                max_round=3,
                tools=tools,
                max_reflection_steps=max_reflection_steps,
                log_base_path=log_base_path,
                experiment_name=experiment_name,
            ),
        )

        await CodeConsultant.register(
            runtime,
            "CodeConsultantB",
            lambda: CodeConsultant(
                model_client=create_model_client(agent_configs["agent_B"]),
                topic_type="CodeConsultantB",
                num_neighbors=2,
                max_round=3,
                tools=tools,
                max_reflection_steps=max_reflection_steps,
                log_base_path=log_base_path,
                experiment_name=experiment_name,
            ),
        )

        await CodeConsultant.register(
            runtime,
            "CodeConsultantC",
            lambda: CodeConsultant(
                model_client=create_model_client(agent_configs["agent_C"]),
                topic_type="CodeConsultantC",
                num_neighbors=2,
                max_round=3,
                tools=tools,
                max_reflection_steps=max_reflection_steps,
                log_base_path=log_base_path,
                experiment_name=experiment_name,
            ),
        )

        await CodeConsultant.register(
            runtime,
            "CodeConsultantD",
            lambda: CodeConsultant(
                model_client=create_model_client(agent_configs["agent_D"]),
                topic_type="CodeConsultantD",
                num_neighbors=2,
                max_round=3,
                tools=tools,
                max_reflection_steps=max_reflection_steps,
                log_base_path=log_base_path,
                experiment_name=experiment_name,
            ),
        )

        # Register aggregator agent
        await CodeConsultantAggregator.register(
            runtime,
            "CodeConsultantAggregator",
            lambda: CodeConsultantAggregator(
                num_solvers=4,
                log_base_path=log_base_path,
                experiment_name=experiment_name,
            ),
        )

    async def _setup_subscriptions(
        runtime: SingleThreadedAgentRuntime, log_path: Path
    ) -> None:
        """Set up the subscriptions between agents."""
        log_subscription_setup(log_path)

        # Subscriptions for CodeConsultantA
        await runtime.add_subscription(
            TypeSubscription("CodeConsultantA", "CodeConsultantD")
        )
        await runtime.add_subscription(
            TypeSubscription("CodeConsultantA", "CodeConsultantB")
        )

        # Subscriptions for CodeConsultantB
        await runtime.add_subscription(
            TypeSubscription("CodeConsultantB", "CodeConsultantA")
        )
        await runtime.add_subscription(
            TypeSubscription("CodeConsultantB", "CodeConsultantC")
        )

        # Subscriptions for CodeConsultantC
        await runtime.add_subscription(
            TypeSubscription("CodeConsultantC", "CodeConsultantB")
        )
        await runtime.add_subscription(
            TypeSubscription("CodeConsultantC", "CodeConsultantD")
        )

        # Subscriptions for CodeConsultantD
        await runtime.add_subscription(
            TypeSubscription("CodeConsultantD", "CodeConsultantC")
        )
        await runtime.add_subscription(
            TypeSubscription("CodeConsultantD", "CodeConsultantA")
        )

    async def _run_debate(
        runtime: SingleThreadedAgentRuntime, question_text: str, log_path: Path
    ) -> None:
        """Run the debate by starting the runtime and publishing the question."""
        log_debate_starting(log_path)

        # Start the runtime
        runtime.start()

        # Publish the initial question
        await runtime.publish_message(Question(content=question_text), DefaultTopicId())

        log_question_published(log_path)

        # Wait for processing to complete
        log_waiting_for_idle(log_path)

        await runtime.stop_when_idle()

        log_debate_complete(log_path)

    async def _collect_token_usage(
        runtime: SingleThreadedAgentRuntime,
        team_token_usage: TokenUsage,
        log_path: Path,
    ) -> TokenUsage:
        """Collect token usage statistics from all consultant agents."""
        log_collecting_token_usage(log_path)

        try:
            # Get references to all consultant agents
            consultantA = await runtime._get_agent(
                AgentId("CodeConsultantA", "default")
            )
            consultantB = await runtime._get_agent(
                AgentId("CodeConsultantB", "default")
            )
            consultantC = await runtime._get_agent(
                AgentId("CodeConsultantC", "default")
            )
            consultantD = await runtime._get_agent(
                AgentId("CodeConsultantD", "default")
            )

            # Collect token usage from each agent
            for agent in [consultantA, consultantB, consultantC, consultantD]:
                if isinstance(agent, CodeConsultant) and hasattr(
                    agent, "_model_client"
                ):
                    team_token_usage.update(agent._model_client.total_usage())

            log_token_usage(log_path, team_token_usage)

        except Exception as e:
            print(f"Error during token usage collection: {type(e).__name__}: {str(e)}")
            log_token_usage_error(log_path, e)

        return team_token_usage

    async def _get_aggregator_result(
        runtime: SingleThreadedAgentRuntime, log_path: Path
    ) -> str:
        """Get the final answer from the aggregator agent."""
        # Get reference to the aggregator agent
        aggregator = await runtime._get_agent(
            AgentId("CodeConsultantAggregator", "default")
        )

        # Get the final answer
        result = (
            aggregator.final_answer
            if hasattr(aggregator, "final_answer")
            else "No answer found"
        )

        log_final_result_retrieval(log_path, len(result))

        return result

    return run_team
