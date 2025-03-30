from datetime import datetime
from pathlib import Path
import os
import json
from typing import Any, Dict, List, Union, Optional


def get_agent_log_path(agent_id: str, log_base_path: str, experiment_name: str) -> Path:
    """
    Generate a unique log file path for an agent.

    Args:
        agent_id: The identifier of the agent
        log_base_path: Base directory path for storing logs
        experiment_name: Name of the experiment for log organization

    Returns:
        Path object for the log file
    """
    # Use microsecond precision and add a random component
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Add a random string to ensure uniqueness even if timestamps match
    random_suffix = os.urandom(4).hex()
    # Sanitize agent_id for filename by replacing slashes and other problematic characters
    safe_agent_id = str(agent_id).replace("/", "_").replace("\\", "_").replace(":", "_")
    # Folder name for organizing logs by experiment parameters
    log_folder_name = log_base_path + experiment_name
    os.makedirs(log_folder_name, exist_ok=True)

    return Path(
        f"{log_folder_name}/conversation_log_agent_{safe_agent_id}_time_{timestamp}_{random_suffix}.txt"
    )


def log_message(log_path: Path, message: str) -> None:
    """
    Log a message with timestamp to a file.

    Args:
        log_path: Path to the log file
        message: Message to log
    """
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}: {message}\n\n\n\n")


# Agent Initialization Logging


def log_initialization(log_path: Path, agent_id: str, **kwargs: Any) -> None:
    """
    Log the initialization of an agent.

    Args:
        log_path: Path to the log file
        agent_id: ID of the agent
        **kwargs: Additional initialization parameters to log
    """
    params_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    log_message(log_path, f"Agent {agent_id} initialized with {params_str}")


# Message Handling Logging


def log_question_received(log_path: Path, agent_id: str, content: str) -> None:
    """
    Log that a question was received by an agent.

    Args:
        log_path: Path to the log file
        agent_id: ID of the agent
        content: Content of the question
    """
    log_message(log_path, f"Agent {agent_id} received Question: {content}")


def log_request_published(
    log_path: Path, agent_id: str, target: str = "all consultants"
) -> None:
    """
    Log that a request was published.

    Args:
        log_path: Path to the log file
        agent_id: ID of the agent
        target: Target of the published request
    """
    log_message(
        log_path, f"Agent {agent_id} published initial solver request to {target}"
    )


def log_response_received(log_path: Path, agent_id: str, sender: Any) -> None:
    """
    Log that a response was received.

    Args:
        log_path: Path to the log file
        agent_id: ID of the agent
        sender: Sender of the response
    """
    log_message(log_path, f"Agent {agent_id} received response from {str(sender)}")


def log_all_responses_received(log_path: Path, agent_id: str, count: int) -> None:
    """
    Log that all responses have been received.

    Args:
        log_path: Path to the log file
        agent_id: ID of the agent
        count: Number of responses received
    """
    log_message(log_path, f"Agent {agent_id} received all {count} expected responses")


def log_final_answer_published(
    log_path: Path, agent_id: str, solution_count: int
) -> None:
    """
    Log that the final answer has been published.

    Args:
        log_path: Path to the log file
        agent_id: ID of the agent
        solution_count: Number of solutions included in the answer
    """
    log_message(
        log_path,
        f"Agent {agent_id} publishes final answer with {solution_count} solutions",
    )


# Tool Execution Logging


def log_tool_execution(
    log_path: Path,
    tool_name: str,
    args: Dict[str, Any],
    result: Any = None,
    error: Optional[Exception] = None,
) -> None:
    """
    Log a tool execution with its arguments and result.

    Args:
        log_path: Path to the log file
        tool_name: Name of the tool
        args: Arguments passed to the tool
        result: Result of the tool execution (if successful)
        error: Error that occurred (if any)
    """
    if result is not None:
        log_message(log_path, f"Tool {tool_name} called with args: {args}")
        log_message(log_path, f"Tool {tool_name} result: {str(result)}")
    elif error is not None:
        log_message(log_path, f"Tool execution error: {str(error)}")
        log_message(log_path, f"Error type: {type(error)}")
        log_message(log_path, f"Error args: {error.args}")


# Reflection Process Logging


def log_reflection_process(
    log_path: Path, step: int, max_steps: int, messages: List[Any]
) -> None:
    """
    Log details of a reflection step.

    Args:
        log_path: Path to the log file
        step: Current step number
        max_steps: Maximum number of steps
        messages: Messages being sent to the LLM
    """
    timestamp = datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"=== STEP {step + 1}/{max_steps} at {timestamp} ===\n")
        # Safely convert messages to JSON if possible
        try:
            msg_dicts = [
                msg.to_dict() if hasattr(msg, "to_dict") else str(msg)
                for msg in messages
            ]
            f.write(
                f"Messages being sent to LLM: {json.dumps(msg_dicts, indent=2)}\n\n\n\n"
            )
        except Exception as e:
            print(f"Could not serialize messages: {str(e)}")
            f.write(f"Could not serialize messages: {str(e)}\n\n\n\n")


def log_llm_response(log_path: Path, response: Any) -> None:
    """
    Log an LLM response.

    Args:
        log_path: Path to the log file
        response: Response from the LLM
    """
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"LLM API call successful!\n")
        f.write(f"Response object type: {type(response)}\n")

        if hasattr(response, "to_dict"):
            try:
                f.write(
                    f"Response content: {json.dumps(response.to_dict(), indent=2)}\n\n\n\n"
                )
            except Exception:
                print(f"Could not serialize response: {str(e)}")
                f.write(
                    f"Response content (not JSON serializable): {str(response)}\n\n\n\n"
                )
        elif hasattr(response, "__dict__"):
            try:
                f.write(
                    f"Response content: {json.dumps({k: str(v) for k, v in response.__dict__.items()}, indent=2)}\n\n\n\n"
                )
            except Exception:
                print(f"Could not serialize response: {str(e)}")
                f.write(
                    f"Response content (not JSON serializable): {str(response)}\n\n\n\n"
                )
        else:
            f.write(f"Response content: {str(response)}\n\n\n\n")


# Runtime Logging


def log_runtime_event(
    log_path: Path, event: str, details: Optional[str] = None
) -> None:
    """
    Log a runtime event.

    Args:
        log_path: Path to the log file
        event: Description of the event
        details: Additional details about the event
    """
    message = event
    if details:
        message += f": {details}"
    log_message(log_path, message)


def log_runtime_creation(log_path: Path) -> None:
    """
    Log that a runtime has been created.

    Args:
        log_path: Path to the log file
    """
    log_message(log_path, "Team orchestration runtime created")


def log_missing_question(log_path: Path) -> None:
    """
    Log that no question was provided in the input.

    Args:
        log_path: Path to the log file
    """
    log_message(log_path, "No question provided in the input")


def log_question_processing(log_path: Path, question: str) -> None:
    """
    Log that a question is being processed.

    Args:
        log_path: Path to the log file
        question: The question being processed
    """
    log_message(log_path, f"Processing question: {question}")


def log_token_usage(log_path: Path, usage: Any) -> None:
    """
    Log token usage information.

    Args:
        log_path: Path to the log file
        usage: Token usage object or information
    """
    log_message(log_path, f"Total token usage: {str(usage)}")


def log_team_setup(log_path: Path, agent_count: int, agent_types: List[str]) -> None:
    """
    Log team setup information.

    Args:
        log_path: Path to the log file
        agent_count: Number of agents in the team
        agent_types: Types of agents in the team
    """
    log_message(
        log_path, f"Team setup with {agent_count} agents: {', '.join(agent_types)}"
    )


def log_tools_setup(log_path: Path, tool_count: int, tool_names: List[str]) -> None:
    """
    Log tools setup information.

    Args:
        log_path: Path to the log file
        tool_count: Number of tools set up
        tool_names: Names of tools set up
    """
    log_message(log_path, f"Set up {tool_count} tools: {', '.join(tool_names)}")


def log_aggregator_result(log_path: Path, result_length: int) -> None:
    """
    Log that the aggregator result has been retrieved.

    Args:
        log_path: Path to the log file
        result_length: Length of the result in characters
    """
    log_message(
        log_path, f"Final result retrieved from aggregator ({result_length} chars)"
    )


def log_debate_completed(log_path: Path, elapsed_time: float) -> None:
    """
    Log that the debate has been completed.

    Args:
        log_path: Path to the log file
        elapsed_time: Time elapsed during the debate in seconds
    """
    log_message(log_path, f"Debate completed in {elapsed_time:.2f} seconds")


def log_agent_registration(
    log_path: Path, agent_count: int = 4, aggregator_count: int = 1
) -> None:
    """Log that agents are being registered."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().isoformat()}: Registering {agent_count} CodeConsultant agents and {aggregator_count} Aggregator\n\n\n\n"
        )


def log_subscription_setup(log_path: Path) -> None:
    """Log that subscriptions are being set up."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()}: Setting up agent subscriptions\n\n\n\n")


def log_debate_starting(log_path: Path) -> None:
    """Log that the debate is starting."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().isoformat()}: Starting runtime and publishing initial question\n\n\n\n"
        )


def log_question_published(log_path: Path) -> None:
    """Log that the initial question has been published."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().isoformat()}: Publishing initial question to runtime\n\n\n\n"
        )


def log_waiting_for_idle(log_path: Path) -> None:
    """Log that the system is waiting for the runtime to become idle."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().isoformat()}: Waiting for runtime to become idle\n\n\n\n"
        )


def log_debate_complete(log_path: Path) -> None:
    """Log that the debate is complete."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().isoformat()}: Runtime is now idle, debate complete\n\n\n\n"
        )


def log_collecting_token_usage(log_path: Path) -> None:
    """Log that token usage statistics are being collected."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().isoformat()}: Collecting token usage statistics\n\n\n\n"
        )


def log_token_usage_error(log_path: Path, error: Exception) -> None:
    """Log an error collecting token usage."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().isoformat()}: Error collecting token usage: {str(error)}\n\n\n\n"
        )


def log_final_result_retrieval(log_path: Path, result_length: int) -> None:
    """Log that the final result has been retrieved."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().isoformat()}: Final result retrieved from aggregator ({result_length} chars)\n\n\n\n"
        )
