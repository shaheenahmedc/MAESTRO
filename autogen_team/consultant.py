from autogen_core import (
    DefaultTopicId,
    FunctionCall,
    MessageContext,
    RoutedAgent,
    default_subscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import BaseTool
from typing import Any, Dict, List, Union
import json

from ..data_models.messages import (
    FinalSolverResponse,
    IntermediateSolverResponse,
    SolverRequest,
)
from ..utils.logging import (
    get_agent_log_path,
    log_initialization,
    log_message,
    log_question_received,
    log_response_received,
    log_all_responses_received,
    log_tool_execution,
    log_reflection_process,
    log_llm_response,
)


@default_subscription
class CodeConsultant(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
        topic_type: str,
        num_neighbors: int,
        max_round: int,
        tools: List[BaseTool[Any, Any]] | None = None,  # Add type parameters
        max_reflection_steps: int = 3,  # Control reflection depth
        log_base_path: str = "/root/inspect_evals/src/inspect_evals/swe_bench",
        experiment_name: str = "test_experiment",
    ) -> None:
        super().__init__("A debator.")
        self._topic_type = topic_type
        self._model_client = model_client
        self._num_neighbors = num_neighbors
        self._history: List[LLMMessage] = []
        self._buffer: Dict[int, List[IntermediateSolverResponse]] = {}
        self._log_path = get_agent_log_path(
            str(self.id), log_base_path, experiment_name
        )
        self._max_reflection_steps = max_reflection_steps
        # Add reflection state tracking
        self._is_reflecting = False
        self._message_queue = []
        self._system_messages = [
            SystemMessage(
                content=(
                    f"You are a helpful assistant with expertise in coding. \
                    Your task is to provide another language model with guidance on how to solve a GitHub issue by providing \
                    a clear and detailed solution. \
                    You should have a codebase already which you can explore. \
                    You don't need to download it, just explore it. \
                    You have a number of tools at your disposal to help you create this guidance. Use these to explore the codebase. \
                    DO NOT RUN ANYTHING WHICH CHANGES THE CODEBASE, ONLY USE THIS TO EXPLORE THE CODEBASE. \
                    These tools are: \
                    - run_bash_tool: to run commands in the terminal (output truncated to 5000 characters) \
                    - open_file: to open a file \
                    - search_dir: to search for a file in a directory \
                    - search_file: to search for a search term in a file \
                    - find_file: to find a file in the current directory \
                    - scroll_down: to scroll down in the terminal \
                    - scroll_up: to scroll up in the terminal \
                    IMPORTANT REQUIREMENTS: \
                    You MUST use at least {int(self._max_reflection_steps * 0.6)} reflection steps before providing a final answer \
                    You will be given messages from other agents at the end of each round. \
                    You should try to be as critical as possible of the messages you receive, and use your reflection loop to explore alternative solutions. \
                    Try to reduce redundant exploration amongst all agents, by summarising what steps you've already taken, in your final answer. \
                    Limit your final answer to 500 words, and explicitly mark it with \
                    FINAL ANSWER: at the beginning."
                )
            )
        ]
        self._round = 0
        self._max_round = max_round
        self._tools = tools or []

        # Log initialization information
        log_initialization(
            self._log_path,
            str(self.id),
            topic_type=topic_type,
            num_neighbors=num_neighbors,
        )

        # Log tools availability
        log_message(
            self._log_path, f"Agent {self.id} has {len(self._tools)} tools available"
        )

    async def _execute_tool_call(
        self, tool_call: Union[str, FunctionCall], ctx: MessageContext
    ) -> FunctionExecutionResult:
        """Execute a single tool call and return the result"""
        if isinstance(tool_call, str):
            return FunctionExecutionResult(
                call_id="",
                content=f"Invalid tool call format: {tool_call}",
                is_error=True,
                name="unknown",
            )

        matching_tool = next(
            (tool for tool in self._tools if tool.name == tool_call.name),
            None,
        )

        if not matching_tool:
            error_msg = f"Tool {tool_call.name} not found"
            log_tool_execution(
                self._log_path, tool_call.name, {}, error=ValueError(error_msg)
            )
            return FunctionExecutionResult(
                call_id=tool_call.id,
                content=error_msg,
                is_error=True,
                name=tool_call.name,
            )

        try:
            # Parse the arguments properly using json.loads
            args = json.loads(tool_call.arguments) if tool_call.arguments else {}
            log_tool_execution(self._log_path, tool_call.name, args)

            # Execute the tool using the run_json method from Autogen Core
            result = await matching_tool.run_json(
                args,
                cancellation_token=ctx.cancellation_token,
            )
            # Get the result as string using the tool's return_value_as_string method
            result_str = matching_tool.return_value_as_string(result)

            # Log successful tool execution
            log_tool_execution(self._log_path, tool_call.name, args, result=result_str)

            return FunctionExecutionResult(
                call_id=tool_call.id,
                content=result_str,
                is_error=False,
                name=tool_call.name,
            )
        except Exception as e:
            # Log tool execution error
            print(f"Error executing tool: {str(e)}")
            log_tool_execution(self._log_path, tool_call.name, args, error=e)

            return FunctionExecutionResult(
                call_id=tool_call.id,
                content=f"Error executing tool: {str(e)}",
                is_error=True,
                name=tool_call.name,
            )

    @message_handler
    async def handle_solver_request(
        self, message: SolverRequest, ctx: MessageContext
    ) -> None:
        """Handle an initial request to solve a problem."""
        log_question_received(self._log_path, str(self.id), message.content)
        # Queue if reflecting
        if self._is_reflecting:
            log_message(
                self._log_path,
                f"Agent {self.id} is reflecting, queueing solver request",
            )
            self._message_queue.append((message, ctx))
            return
        # Start reflection process
        reflection_result = await self._reflect_on_problem(message.content, ctx)

        # Extract final answer from reflection result
        final_answer = self._extract_final_answer(reflection_result)

        # Update history
        self._update_history_with_reflection(message.content, reflection_result)

        # Increment the round counter
        self._round += 1
        # Check if we need to publish intermediate or final response
        if self._is_final_round():
            await self._publish_final_response(final_answer)
        else:
            await self._publish_intermediate_response(message.question, final_answer)

    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from a response."""
        if "FINAL ANSWER:" in response:
            return response[response.find("FINAL ANSWER:") :]
        else:
            return f"FINAL ANSWER: {response}"

    async def _reflect_on_problem(
        self, message_content: str, ctx: MessageContext
    ) -> str:
        """Reflect on a problem through iterative LLM calls and tool use."""
        # Set reflection flag at the beginning
        if self._is_reflecting:
            log_message(
                self._log_path,
                f"Agent {self.id} already reflecting, queueing redundant reflection request",
            )
            return ""
        self._is_reflecting = True

        try:
            # Initialize messages with our system message and the problem to solve
            messages = self._prepare_initial_messages(message_content)
            # Log the start of reflection
            self._log_reflection_start(message_content, messages)

            # Main reflection loop
            final_answer = None
            retry_count = 0
            for step in range(self._max_reflection_steps):
                try:
                    # Log current reflection step
                    log_reflection_process(
                        self._log_path, step, self._max_reflection_steps, messages
                    )

                    # Call the model to get a response
                    response = await self._model_client.create(
                        messages=messages,
                        cancellation_token=ctx.cancellation_token,
                        tools=self._tools,
                    )

                    # Log model's response
                    log_llm_response(self._log_path, response)

                    # Handle model response based on type
                    if self._is_tool_call_response(response):
                        # Add the assistant's response containing the tool calls
                        messages.append(
                            AssistantMessage(
                                content=response.content, source="assistant"
                            )
                        )

                        # Process each tool call
                        tool_results = []
                        for tool_call in response.content:
                            tool_result = await self._execute_tool_call(tool_call, ctx)
                            tool_results.append(tool_result)

                        # Add all tool results in a single message
                        messages.append(
                            FunctionExecutionResultMessage(content=tool_results)
                        )
                    else:
                        # Check if response contains final answer
                        content = str(response.content)
                        messages.append(
                            AssistantMessage(content=content, source="assistant")
                        )

                        if "FINAL ANSWER:" in content:
                            final_answer = content[content.find("FINAL ANSWER:") :]
                            log_message(
                                self._log_path,
                                f"Found final answer in step {step + 1}: {final_answer[:100]}...",
                            )
                            break

                except Exception as e:
                    print(f"Error during reflection: {type(e).__name__}: {str(e)}")
                    self._handle_reflection_error(e, messages)
                    retry_count += 1

                    if retry_count > 2:
                        log_message(
                            self._log_path,
                            f"Too many errors ({retry_count}), terminating reflection",
                        )
                        break

            # If no final answer was found, ask explicitly for one
            if final_answer is None:
                final_answer = await self._get_final_answer(messages, ctx)

            log_message(
                self._log_path,
                f"Final answer after reflection: {final_answer[:100]}...",
            )

            return final_answer
        finally:
            # Reset reflection state after reflection is complete
            self._is_reflecting = False
            await self._process_queued_messages()

    async def _process_queued_messages(self) -> None:
        """Process any messages that were queued during reflection."""
        if not self._message_queue:
            return

        log_message(
            self._log_path,
            f"Agent {self.id} processing {len(self._message_queue)} queued messages",
        )

        # Copy and clear queue before processing to prevent recursion
        messages_to_process = self._message_queue.copy()
        self._message_queue = []

        # Process each message
        for msg, ctx in messages_to_process:
            await self.on_message_impl(msg, ctx)

    def _prepare_initial_messages(self, message_content: str) -> List[LLMMessage]:
        """Prepare initial messages for the LLM, including relevant history."""
        # Start with system messages
        messages: List[LLMMessage] = self._system_messages.copy()

        # Include relevant history if available
        if self._history:
            # Option to truncate history if it gets too long
            relevant_history = self._history[
                :
            ]  # Keep last n messages, adjust as needed
            messages.extend(relevant_history)

        # Add the current problem last
        messages.append(
            UserMessage(
                content=f"GitHub issue to solve: {message_content}",
                source="user",
            )
        )
        return messages

    def _log_reflection_start(
        self, message_content: str, messages: List[LLMMessage]
    ) -> None:
        """Log the start of reflection process"""
        # Create a message that captures the start of reflection
        tools_info = [tool.name for tool in self._tools]
        model_info = type(self._model_client).__name__

        log_message(
            self._log_path, f"=== DEBUGGING START: Agent {self.id} reflecting ==="
        )

        log_message(self._log_path, f"Message content: {message_content}")

        log_message(self._log_path, f"Available tools: {tools_info}")

        log_message(self._log_path, f"Model client type: {model_info}")

        # Try to safely log initial messages
        try:
            msg_dicts = [
                msg.to_dict() if hasattr(msg, "to_dict") else str(msg)
                for msg in messages
            ]
            log_message(
                self._log_path, f"Initial messages: {json.dumps(msg_dicts, indent=2)}"
            )
        except Exception as e:
            print(f"Could not serialize initial messages: {str(e)}")
            log_message(
                self._log_path, f"Could not serialize initial messages: {str(e)}"
            )

        log_message(self._log_path, f"Agent {self.id} about to start reflection")

    def _is_tool_call_response(self, response: Any) -> bool:
        """Check if the response is a tool call."""
        if not hasattr(response, "content"):
            return False
        if not response.content:
            return False
        if isinstance(response.content, List) and len(response.content) > 0:
            return isinstance(response.content[0], FunctionCall)
        return False

    def _handle_reflection_error(
        self, error: Exception, messages: List[LLMMessage]
    ) -> None:
        """Handle and log errors during reflection."""
        log_message(
            self._log_path,
            f"Error during reflection: {type(error).__name__}: {str(error)}",
        )

        # Add error message to the conversation
        messages.append(
            UserMessage(
                content=f"Error occurred: {str(error)}. Please continue.",
                source="user",
            )
        )

    async def _get_final_answer(
        self, messages: List[LLMMessage], ctx: MessageContext
    ) -> str:
        """Get final answer when reflection cycle ends without a clear answer."""
        log_message(
            self._log_path,
            "No final answer found in reflection. Requesting explicit final answer.",
        )

        # Add request for final answer
        messages.append(
            UserMessage(
                content="Please provide your FINAL ANSWER now, starting with 'FINAL ANSWER:'",
                source="user",
            )
        )

        try:
            response = await self._model_client.create(
                messages=messages,
                cancellation_token=ctx.cancellation_token,
                tools=self._tools,
            )

            log_llm_response(self._log_path, response)

            content = str(response.content)
            if "FINAL ANSWER:" in content:
                return content[content.find("FINAL ANSWER:") :]
            else:
                return f"FINAL ANSWER: {content}"

        except Exception as e:
            print(f"Error getting final answer: {type(e).__name__}: {str(e)}")
            log_message(
                self._log_path,
                f"Error getting final answer: {type(e).__name__}: {str(e)}",
            )
            return "FINAL ANSWER: Could not generate a response due to an error."

    def _update_history_with_reflection(
        self, message_content: str, reflection_result: str
    ) -> None:
        """Update agent history with reflection result."""
        # Add the incoming message with all context to history
        self._history.append(UserMessage(content=message_content, source="user"))

        # Add the agent's own reflection result
        self._history.append(
            AssistantMessage(content=reflection_result, source=self.metadata["type"])
        )

        log_message(
            self._log_path,
            f"Updated history with reflection result (history length: {len(self._history)})",
        )

    def _is_final_round(self) -> bool:
        """Check if the current round is the final round."""
        return self._round >= self._max_round

    async def _publish_final_response(self, answer: str) -> None:
        """Publish the final response for this conversation."""
        await self.publish_message(
            FinalSolverResponse(answer=answer), topic_id=DefaultTopicId()
        )

        log_message(
            self._log_path,
            f"Agent {self.id} reached max round {self._max_round}, publishing final response",
        )

    async def _publish_intermediate_response(self, question: str, answer: str) -> None:
        """Publish an intermediate response to the agent's topic."""
        topic_id = DefaultTopicId(type=self._topic_type)

        await self.publish_message(
            IntermediateSolverResponse(
                content=answer,
                question=question,
                answer=answer,
                round=self._round,
            ),
            topic_id=topic_id,
        )

        log_message(
            self._log_path,
            f"Agent {self.id} round {self._round}, publishing intermediate response to topic {topic_id}",
        )

    @message_handler
    async def handle_response(
        self, message: IntermediateSolverResponse, ctx: MessageContext
    ) -> None:
        """Handle an incoming response from another agent."""
        log_response_received(self._log_path, str(self.id), ctx.sender)
        # Queue if reflecting
        if self._is_reflecting:
            log_message(
                self._log_path,
                f"Agent {self.id} is reflecting, queueing response from {ctx.sender}",
            )
            self._message_queue.append((message, ctx))
            return

        self._add_response_to_buffer(message)

        if self._have_all_neighbor_responses(message.round):
            consolidated_prompt = self._create_consolidated_prompt(message)
            await self._send_consolidated_prompt_to_self(
                message.question, consolidated_prompt
            )
            self._clear_buffer_for_round(message.round)

    def _add_response_to_buffer(self, message: IntermediateSolverResponse) -> None:
        """Add a response to the buffer for its round."""
        self._buffer.setdefault(message.round, []).append(message)

    def _have_all_neighbor_responses(self, round_num: int) -> bool:
        """Check if all expected neighbor responses have been received for a round."""
        received_count = len(self._buffer.get(round_num, []))
        has_all_responses = received_count >= self._num_neighbors

        if has_all_responses:
            log_all_responses_received(self._log_path, str(self.id), received_count)
            log_message(
                self._log_path,
                f"Agent {self.id} received all {received_count} neighbor responses for round {round_num}",
            )

        return has_all_responses

    def _create_consolidated_prompt(self, message: IntermediateSolverResponse) -> str:
        """Create a consolidated prompt from all responses for a round."""
        prompt = f"ROUND {self._round + 1} - New solutions from other agents:\n"

        for resp in self._buffer[message.round]:
            prompt += f"One agent solution: {resp.content}\n"

        prompt += (
            "Using the solutions from other agents as additional information, "
            "can you provide your solution to the GitHub issue? "
            f"The original coding problem is {message.question}. "
        )

        # Add the agent's conversation history
        if self._history and len(self._history) >= 2:
            prompt += "\nYour conversation history:\n"
            # Add a summarized version of the history
            for i, msg in enumerate(self._history):
                if i == 0:
                    prompt += f"Original problem: {msg.content}...\n"
                elif isinstance(msg, AssistantMessage):
                    prompt += f"Your previous analysis: {msg.content}...\n"
                else:
                    prompt += f"Previous input: {msg.content}...\n"

        prompt += (
            "\nUsing all available information, revise your solution. "
            "Your final answer should be an explanation of the solution, "
            "in the form of FINAL ANSWER: [your answer], at the end of your response."
        )

        return prompt

    async def _send_consolidated_prompt_to_self(
        self, question: str, prompt: str
    ) -> None:
        """Send the consolidated prompt to the agent itself for processing."""
        await self.send_message(
            SolverRequest(content=prompt, question=question), self.id
        )

        log_message(
            self._log_path,
            f"Agent {self.id} sending consolidated message to self for processing",
        )

    def _clear_buffer_for_round(self, round_num: int) -> None:
        """Clear the buffer for a specific round."""
        self._buffer.pop(round_num, None)
