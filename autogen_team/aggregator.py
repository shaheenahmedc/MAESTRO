from typing import List

from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    default_subscription,
    message_handler,
)

from ..data_models.messages import Question, SolverRequest, FinalSolverResponse, Answer
from ..utils.logging import (
    get_agent_log_path,
    log_initialization,
    log_message,
    log_question_received,
    log_request_published,
    log_response_received,
    log_all_responses_received,
    log_final_answer_published,
)


@default_subscription
class CodeConsultantAggregator(RoutedAgent):
    def __init__(
        self, num_solvers: int, log_base_path: str, experiment_name: str
    ) -> None:
        super().__init__("CodeConsultant Aggregator")
        self._num_solvers = num_solvers
        self._buffer: List[FinalSolverResponse] = []
        self.final_answer: str = ""
        self._log_base_path = log_base_path
        self._experiment_name = experiment_name
        self._log_path = get_agent_log_path(
            str(self.id), self._log_base_path, self._experiment_name
        )
        log_initialization(self._log_path, str(self.id), num_solvers=num_solvers)

    @message_handler
    async def handle_question(self, message: Question, ctx: MessageContext) -> None:
        """Handle an incoming question by logging it and publishing to solvers."""
        log_question_received(self._log_path, str(self.id), message.content)

        prompt = self._create_solver_prompt(message.content)

        await self._publish_solver_request(prompt, message.content)

        log_request_published(self._log_path, str(self.id))

    def _create_solver_prompt(self, question_content: str) -> str:
        """Create a prompt for the solver agents based on the question."""
        return (
            f"Can you solve the following GitHub issue?\n{question_content}\n"
            "Explain your reasoning. Your final answer should be a general description of the solution, and steps to take to fix the issue.\n"
            "Provide your answer in the form of FINAL ANSWER: [your answer], at the end of your response."
        )

    async def _publish_solver_request(self, prompt: str, question: str) -> None:
        """Publish a solver request to all consultants."""
        await self.publish_message(
            SolverRequest(content=prompt, question=question),
            topic_id=DefaultTopicId(),
        )

    @message_handler
    async def handle_final_solver_response(
        self, message: FinalSolverResponse, ctx: MessageContext
    ) -> None:
        """Handle a final solution response from a solver agent."""
        log_response_received(self._log_path, str(self.id), ctx.sender)

        self._add_response_to_buffer(message)

        if self._have_all_responses():
            aggregated_response = self._create_aggregated_response()

            self._store_final_answer(aggregated_response)

            await self._publish_final_answer(aggregated_response)

            log_final_answer_published(self._log_path, str(self.id), len(self._buffer))

            self._buffer.clear()

    def _add_response_to_buffer(self, message: FinalSolverResponse) -> None:
        """Add a response to the buffer."""
        self._buffer.append(message)

    def _have_all_responses(self) -> bool:
        """Check if all expected responses have been received."""
        have_all = len(self._buffer) == self._num_solvers

        if have_all:
            log_all_responses_received(self._log_path, str(self.id), self._num_solvers)

        return have_all

    def _create_aggregated_response(self) -> str:
        """Create an aggregated response from all solutions."""
        all_solutions = [resp.answer for resp in self._buffer]

        # Format the response with all solutions
        aggregated_response = "## All solutions from the solver agents:\n\n"
        for i, solution in enumerate(all_solutions, 1):
            aggregated_response += f"### Solution {i}:\n{solution}\n\n"

        return aggregated_response

    def _store_final_answer(self, aggregated_response: str) -> None:
        """Store the final answer for future reference."""
        self.final_answer = aggregated_response

        log_message(
            self._log_path,
            f"Aggregator {self.id} publishing final answer with {len(self._buffer)} solutions",
        )

    async def _publish_final_answer(self, aggregated_response: str) -> None:
        """Publish the final aggregated answer."""
        await self.publish_message(
            Answer(content=aggregated_response), topic_id=DefaultTopicId()
        )
