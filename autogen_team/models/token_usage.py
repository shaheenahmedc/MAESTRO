from dataclasses import dataclass
from typing import Any


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def update(self, usage: Any) -> None:
        """Update token counts from RequestUsage object"""
        if not usage:
            return

        # Handle RequestUsage objects from AutoGen
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens

    def __str__(self) -> str:
        return f"Tokens: {self.total_tokens} (Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens})"
