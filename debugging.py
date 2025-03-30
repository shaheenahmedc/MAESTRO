# import asyncio
# import inspect
# import json
# import os
# from typing import Any

# from autogen_core.models import UserMessage
# from autogen_core.tools import FunctionTool
# from autogen_ext.models.openai import OpenAIChatCompletionClient

# # Create a client that uses OpenRouter
# client = OpenAIChatCompletionClient(
#     model="amazon/nova-pro-v1",
#     api_key=os.environ["OPENROUTER_API_KEY"],
#     base_url="https://openrouter.ai/api/v1",
#     model_info={
#         "family": "unknown",
#         "vision": True,
#         "function_calling": True,
#         "json_output": True,
#     },
# )

# # Patch the client's internal methods to debug API responses
# original_create_method = client._client.chat.completions.create


# async def debug_create(*args: Any, **kwargs: Any) -> Any:
#     print("API CALL ARGS:", json.dumps(kwargs, indent=2))
#     try:
#         response = await original_create_method(*args, **kwargs)
#         print("API RESPONSE TYPE:", type(response))
#         print("API RESPONSE DIR:", dir(response))
#         # Try to safely access and print properties
#         if hasattr(response, "model_dump"):
#             print("RESPONSE DUMP:", json.dumps(response.model_dump(), indent=2))
#         return response
#     except Exception as e:
#         print(f"ERROR: {e}")
#         # Return raw exception info for debugging
#         raise


# setattr(client._client.chat.completions, "create", debug_create)


# # Test the client
# async def test() -> None:
#     result = await client.create([UserMessage(content="What is 2+2?", source="user")])
#     print("FINAL RESULT:", result)


# # asyncio.run(test())


# def calculator(a: int, b: int, operation: str) -> str:
#     """Simple calculator function"""
#     if operation == "add":
#         return str(a + b)
#     elif operation == "multiply":
#         return str(a * b)
#     return "Unsupported operation"


# calculator_tool = FunctionTool(
#     calculator, description="A simple calculator that can add or multiply two numbers"
# )


# async def test_with_tools() -> None:
#     # First call to request function execution
#     result = await client.create(
#         [UserMessage(content="What is 10 multiplied by 5?", source="user")],
#         tools=[calculator_tool],
#     )
#     print("TOOL CALL RESULT:", result)

#     # If successful, process the second call with the result
#     if isinstance(result.content, list):
#         print("FUNCTION CALL DETECTED")
#         # Processing would continue in real implementation
#     else:
#         print("NO FUNCTION CALL MADE")


# # asyncio.run(test_with_tools())


# async def test_complex_interaction():
#     """Test a multi-turn conversation with tool usage that better mimics the MAS system"""

#     # First turn: Ask a question that needs a tool
#     result1 = await client.create(
#         [UserMessage(content="What is 10 multiplied by 5?", source="user")],
#         tools=[calculator_tool],
#     )
#     print("FIRST CALL RESULT:", result1)

#     # Second turn: Send back the result of the tool call
#     from autogen_core.models import (
#         AssistantMessage,
#         FunctionExecutionResultMessage,
#         FunctionExecutionResult,
#     )

#     if isinstance(result1.content, list) and len(result1.content) > 0:
#         func_call = result1.content[0]

#         # Create a conversation history
#         result2 = await client.create(
#             [
#                 UserMessage(content="What is 10 multiplied by 5?", source="user"),
#                 AssistantMessage(
#                     content=result1.content, thought=result1.thought, source="assistant"
#                 ),
#                 FunctionExecutionResultMessage(
#                     content=[
#                         FunctionExecutionResult(
#                             content="50",
#                             call_id=func_call.id,
#                             is_error=False,
#                             name=func_call.name,
#                         )
#                     ]
#                 ),
#             ],
#             tools=[calculator_tool],
#         )
#         print("SECOND CALL RESULT:", result2)


# asyncio.run(test_complex_interaction())

import asyncio
import json
import os
from typing import Any, Dict, Optional

from autogen_core.models import (
    UserMessage,
    AssistantMessage,
    FunctionExecutionResultMessage,
    FunctionExecutionResult,
)
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Create a client that uses OpenRouter for Nova
client = OpenAIChatCompletionClient(
    model="amazon/nova-pro-v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    model_info={
        "family": "unknown",
        "vision": True,
        "function_calling": True,
        "json_output": True,
    },
)

# Patch the client's internal methods to debug API responses
original_create_method = client._client.chat.completions.create


async def debug_create(*args: Any, **kwargs: Any) -> Any:
    print("API CALL ARGS:", json.dumps(kwargs, indent=2))
    try:
        response = await original_create_method(*args, **kwargs)
        print("API RESPONSE TYPE:", type(response))
        print("API RESPONSE DUMP:", json.dumps(response.model_dump(), indent=2))
        return response
    except Exception as e:
        print(f"ERROR: {e}")
        raise


setattr(client._client.chat.completions, "create", debug_create)


# Create scroll tools with dummy arguments
def scroll_down(dummy: Optional[str] = None) -> str:
    """Scroll down in the terminal to see more content."""
    return "Scrolled down successfully."


def scroll_up(dummy: Optional[str] = None) -> str:
    """Scroll up in the terminal to see previous content."""
    return "Scrolled up successfully."


# Create function tools
scroll_down_tool = FunctionTool(
    scroll_down, description="Scroll down in the terminal to see more content"
)

scroll_up_tool = FunctionTool(
    scroll_up, description="Scroll up in the terminal to see previous content"
)


# Test tools separately
async def test_scroll_down_with_dummy():
    """Test how Nova handles scroll_down with dummy args"""
    print("\n=== TESTING SCROLL DOWN WITH DUMMY ARG ===")
    result = await client.create(
        [UserMessage(content="Please scroll down to show me more text", source="user")],
        tools=[scroll_down_tool],
    )
    print("SCROLL DOWN RESULT:", result)

    # Detailed analysis of function call
    if (
        hasattr(result, "content")
        and isinstance(result.content, list)
        and len(result.content) > 0
    ):
        func_call = result.content[0]
        print(f"\nFUNCTION CALL ANALYSIS:")
        print(f"Name: {func_call.name}")
        print(f"Arguments (type): {type(func_call.arguments)}")
        print(f"Arguments (value): {repr(func_call.arguments)}")

        # Try parsing the arguments
        try:
            if isinstance(func_call.arguments, str):
                args_dict = json.loads(func_call.arguments)
                print(f"Parsed arguments: {args_dict}")
                print(f"Has dummy key: {'dummy' in args_dict}")
                print(f"Dummy value: {args_dict.get('dummy')}")
        except json.JSONDecodeError:
            print("Failed to parse arguments as JSON")


# Test in simulated agent context
async def test_agent_scroll_conversation():
    """Test a multi-turn conversation that simulates agent behavior"""
    print("\n=== TESTING AGENT-LIKE CONVERSATION WITH SCROLL TOOLS ===")

    # First turn: Ask to scroll down
    result1 = await client.create(
        [
            UserMessage(
                content="I need to see more of the code file. Please scroll down.",
                source="user",
            )
        ],
        tools=[scroll_down_tool, scroll_up_tool],
    )
    print("FIRST CALL RESULT:", result1)

    # Extract function call details
    if (
        hasattr(result1, "content")
        and isinstance(result1.content, list)
        and len(result1.content) > 0
    ):
        func_call = result1.content[0]

        # Create a conversation history with function result
        print("\n=== SECOND TURN: RETURNING FUNCTION RESULT ===")
        result2 = await client.create(
            [
                UserMessage(
                    content="I need to see more of the code file. Please scroll down.",
                    source="user",
                ),
                AssistantMessage(
                    content=result1.content, thought=result1.thought, source="assistant"
                ),
                FunctionExecutionResultMessage(
                    content=[
                        FunctionExecutionResult(
                            content="Scrolled down successfully. Here is the next page of code...",
                            call_id=func_call.id,
                            is_error=False,
                            name=func_call.name,
                        )
                    ]
                ),
            ],
            tools=[scroll_down_tool, scroll_up_tool],
        )
        print("SECOND CALL RESULT:", result2)


# Test both directions in sequence
async def test_up_then_down():
    """Test scrolling up then down in sequence"""
    print("\n=== TESTING SCROLL UP THEN DOWN IN SEQUENCE ===")

    initial_messages = [
        UserMessage(
            content="I'm looking at a code file. First scroll up to see imports, then scroll down to see functions.",
            source="user",
        )
    ]

    # First call should trigger scroll_up
    result1 = await client.create(
        initial_messages,
        tools=[scroll_up_tool, scroll_down_tool],
    )
    print("FIRST DIRECTION RESULT:", result1)

    if (
        hasattr(result1, "content")
        and isinstance(result1.content, list)
        and len(result1.content) > 0
    ):
        func_call1 = result1.content[0]

        # Add first result
        messages = initial_messages.copy()
        messages.append(
            AssistantMessage(
                content=result1.content, thought=result1.thought, source="assistant"
            )
        )
        messages.append(
            FunctionExecutionResultMessage(
                content=[
                    FunctionExecutionResult(
                        content="Scrolled up successfully. You can now see the import statements.",
                        call_id=func_call1.id,
                        is_error=False,
                        name=func_call1.name,
                    )
                ]
            )
        )

        # Second call should trigger scroll_down
        result2 = await client.create(
            messages,
            tools=[scroll_up_tool, scroll_down_tool],
        )
        print("\nSECOND DIRECTION RESULT:", result2)


async def run_all_tests():
    await test_scroll_down_with_dummy()
    await test_agent_scroll_conversation()
    await test_up_then_down()


# Run all tests
asyncio.run(run_all_tests())
