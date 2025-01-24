# swarm/huggingface.py
import os
import json
import logging
from typing import Any, Dict, List, Union, Generator

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

class HuggingFaceClient:
    """
    A Hugging Face client that uses huggingface_hub's InferenceClient
    to replicate an OpenAI-like interface for Swarm, including tool usage.
    """

    def __init__(
        self,
        api_token: str = None,
        default_model: str = None,
        timeout: int = 60
    ):
        """
        Args:
            api_token (str): Hugging Face token. If omitted, uses HF_API_TOKEN from env.
            default_model (str): E.g. "Qwen/Qwen2.5-1.5B-Instruct"
            timeout (int): Request timeout in seconds.
        """
        # Let environment override the constructor if not provided
        self.api_token = api_token or os.getenv("HF_API_TOKEN", "")
        self.default_model = default_model or os.getenv("HF_DEFAULT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        self.timeout = timeout

        # Create the huggingface_hub client
        self._client = InferenceClient(api_key=self.api_token, timeout=self.timeout)
        self._completions = self._Completions(self._client, self.default_model)

    @property
    def chat(self):
        # For usage: self.client.chat.completions.create(...)
        return self

    @property
    def completions(self):
        # Returns an object on which we can call .create(...)
        return self._completions

    class _Completions:
        def __init__(self, hf_client: InferenceClient, default_model: str):
            self.hf_client = hf_client
            self.default_model = default_model

        def create(
            self,
            model: str = None,
            messages: List[Dict[str, str]] = None,
            stream: bool = False,
            tools: Union[List[Dict], None] = None,
            tool_choice: Union[str, None] = None,
            parallel_tool_calls: bool = True,
            **kwargs
        ) -> Any:
            """
            Create a chat completion using huggingface_hub.InferenceClient.
            The 'stream' parameter is ignored as HF doesn't provide SSE streaming for chat (yet).
            
            If you pass "tools" and "tool_choice='auto'", the model may attempt function calls.
            """
            # If no model specified, fall back to default_model
            effective_model = model or self.default_model

            try:
                response = self.hf_client.chat.completions.create(
                    model=effective_model,
                    messages=messages or [],
                    tools=tools if tools else None,          # pass tools if provided
                    tool_choice=tool_choice if tool_choice else None,
                    **kwargs  # e.g., max_tokens, temperature, etc.
                )
            except Exception as e:
                logger.error(f"HuggingFace chat completion failed: {e}")
                raise

            # Convert to an OpenAI-like object so Swarm can parse .choices[0].message
            return self._hf_to_openai_response(response)

        def _hf_to_openai_response(self, hf_response: Any) -> Any:
            """
            Convert huggingface_hub InferenceClient response into {choices: [ {message: ...} ]}
            so that Swarm can read completion.choices[0].message.
            """
            if not hasattr(hf_response, "choices"):
                return {
                    "choices": [
                        {"message": {"role": "assistant", "content": "No content"}}
                    ]
                }

            class MockChoice:
                def __init__(self, role, content, tool_calls=None):
                    self.message = MockMessage(role, content, tool_calls)

            class MockMessage:
                def __init__(self, role, content, tool_calls):
                    self.role = role
                    self.content = content
                    # Hugging Face might return .tool_calls inside each choice
                    self.tool_calls = tool_calls

                def model_dump_json(self):
                    return json.dumps({
                        "role": self.role,
                        "content": self.content,
                        "tool_calls": self.tool_calls
                    })

            class MockOpenAIResponse:
                def __init__(self, hf_resp):
                    self.choices = []
                    for c in hf_resp.choices:
                        role = c.message.role
                        content = c.message.content
                        # If the model performed a function call, huggingface_hub sets c.message.tool_calls
                        # We'll pass it along
                        tool_calls = getattr(c.message, "tool_calls", None)
                        self.choices.append(MockChoice(role, content, tool_calls))

            return MockOpenAIResponse(hf_response)
