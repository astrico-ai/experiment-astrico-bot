"""
GPT-5.2 client wrapper with reasoning support.
"""
import json
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import time

from ..config import settings

logger = logging.getLogger(__name__)

# Create separate logger for LLM debugging
llm_debug_logger = logging.getLogger("llm_debug")
llm_debug_logger.setLevel(logging.DEBUG)

# Create file handler for LLM debug logs
llm_handler = logging.FileHandler("llm_debug.log")
llm_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
llm_handler.setFormatter(formatter)

# Add handler to logger
llm_debug_logger.addHandler(llm_handler)


class LLMClient:
    """Wrapper for GPT-5.2 API with reasoning capabilities."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None,
        reasoning_effort: str = None,
        use_openrouter: bool = None
    ):
        """
        Initialize LLM client.

        Args:
            api_key: API key (defaults to settings)
            model: Model to use (defaults to settings)
            reasoning_effort: Reasoning effort level (defaults to settings)
            use_openrouter: If True, use OpenRouter instead of direct OpenAI (defaults to settings)
        """
        self.model = model or settings.llm_model
        self.reasoning_effort = reasoning_effort or settings.reasoning_effort
        self.max_tokens = settings.max_tokens
        self.use_openrouter = use_openrouter if use_openrouter is not None else settings.use_openrouter

        if self.use_openrouter:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key or settings.openrouter_api_key,
                timeout=120.0,  # 120 second timeout for connection and read
                default_headers={
                    "HTTP-Referer": "https://experiment-bot.local",
                    "X-Title": "Experiment Bot"
                }
            )
            logger.info(f"LLM client initialized with OpenRouter: model={self.model}, timeout=120s (App: Experiment Bot)")
        else:
            self.client = OpenAI(
                api_key=api_key or settings.openai_api_key,
                timeout=120.0  # 120 second timeout for connection and read
            )
            logger.info(f"LLM client initialized with OpenAI: model={self.model}, timeout=120s")

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        reasoning_effort: Optional[str] = None,
        stream: bool = False
    ) -> Any:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            reasoning_effort: Override default reasoning effort
            stream: If True, return streaming response

        Returns:
            OpenAI ChatCompletion response object or stream
        """
        effort = reasoning_effort or self.reasoning_effort

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "stream": stream
            }

            if tools:
                kwargs["tools"] = tools

            # Add reasoning_effort for OpenRouter using extra_body
            # For non-reasoning models, explicitly set to "none" to avoid overhead
            if self.use_openrouter:
                kwargs["extra_body"] = {
                    "reasoning": {
                        "effort": "none"  # Disable reasoning for fast inference
                    }
                }

            # LOG REQUEST (summary only)
            start_time = time.time()

            # Get system prompt length for monitoring
            system_messages = [m for m in messages if m.get('role') == 'system']
            system_prompt_len = len(system_messages[0].get('content', '')) if system_messages else 0

            llm_debug_logger.info("="*80)
            llm_debug_logger.info(f"REQUEST - Model: {self.model}, Stream: {stream}")
            llm_debug_logger.info(f"Messages: {len(messages)}, Tools: {len(tools) if tools else 0}, System prompt: {system_prompt_len} chars")

            response = self.client.chat.completions.create(**kwargs)

            end_time = time.time()
            elapsed = end_time - start_time

            if not stream:
                # LOG NON-STREAMING RESPONSE (minimal)
                llm_debug_logger.info(f"RESPONSE - Time: {elapsed:.3f}s")

                if hasattr(response, 'choices') and response.choices:
                    choice = response.choices[0]
                    llm_debug_logger.info(f"Finish: {choice.finish_reason}")

                    if choice.message.content:
                        llm_debug_logger.info(f"Content: {len(choice.message.content)} chars")

                    if choice.message.tool_calls:
                        tool_names = [tc.function.name for tc in choice.message.tool_calls]
                        llm_debug_logger.info(f"Tool calls: {', '.join(tool_names)}")

                if hasattr(response, 'usage'):
                    llm_debug_logger.info(f"Tokens - Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")

                llm_debug_logger.info("="*80)
                logger.info(f"LLM request completed. Model: {self.model}, Time: {elapsed:.3f}s")
            else:
                llm_debug_logger.info(f"LLM streaming started - Time to first chunk: {elapsed:.3f}s")
                llm_debug_logger.info("="*80)
                logger.info(f"LLM streaming started. Model: {self.model}")

            return response

        except TimeoutError as e:
            llm_debug_logger.error(f"LLM request timed out after 120s: {e}")
            logger.error(f"LLM request timed out after 120s. Model: {self.model}")
            raise TimeoutError(f"LLM request timed out after 120 seconds. This may indicate API unavailability or network issues.") from e
        except Exception as e:
            llm_debug_logger.error(f"LLM request failed: {e}")
            logger.error(f"LLM request failed: {e}")
            raise

    def chat_with_reasoning(
        self,
        system_prompt: str,
        user_prompt: str,
        reasoning_effort: Optional[str] = None
    ) -> tuple[str, Optional[str]]:
        """
        Simple chat with reasoning, return content and reasoning details.

        Args:
            system_prompt: System message
            user_prompt: User message
            reasoning_effort: Override default reasoning effort

        Returns:
            tuple: (response_content, reasoning_details)
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.chat(messages, reasoning_effort=reasoning_effort)
        message = response.choices[0].message

        return message.content, None  # GPT-5.2 may expose reasoning in future

    def compact_context(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Use GPT-5.2's context compaction to reduce token footprint.

        Args:
            messages: Long conversation history

        Returns:
            List[Dict]: Compacted messages
        """
        try:
            compact_response = self.client.responses.compact.create(
                model=self.model,
                messages=messages
            )
            logger.info("Context compacted successfully")
            return compact_response.compacted_messages
        except Exception as e:
            logger.warning(f"Context compaction failed, returning original: {e}")
            return messages

    def parse_json_response(self, content: str) -> Optional[dict]:
        """
        Parse JSON from LLM response, handling markdown code blocks.

        Args:
            content: LLM response content

        Returns:
            dict or None if parsing fails
        """
        try:
            # Try direct JSON parse first
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                json_str = content[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

            logger.error(f"Failed to parse JSON from LLM response: {content[:200]}...")
            return None


# Global client instance
llm_client = LLMClient()
