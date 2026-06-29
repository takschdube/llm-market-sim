# src/agents/llm_client.py
"""
Unified LLM client for all providers.

This module provides a thin abstraction over multiple LLM providers,
keeping the interface simple while allowing easy provider switching.

Supported Providers:
- DeepSeek (default, cost-efficient)
- Anthropic (Claude models)
- OpenAI (GPT models)
- Google (Gemini models)

Example:
    client = LLMClient("anthropic", "claude-sonnet-4-5-20250929")
    response = client.call(
        system="You are a trader.",
        user="What should I buy?",
        max_tokens=256
    )
"""
from __future__ import annotations

import os
from typing import Literal, Optional

# Type alias for supported providers
LLMProvider = Literal["anthropic", "deepseek", "openai", "google"]

# Default models by provider
DEFAULT_MODELS = {
    "deepseek": "deepseek-chat",
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-5.2",
    "google": "gemini-2.0-flash",
}

# Alternative models for experimentation
ALTERNATIVE_MODELS = {
    "deepseek": {
        "deepseek-chat": "DeepSeek V3 - fast, cost-efficient (default)",
        "deepseek-reasoner": "DeepSeek R1 - reasoning model",
    },
    "anthropic": {
        "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5 - flagship tier (default)",
        "claude-opus-4-20250514": "Claude Opus 4.5 - highest capability",
    },
    "openai": {
        "gpt-5.2": "GPT-5.2 - flagship tier (default)",
        "gpt-5-mini": "GPT-5 Mini - faster variant",
    },
    "google": {
        "gemini-2.0-flash": "Gemini 2.0 Flash - fast multimodal (default)",
        "gemini-1.5-pro": "Gemini 1.5 Pro - larger context window",
    },
}


class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    This is a thin wrapper that normalizes the interface across providers.
    Each provider has its own SDK, but they all expose the same call() method.
    Supports both sync and async calls for parallel execution.

    Attributes:
        provider: Which LLM provider to use
        model: Model name/ID
    """

    def __init__(self, provider: LLMProvider = "deepseek", model: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            provider: LLM provider ("deepseek", "anthropic", "openai", "google")
            model: Model name. If None, uses default for provider.

        Raises:
            ValueError: If required API key is not set
        """
        self.provider = provider
        self.model = model or DEFAULT_MODELS.get(provider, "deepseek-chat")
        self._client = None
        self._async_client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the provider-specific client."""
        if self.provider == "anthropic":
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable is not set. "
                    "Get your key from: https://console.anthropic.com/settings/keys"
                )
            self._client = anthropic.Anthropic(api_key=api_key)

        elif self.provider == "deepseek":
            from openai import OpenAI
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY environment variable is not set. "
                    "Get your key from: https://platform.deepseek.com/api_keys"
                )
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

        elif self.provider == "openai":
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Get your key from: https://platform.openai.com/api-keys"
                )
            self._client = OpenAI(api_key=api_key)

        elif self.provider == "google":
            import google.generativeai as genai
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable is not set. "
                    "Get your key from: https://aistudio.google.com/apikey"
                )
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model)

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def call(self, system: str, user: str, max_tokens: int = 256, temperature: float = 1.0) -> str:
        """
        Make an LLM call and return the response text.

        Args:
            system: System prompt (instructions)
            user: User message (the actual query)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (default 1.0 for stochastic outputs)

        Returns:
            Response text from the model
        """
        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            return response.content[0].text

        elif self.provider in ("deepseek", "openai"):
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            )
            return response.choices[0].message.content

        elif self.provider == "google":
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            # Gemini combines system and user prompts
            combined_prompt = f"{system}\n\n{user}"
            response = self._client.generate_content(
                combined_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                ),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            # Handle blocked responses
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                raise ValueError(f"Gemini blocked response (finish_reason={finish_reason})")
            return response.text

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _ensure_async_client(self) -> None:
        """Lazily initialize async client on first async call."""
        if self._async_client is not None:
            return

        if self.provider == "anthropic":
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            self._async_client = anthropic.AsyncAnthropic(api_key=api_key)

        elif self.provider in ("deepseek", "openai"):
            from openai import AsyncOpenAI
            if self.provider == "deepseek":
                api_key = os.environ.get("DEEPSEEK_API_KEY")
                self._async_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com"
                )
            else:
                api_key = os.environ.get("OPENAI_API_KEY")
                self._async_client = AsyncOpenAI(api_key=api_key)

        elif self.provider == "google":
            # Google's genai doesn't have a separate async client
            # We'll use the sync client in an executor
            pass

    async def call_async(self, system: str, user: str, max_tokens: int = 256, temperature: float = 1.0) -> str:
        """
        Make an async LLM call and return the response text.

        Args:
            system: System prompt (instructions)
            user: User message (the actual query)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (default 1.0 for stochastic outputs)

        Returns:
            Response text from the model
        """
        self._ensure_async_client()

        if self.provider == "anthropic":
            response = await self._async_client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            return response.content[0].text

        elif self.provider in ("deepseek", "openai"):
            response = await self._async_client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            )
            return response.choices[0].message.content

        elif self.provider == "google":
            # Google doesn't have native async, use sync in thread
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.call(system, user, max_tokens, temperature)
            )

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def __repr__(self) -> str:
        return f"LLMClient(provider='{self.provider}', model='{self.model}')"
