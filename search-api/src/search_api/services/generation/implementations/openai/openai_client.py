"""OpenAI LLM client implementation."""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from openai import AzureOpenAI
from ...abstractions.llm_client import LLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """OpenAI implementation of the LLM client."""
    
    def __init__(self):
        """Initialize the OpenAI client with Azure configuration."""
        self.client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        # Check both possible environment variable names for deployment
        self.deployment_name = (
            os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") or 
            os.environ.get("AZURE_OPENAI_DEPLOYMENT") or 
            "gpt-4"  # Default to gpt-4 instead of gpt-4o
        )
        
    def chat_completions_create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        **kwargs
    ) -> Any:
        """Create a chat completion using Azure OpenAI.
        
        Args:
            model: The model name (will use configured deployment).
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature (0.0 to 2.0).
            **kwargs: Additional parameters (max_tokens, tools, tool_choice, etc.).
            
        Returns:
            Chat completion response object.
            
        Raises:
            Exception: If the API request fails.
        """
        return self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=kwargs.get('max_tokens'),
            tools=kwargs.get('tools'),
            tool_choice=kwargs.get('tool_choice')
        )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a chat completion request to OpenAI.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            tools: List of tool definitions for function calling.
            tool_choice: Tool choice strategy ("auto", "none", or specific tool).
            
        Returns:
            Dict containing the response data.
            
        Raises:
            Exception: If the API request fails.
        """
        try:
            kwargs = {
                "model": model or self.deployment_name,
                "messages": messages,
                "temperature": temperature
            }
            
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
                
            if tools:
                kwargs["tools"] = tools
                
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            
            logger.info(f"Sending chat completion request to OpenAI with {len(messages)} messages")
            response = self.client.chat.completions.create(**kwargs)
            
            # Convert response to dictionary format
            result = {
                "choices": []
            }
            
            for choice in response.choices:
                choice_dict = {
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content
                    },
                    "finish_reason": choice.finish_reason
                }
                
                # Add tool calls if present
                if choice.message.tool_calls:
                    choice_dict["message"]["tool_calls"] = []
                    for tool_call in choice.message.tool_calls:
                        tool_call_dict = {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                        choice_dict["message"]["tool_calls"].append(tool_call_dict)
                
                result["choices"].append(choice_dict)
            
            logger.info("Chat completion request completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI chat completion failed: {str(e)}")
            raise
    
    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ):
        """Stream a chat completion, yielding text tokens as they arrive.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            model: Optional deployment override (defaults to self.deployment_name).

        Yields:
            str: Text token chunks as they stream from the API.
        """
        kwargs = {
            "model": model or self.deployment_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        logger.info(f"Starting streaming chat completion ({kwargs['model']})")
        response = self.client.chat.completions.create(**kwargs)
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "openai"
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.deployment_name