# backend/src/services/deepinfra_llm.py
"""
DeepInfra LLM Service for Pipecat
Provides access to Llama and other models through DeepInfra's API
"""
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallCancelFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    UserImageRawFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMAssistantContextAggregator,
    LLMUserAggregatorParams,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.utils.tracing.service_decorators import traced_llm


@dataclass
class DeepInfraContextAggregatorPair:
    _user: "DeepInfraUserContextAggregator"
    _assistant: "DeepInfraAssistantContextAggregator"

    def user(self) -> "DeepInfraUserContextAggregator":
        return self._user

    def assistant(self) -> "DeepInfraAssistantContextAggregator":
        return self._assistant


class DeepInfraLLMService(LLMService):
    """This class implements inference with DeepInfra's AI models.
    
    DeepInfra supports various open-source models including Llama.
    The API is OpenAI-compatible, making integration straightforward.
    """

    class InputParams(BaseModel):
        max_tokens: Optional[int] = Field(default_factory=lambda: 4096, ge=1)
        temperature: Optional[float] = Field(default_factory=lambda: 0.7, ge=0.0, le=2.0)
        top_p: Optional[float] = Field(default_factory=lambda: 1.0, ge=0.0, le=1.0)
        frequency_penalty: Optional[float] = Field(default_factory=lambda: 0.0, ge=-2.0, le=2.0)
        presence_penalty: Optional[float] = Field(default_factory=lambda: 0.0, ge=-2.0, le=2.0)
        stop: Optional[List[str]] = Field(default=None)
        stream: Optional[bool] = Field(default=True)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        base_url: str = "https://api.deepinfra.com/v1/openai",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._base_url = base_url
        self.set_model_name(model)
        
        params = params or DeepInfraLLMService.InputParams()
        self._settings = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
            "stop": params.stop,
            "stream": params.stream,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }
        
        # Create HTTP client with longer timeout for large models
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            },
            timeout=httpx.Timeout(60.0, connect=10.0)
        )

    def can_generate_metrics(self) -> bool:
        return True

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> DeepInfraContextAggregatorPair:
        """Create an instance of DeepInfraContextAggregatorPair from an OpenAILLMContext."""
        context.set_llm_adapter(self.get_llm_adapter())
        user = DeepInfraUserContextAggregator(context, params=user_params)
        assistant = DeepInfraAssistantContextAggregator(context, params=assistant_params)
        return DeepInfraContextAggregatorPair(_user=user, _assistant=assistant)

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext):
        # Token usage tracking
        prompt_tokens = 0
        completion_tokens = 0
        completion_tokens_estimate = 0
        use_completion_tokens_estimate = False

        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()

            # Extract system message from context
            system_message = None
            messages = []
            
            # Check if first message is a system message
            if context.messages and context.messages[0].get("role") == "system":
                system_message = context.messages[0].get("content", "")
                messages = context.messages[1:]  # Skip the system message
            else:
                messages = context.messages

            logger.debug(
                f"{self}: Generating chat [system: {system_message}] | [{context.get_messages_for_logging()}]"
            )

            # Build final messages list
            final_messages = []
            if system_message:
                final_messages.append({"role": "system", "content": system_message})
            final_messages.extend(messages)

            await self.start_ttfb_metrics()

            params = {
                "model": self.model_name,
                "messages": final_messages,
                "max_tokens": self._settings["max_tokens"],
                "temperature": self._settings["temperature"],
                "top_p": self._settings["top_p"],
                "frequency_penalty": self._settings["frequency_penalty"],
                "presence_penalty": self._settings["presence_penalty"],
                "stream": self._settings["stream"],
            }

            if self._settings["stop"]:
                params["stop"] = self._settings["stop"]
                
            if context.tools:
                params["tools"] = context.tools
                # DeepInfra supports "none" and "auto" for tool_choice
                # Default to "auto" if tools are present
                params["tool_choice"] = "auto"
                
            params.update(self._settings["extra"])

            # Make streaming request
            response = self._client.stream(
                "POST",
                "/chat/completions",
                json=params
            )

            await self.stop_ttfb_metrics()

            # Process streaming response
            async with response as stream:
                async for line in stream.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                            
                        try:
                            chunk = json.loads(data)
                            
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                choice = chunk["choices"][0]
                                delta = choice.get("delta", {})
                                
                                # Handle text content
                                if "content" in delta and delta["content"]:
                                    content = delta["content"]
                                    # Skip end-of-sequence tokens
                                    if content != "</s>":
                                        await self.push_frame(LLMTextFrame(content))
                                        completion_tokens_estimate += self._estimate_tokens(content)
                                
                                # Handle function calls (if supported by the model)
                                if "tool_calls" in delta and delta["tool_calls"]:
                                    for tool_call in delta["tool_calls"]:
                                        if tool_call and "function" in tool_call:
                                            await self._handle_function_call(
                                                context, tool_call["function"]
                                            )
                                
                                # Check for finish reason
                                finish_reason = choice.get("finish_reason")
                                if finish_reason and finish_reason != "null":
                                    logger.debug(f"Stream finish reason: {finish_reason}")
                                    
                                    # If we have usage data in the final chunk
                                    if "usage" in chunk:
                                        usage = chunk["usage"]
                                        if isinstance(usage, dict):
                                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                            completion_tokens = usage.get("completion_tokens", completion_tokens)
                                        else:
                                            prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens)
                                            completion_tokens = getattr(usage, "completion_tokens", completion_tokens)
                            
                            # Track usage if provided
                            if "usage" in chunk:
                                usage = chunk["usage"]
                                if isinstance(usage, dict):
                                    prompt_tokens = usage.get("prompt_tokens", 0)
                                    completion_tokens = usage.get("completion_tokens", 0)
                                else:
                                    # Handle object-style access
                                    prompt_tokens = getattr(usage, "prompt_tokens", 0)
                                    completion_tokens = getattr(usage, "completion_tokens", 0)
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse streaming chunk: {data}")

        except asyncio.CancelledError:
            use_completion_tokens_estimate = True
            raise
        except httpx.TimeoutException:
            await self._call_event_handler("on_completion_timeout")
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
            comp_tokens = (
                completion_tokens
                if not use_completion_tokens_estimate
                else completion_tokens_estimate
            )
            await self._report_usage_metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens
            )

    async def _handle_function_call(self, context: OpenAILLMContext, function_data: dict):
        """Handle function call from the model"""
        if "name" in function_data and "arguments" in function_data:
            try:
                arguments = json.loads(function_data["arguments"])
                tool_call_id = f"call_{self._generate_call_id()}"
                
                await self.call_function(
                    context=context,
                    tool_call_id=tool_call_id,
                    function_name=function_data["name"],
                    arguments=arguments,
                )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse function arguments: {function_data['arguments']}")

    def _generate_call_id(self) -> str:
        """Generate a unique call ID for function calls"""
        import uuid
        return str(uuid.uuid4())[:8]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a piece of text"""
        return int(len(re.split(r"[^\w]+", text)) * 1.3)

    async def _report_usage_metrics(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        """Report token usage metrics"""
        if prompt_tokens or completion_tokens:
            tokens = LLMTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            await self.start_llm_usage_metrics(tokens)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = OpenAILLMContext.from_image_frame(frame)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)

    async def _update_settings(self, settings: Dict[str, Any]):
        """Update settings from a frame"""
        for key, value in settings.items():
            if key in self._settings:
                logger.debug(f"Updating setting {key} to {value}")
                self._settings[key] = value


class DeepInfraUserContextAggregator(LLMUserContextAggregator):
    pass


class DeepInfraAssistantContextAggregator(LLMAssistantContextAggregator):
    
    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        self._context.add_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": frame.tool_call_id,
                        "function": {
                            "name": frame.function_name,
                            "arguments": json.dumps(frame.arguments),
                        },
                        "type": "function",
                    }
                ],
            }
        )
        self._context.add_message(
            {
                "role": "tool",
                "content": "IN_PROGRESS",
                "tool_call_id": frame.tool_call_id,
            }
        )

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        if frame.result:
            result = json.dumps(frame.result)
            await self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
        else:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, "COMPLETED"
            )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        await self._update_function_call_result(
            frame.function_name, frame.tool_call_id, "CANCELLED"
        )

    async def _update_function_call_result(
        self, function_name: str, tool_call_id: str, result: Any
    ):
        for message in self._context.messages:
            if (
                message.get("role") == "tool"
                and message.get("tool_call_id") == tool_call_id
            ):
                message["content"] = result

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        await self._update_function_call_result(
            frame.request.function_name, frame.request.tool_call_id, "COMPLETED"
        )
        self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.request.context,
        )