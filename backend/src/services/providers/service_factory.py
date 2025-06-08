"""
Factory functions for creating STT, LLM, and TTS services.
"""
from typing import Optional, Tuple, Any
import os
import logging

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transcriptions.language import Language

from src.services.providers.deepinfra_llm import DeepInfraLLMService
from src.services.providers.deepinfra_tts import DeepInfraHttpTTSService

logger = logging.getLogger(__name__)


def create_stt_service(service_name: str, **kwargs):
    """Create the appropriate STT service"""
    if service_name == "deepgram":
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")
        
        model = kwargs.get("model", "nova-2")
        
        from deepgram import LiveOptions
        
        live_options = LiveOptions(
            model=model,
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            interim_results=True,
            utterance_end_ms=kwargs.get("utterance_end_ms", 1000),
            vad_events=kwargs.get("vad_events", True),
            smart_format=kwargs.get("smart_format", True),
            punctuate=kwargs.get("punctuate", True)
        )
        
        return DeepgramSTTService(
            api_key=api_key,
            live_options=live_options
        )
    
    elif service_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        model = kwargs.get("model", "whisper-1")
        language = kwargs.get("language", Language.EN)
        
        return OpenAISTTService(
            api_key=api_key,
            model=model,
            language=language
        )
    
    else:
        raise ValueError(f"Unknown STT service: {service_name}")


def create_llm_service(service_name: str, **kwargs):
    """Create the appropriate LLM service"""
    if service_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        model = kwargs.get("model", "gpt-3.5-turbo")
        
        return OpenAILLMService(
            api_key=api_key,
            model=model
        )
    
    elif service_name == "deepinfra":
        api_key = os.getenv("DEEPINFRA_API_KEY")
        if not api_key:
            raise ValueError("DEEPINFRA_API_KEY not set")
        
        model = kwargs.get("model", "meta-llama/Meta-Llama-3.1-70B-Instruct")
        base_url = kwargs.get("base_url", "https://api.deepinfra.com/v1/openai")
        
        params_dict = {
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
            "stop": kwargs.get("stop", None),
            "stream": kwargs.get("stream", True),
            "extra": kwargs.get("extra", {})
        }
        
        params = DeepInfraLLMService.InputParams(**params_dict)
        
        return DeepInfraLLMService(
            api_key=api_key,
            model=model,
            base_url=base_url,
            params=params
        )
    
    else:
        raise ValueError(f"Unknown LLM service: {service_name}")


def create_tts_service(service_name: str, **kwargs) -> Tuple[Any, int]:
    """
    Create the appropriate TTS service.
    Returns (service, output_sample_rate)
    """
    if service_name == "elevenlabs":
        api_key = os.getenv("ELEVEN_API_KEY")
        if not api_key:
            raise ValueError("ELEVEN_API_KEY not set")

        return ElevenLabsTTSService(
            api_key=api_key,
            voice_id=kwargs.get("voice_id", "21m00Tcm4TlvDq8ikWAM"),
            model=kwargs.get("model", "eleven_flash_v2_5"),
            sample_rate=kwargs.get("sample_rate", 16000),
            params=kwargs.get("params"),
        ), kwargs.get("sample_rate", 16000)

    elif service_name == "deepinfra":
        api_key = os.getenv("DEEPINFRA_API_KEY")
        if not api_key:
            raise ValueError("DEEPINFRA_API_KEY not set")

        aiohttp_session = kwargs.get("aiohttp_session")
        if aiohttp_session is None:
            raise ValueError("create_tts_service('deepinfra') needs aiohttp_session=<ClientSession>")

        sample_rate = kwargs.get("sample_rate", 24000)
        
        return DeepInfraHttpTTSService(
            api_key=api_key,
            voice_id=kwargs.get("voice_id", "af_bella"),
            model=kwargs.get("model", "hexgrad/Kokoro-82M"),
            aiohttp_session=aiohttp_session,
            sample_rate=sample_rate,
            params=kwargs.get("params"),
        ), sample_rate

    else:
        raise ValueError(f"Unknown TTS service: {service_name}")


def create_llm_context(llm_service_type: str, system_prompt: str = None, **kwargs) -> OpenAILLMContext:
    """Create the appropriate LLM context based on the service type"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    initial_messages = kwargs.get("initial_messages", [])
    messages.extend(initial_messages)
    
    return OpenAILLMContext(messages)