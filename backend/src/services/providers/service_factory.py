"""
Factory functions for creating STT, LLM, and TTS services.
"""
from typing import Optional, Tuple, Any
import os
import logging

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.together.llm import TogetherLLMService
from pipecat.services.riva.tts import RivaTTSService
from pipecat.services.rime.tts import RimeTTSService
from pipecat.services.riva.stt import RivaSTTService
from pipecat.services.riva.stt import RivaSegmentedSTTService
from pipecat.transcriptions.language import Language

from src.services.providers.deepinfra_llm import DeepInfraLLMService
from src.services.providers.deepinfra_tts import DeepInfraHttpTTSService
from src.services.providers.speechify_tts import SpeechifyTTSService
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.groq.tts import GroqTTSService

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
        
        model = kwargs.get("model", "gpt-4o-transcribe")
        language = kwargs.get("language", Language.EN)
        
        return OpenAISTTService(
            api_key=api_key,
            model=model,
            language=language
        )
    
    elif service_name == "assemblyai":
        api_key = os.getenv("ASSEMBLY_API_KEY")
        if not api_key:
            raise ValueError("ASSEMBLY_API_KEY not set")
        
        # Get connection parameters if provided
        connection_params = kwargs.get("connection_params")
        
        # If no connection params provided, create with defaults
        if connection_params is None:
            from pipecat.services.assemblyai.stt import AssemblyAIConnectionParams
            
            connection_params = AssemblyAIConnectionParams(
                sample_rate=kwargs.get("sample_rate", 16000),
                encoding=kwargs.get("encoding", "pcm_s16le"),
                formatted_finals=kwargs.get("formatted_finals", True),
                word_finalization_max_wait_time=kwargs.get("word_finalization_max_wait_time"),
                end_of_turn_confidence_threshold=kwargs.get("end_of_turn_confidence_threshold"),
                min_end_of_turn_silence_when_confident=kwargs.get("min_end_of_turn_silence_when_confident"),
                max_turn_silence=kwargs.get("max_turn_silence")
            )
        
        # Build the service kwargs
        service_kwargs = {
            "api_key": api_key,
            "connection_params": connection_params,
            "vad_force_turn_endpoint": kwargs.get("vad_force_turn_endpoint", True),
            "language": Language.EN,  # AssemblyAI only supports English for streaming
        }
        
        # Only add api_endpoint_base_url if it's provided
        if "api_endpoint_base_url" in kwargs:
            service_kwargs["api_endpoint_base_url"] = kwargs["api_endpoint_base_url"]
        
        return AssemblyAISTTService(**service_kwargs)
    
    elif service_name == "riva":
        api_key = os.getenv("RIVA_API_KEY")
        if not api_key:
            raise ValueError("RIVA_API_KEY not set")

        # Use standard streaming service instead of segmented
        model_name = kwargs.get("model", "parakeet-ctc-1.1b-asr")
        function_id = kwargs.get("function_id", "1598d209-5e27-4d3c-8079-4751568b1081")

        stt = RivaSTTService(
            api_key=api_key,
            model_function_map={
                "function_id": function_id,
                "model_name": model_name,
            },
            params=RivaSTTService.InputParams(
                language=Language.EN_US
            )
        )
        
        # Configure for better transcription quality
        stt._automatic_punctuation = True
        stt._no_verbatim_transcripts = False
        stt._profanity_filter = False
        
        return stt
    
    elif service_name == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        
        model = kwargs.get("model", "llama-3.3-70b-versatile")
        return GroqSTTService(
            model="whisper-large-v3-turbo",
            api_key=api_key,
            language=Language.EN,
            prompt="Transcribe the following conversation",
            temperature=0.0
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
            model=model,
            

            

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

    elif service_name == "together":
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not set")
        
        model = kwargs.get("model", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
        llm = TogetherLLMService(
            api_key=api_key,
            model=model
        )
        return llm
    elif service_name == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        
        model = kwargs.get("model", "llama-3.3-70b-versatile")
        return GroqLLMService(
            api_key=api_key,
            model=model
        )
    else:
        raise ValueError(f"Unknown LLM service: {service_name}")


def create_tts_service(service_name: str, **kwargs) -> Tuple[Any, int]:
    """
    Create the appropriate TTS service.
    Returns (service, output_sample_rate)
    """
    if service_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        # OpenAI TTS only supports 24000 Hz
        sample_rate = 24000
        
        return OpenAITTSService(
            api_key=api_key,
            voice=kwargs.get("voice", "alloy"),
            model=kwargs.get("model", "gpt-4o-mini-tts"),
            sample_rate=sample_rate
        ), sample_rate

    elif service_name == "elevenlabs":
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

    elif service_name == "deepgram":
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")

        # Default sample rate for Deepgram TTS
        sample_rate = kwargs.get("sample_rate", 16000)
        
        return DeepgramTTSService(
            api_key=api_key,
            voice=kwargs.get("voice", "aura-2-helena-en"),
            base_url=kwargs.get("base_url", ""),
            sample_rate=sample_rate,
            encoding=kwargs.get("encoding", "linear16")
        ), sample_rate

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

    elif service_name == "speechify":
        api_key = os.getenv("SPEECHIFY_API_KEY")
        if not api_key:
            raise ValueError("SPEECHIFY_API_KEY not set")

        aiohttp_session = kwargs.get("aiohttp_session")
        if aiohttp_session is None:
            raise ValueError("create_tts_service('speechify') needs aiohttp_session=<ClientSession>")

        # Speechify uses 24000 Hz sample rate
        sample_rate = 24000
        
        # Default to English (en-US) since language is always English
        params = SpeechifyTTSService.InputParams(
            language="en-US",
            model=kwargs.get("model", "simba-english")
        )
        
        return SpeechifyTTSService(
            api_key=api_key,
            voice_id=kwargs.get("voice_id", "kristy"),  # You'll need actual Speechify voice IDs
            model=kwargs.get("model", "simba-english"),
            aiohttp_session=aiohttp_session,
            sample_rate=sample_rate,
            params=params,
            base_url=kwargs.get("base_url", "https://api.sws.speechify.com")
        ), sample_rate
    elif service_name == "rime":
        sample_rate = 24000
        api_key = os.getenv("RIME_API_KEY")
        if not api_key:
            raise ValueError("RIME_API_KEY not set")
        return RimeTTSService(
            api_key=api_key,
            voice_id="cove",
            model="mistv2",
    params=RimeTTSService.InputParams(
        language=Language.EN,
        speed_alpha=1.0
    )
), sample_rate
    elif service_name == "riva":
        api_key = os.getenv("RIVA_API_KEY")
        if not api_key:
            raise ValueError("RIVA_API_KEY not set")

        # Which model?  default = our #1 pick
        model_name = kwargs.get("model", "radtts-hifigan-tts")
        RIVA_TTS_MODEL_MAP = {
    "radtts-hifigan-tts":  "5e607c81-7aa6-44ce-a11d-9e08f0a3fe49",   # ← new
    "fastpitch-hifigan-tts": "0149dedb-2be8-4195-b9a0-e57e0e14f972",
}
        # Look up function-id or allow override
        function_id = kwargs.get(
            "function_id",
            RIVA_TTS_MODEL_MAP.get(model_name)
        )
        if function_id is None:
            raise ValueError(
                f"No function-id known for '{model_name}'. "
                "Pass function_id=<uuid> explicitly."
            )

        # Quality flag: 20 = best, lower = faster
        quality = kwargs.get("quality", 20)

        # All Riva voices are mono PCM - we’ll output 24 kHz to match most pipelines
        sample_rate = kwargs.get("sample_rate", 24000)

        tts = RivaTTSService(
            api_key=api_key,
            voice_id=kwargs.get("voice_id", "English-US.Female-1"),
            sample_rate=sample_rate,
            model_function_map={
                "function_id": function_id,
                "model_name": model_name,
            },
            params=RivaTTSService.InputParams(
                language=Language.EN_US,
                quality=quality,
            ),
        )
        # Optional tweaks
        # tts._settings["noise_scale"] = 0.667  # for subtle variation
        return tts, sample_rate
    elif service_name == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        sample_rate = 24000
        model = kwargs.get("model", "playai-tts")
        return GroqTTSService(
            api_key=api_key,
            model_name=model,
            voice_id="Celeste-PlayAI",
            params=GroqTTSService.InputParams(
        language=Language.EN,
        speed=1.0,
        seed=42
    )
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