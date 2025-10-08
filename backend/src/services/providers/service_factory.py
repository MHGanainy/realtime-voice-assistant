"""
Factory functions for creating STT, LLM, and TTS services.
"""
from typing import Optional, Tuple, Any
import os
import logging
from pipecat.services.google.tts import GoogleTTSService, GoogleHttpTTSService
import json
import tempfile
from src.config.settings import get_settings

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.together.llm import TogetherLLMService
from pipecat.transcriptions.language import Language
from pipecat.services.inworld.tts import InworldTTSService

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
        # Check both possible environment variable names
        api_key = os.getenv("ASSEMBLYAI_API_KEY") or os.getenv("ASSEMBLY_API_KEY")
        if not api_key:
            raise ValueError("ASSEMBLYAI_API_KEY or ASSEMBLY_API_KEY not set")
        
        # Get connection parameters if provided
        connection_params = kwargs.get("connection_params")
        
        # If no connection params provided, create with all enhanced defaults
        if connection_params is None:
            from pipecat.services.assemblyai.stt import AssemblyAIConnectionParams
            
            # Build params dict with all possible settings
            params_dict = {
                "sample_rate": kwargs.get("sample_rate", 16000),
                "encoding": kwargs.get("encoding", "pcm_s16le"),
            }
            
            # Add optional parameters only if they're provided
            if kwargs.get("model"):
                params_dict["model"] = kwargs.get("model")
            if kwargs.get("format_turns") is not None:
                params_dict["formatted_finals"] = kwargs.get("format_turns")
            if kwargs.get("formatted_finals") is not None:
                params_dict["formatted_finals"] = kwargs.get("formatted_finals")
            if kwargs.get("enable_partial_transcripts") is not None:
                params_dict["enable_partial_transcripts"] = kwargs.get("enable_partial_transcripts")
            if kwargs.get("use_immutable_finals") is not None:
                params_dict["use_immutable_finals"] = kwargs.get("use_immutable_finals")
            if kwargs.get("punctuate") is not None:
                params_dict["punctuate"] = kwargs.get("punctuate")
            if kwargs.get("format_text") is not None:
                params_dict["format_text"] = kwargs.get("format_text")
            if kwargs.get("word_finalization_max_wait_time"):
                params_dict["word_finalization_max_wait_time"] = kwargs.get("word_finalization_max_wait_time")
            if kwargs.get("end_of_turn_confidence_threshold"):
                params_dict["end_of_turn_confidence_threshold"] = kwargs.get("end_of_turn_confidence_threshold")
            if kwargs.get("min_end_of_turn_silence_when_confident"):
                params_dict["min_end_of_turn_silence_when_confident"] = kwargs.get("min_end_of_turn_silence_when_confident")
            if kwargs.get("max_turn_silence"):
                params_dict["max_turn_silence"] = kwargs.get("max_turn_silence")
            
            connection_params = AssemblyAIConnectionParams(**params_dict)
        
        # Build the service kwargs
        service_kwargs = {
            "api_key": api_key,
            "connection_params": connection_params,
            "vad_force_turn_endpoint": kwargs.get("vad_force_turn_endpoint", True),
            "language": kwargs.get("language", Language.EN),  # AssemblyAI only supports English for streaming
        }
        
        # Only add api_endpoint_base_url if it's provided
        if "api_endpoint_base_url" in kwargs:
            service_kwargs["api_endpoint_base_url"] = kwargs["api_endpoint_base_url"]
        
        return AssemblyAISTTService(**service_kwargs)
    
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
            "max_tokens": 4096 #kwargs.get("max_tokens", 4096),
            "temperature": 0.3 #kwargs.get("temperature", 0.7),
            "top_p": 0.8 #kwargs.get("top_p", 1.0),
            "frequency_penalty":0.15 # kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": 0.30 #kwargs.get("presence_penalty", 0.0),
            "stop": None, #kwargs.get("stop", None),
            "stream": True, #kwargs.get("stream", True),
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
        
        # Create InputParams with all settings for enhanced configuration
        params_dict = {
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.6),
            "top_p": kwargs.get("top_p", 0.8),
            "presence_penalty": kwargs.get("presence_penalty", 0.15),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.30),
            "stream": kwargs.get("stream", True),
        }
        
        # Only add optional parameters if they exist in the InputParams model
        if hasattr(GroqLLMService, 'InputParams'):
            params = GroqLLMService.InputParams(**params_dict)
            return GroqLLMService(
                api_key=api_key,
                model=model,
                params=params
            )
        else:
            # Fallback if InputParams doesn't exist or has different structure
            return GroqLLMService(
                api_key=api_key,
                model=model,
                **params_dict  # Pass as kwargs directly
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
    
    elif service_name == "inworld":
        api_key = os.getenv("INWORLD_API_KEY")
        if not api_key:
            raise ValueError("INWORLD_API_KEY not set")
        
        aiohttp_session = kwargs.get("aiohttp_session")
        if aiohttp_session is None:
            raise ValueError("create_tts_service('inworld') needs aiohttp_session=<ClientSession>")
        
        sample_rate = kwargs.get("sample_rate", 24000)
        
        # Voice configuration with default speeds
        voice_id = kwargs.get("voice_id", "Edward")
        
        # Default speed mapping for each voice
        default_speeds = {
            "Craig": 1.2,
            "Edward": 1.0,
            "Olivia": 1.0,
            "Wendy": 1.2,
            "Priya": 1.0
        }
        
        # Use provided speed or default for the voice
        speed = kwargs.get("speed")
        if speed is None:
            speed = default_speeds.get(voice_id, 1.0)
        
        # Create InputParams with temperature and speed
        params = InworldTTSService.InputParams(
            temperature=kwargs.get("temperature", 1.1),
            speed=speed
        )
        
        logger.info(f"Creating Inworld TTS with voice={voice_id}, speed={speed}")
        
        return InworldTTSService(
            api_key=api_key,
            voice_id=voice_id,
            model=kwargs.get("model", "inworld-tts-1"),
            aiohttp_session=aiohttp_session,
            sample_rate=sample_rate,
            streaming=kwargs.get("streaming", True),
            params=params,
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
            voice_id=kwargs.get("voice_id", "kristy"),
            model=kwargs.get("model", "simba-english"),
            aiohttp_session=aiohttp_session,
            sample_rate=sample_rate,
            params=params,
            base_url=kwargs.get("base_url", "https://api.sws.speechify.com")
        ), sample_rate
        
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
        
    elif service_name == "google":    
        # Get settings
        settings = get_settings()
        
        # Try to get credentials from environment variables first
        google_creds = settings.get_google_credentials_json()
        
        if google_creds:
            # Write credentials to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(google_creds, f)
                credentials_path = f.name
        else:
            # Fall back to GOOGLE_APPLICATION_CREDENTIALS if set
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not credentials_path:
                raise ValueError("Google credentials not found in environment variables or GOOGLE_APPLICATION_CREDENTIALS")
        
        # Get voice_id and determine which service to use
        voice_id = kwargs.get("voice_id", "en-US-Chirp3-HD-Charon")
        
        # Check if this is a Chirp HD or Journey voice (streaming only)
        is_chirp_hd = "chirp" in voice_id.lower() and ("hd" in voice_id.lower() or "chirp3" in voice_id.lower())
        is_journey = "journey" in voice_id.lower()
        
        # Determine if we should use streaming based on voice type
        use_streaming = kwargs.get("use_streaming", is_chirp_hd or is_journey)
        
        # Set sample rate based on voice type
        if is_chirp_hd or is_journey:
            sample_rate = kwargs.get("sample_rate", 24000)  # Chirp HD works best at 24kHz
        else:
            sample_rate = kwargs.get("sample_rate", 16000)  # Standard voices at 16kHz
        
        # Create params if needed
        params = None
        if any(k in kwargs for k in ["language", "gender", "pitch", "rate", "volume", "emphasis", "google_style"]):
            # Streaming service only supports language
            if use_streaming:
                params = GoogleTTSService.InputParams(
                    language=kwargs.get("language", Language.EN)
                )
            else:
                # Non-streaming service supports all parameters
                params = GoogleHttpTTSService.InputParams(
                    language=kwargs.get("language", Language.EN),
                    gender=kwargs.get("gender"),
                    pitch=kwargs.get("pitch"),
                    rate=kwargs.get("rate"),
                    volume=kwargs.get("volume"),
                    emphasis=kwargs.get("emphasis"),
                    google_style=kwargs.get("google_style")
                )
        
        # Create the appropriate service
        if use_streaming:
            # Streaming service for Chirp HD voices
            service = GoogleTTSService(
                credentials_path=credentials_path,
                voice_id=voice_id,
                sample_rate=sample_rate,
                params=params
            )
        else:
            # Non-streaming service for all other voices
            service = GoogleHttpTTSService(
                credentials_path=credentials_path,
                voice_id=voice_id,
                sample_rate=sample_rate,
                params=params
            )
        
        # Clean up temp file if we created one
        if google_creds and credentials_path:
            import atexit
            atexit.register(lambda: os.unlink(credentials_path))
        
        return service, sample_rate
          
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