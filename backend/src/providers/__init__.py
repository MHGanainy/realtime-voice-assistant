# Import providers to trigger auto-registration via @provider decorator
from .deepgram.stt import DeepgramSTT
from .openai.llm import OpenAIChat
from .elevenlabs.tts import ElevenLabsTTS

# Export factory functions for backward compatibility
from ..core.factories.registry import create_stt, create_llm, create_tts

__all__ = [
    'create_stt',
    'create_llm', 
    'create_tts',
    'DeepgramSTT',
    'OpenAIChat',
    'ElevenLabsTTS'
]