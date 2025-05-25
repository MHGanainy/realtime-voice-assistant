from ..config import settings
from .deepgram_stt import DeepgramSTT
from .openai_llm import OpenAIChat
from .eleven_tts import ElevenTTS
from typing import Dict

_provider_map: Dict[str, object] = {
    "deepgram": DeepgramSTT,
    "openai": OpenAIChat,
    "elevenlabs": ElevenTTS,
}

def make_stt():
    return _provider_map[settings.stt_provider]()

def make_llm():
    return _provider_map[settings.llm_provider]()

def make_tts():
    return _provider_map[settings.tts_provider]()