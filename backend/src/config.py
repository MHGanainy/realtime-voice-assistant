from typing import Optional, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path
import os
from dotenv import load_dotenv

# Get the directory where this file is located
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file explicitly
load_dotenv(BASE_DIR / ".env")


class ProviderConfig(BaseSettings):
    """Base configuration for providers"""
    enabled: bool = True
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    retry_max_delay: float = 30.0

class DeepgramConfig(ProviderConfig):
    api_key: str = Field(default_factory=lambda: os.getenv("DEEPGRAM_API_KEY", ""))
    model: str = Field(default="nova-2")
    punctuate: bool = True
    interim_results: bool = False
    endpointing: int = 300

class OpenAIConfig(ProviderConfig):
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default="gpt-3.5-turbo")
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: str = "You are a helpful assistant."

class ElevenLabsConfig(ProviderConfig):
    api_key: str = Field(default_factory=lambda: os.getenv("ELEVEN_API_KEY", ""))
    voice_id: str = "EXAVITQu4vr4xnSDxMaL"
    model_id: str = "eleven_monolingual_v1"
    stability: float = 0.5
    similarity_boost: float = 0.5
    optimize_streaming_latency: int = 1

class Settings(BaseSettings):
    # Application settings
    app_name: str = "Voice Assistant Framework"
    debug: bool = Field(default_factory=lambda: os.getenv("APP_DEBUG", "false").lower() == "true")
    
    # Provider selection
    stt_provider: str = Field(default_factory=lambda: os.getenv("STT_PROVIDER", "deepgram"))
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    tts_provider: str = Field(default_factory=lambda: os.getenv("TTS_PROVIDER", "elevenlabs"))
    
    # Provider configurations
    deepgram:    DeepgramConfig     = Field(default_factory=DeepgramConfig)
    openai:      OpenAIConfig       = Field(default_factory=OpenAIConfig)
    elevenlabs:  ElevenLabsConfig   = Field(default_factory=ElevenLabsConfig)
    
    # WebSocket settings
    ws_ping_interval: int = 10
    ws_timeout: int = 300
    
    # Pipeline settings
    enable_middleware: bool = True
    enable_event_bus: bool = True
    
    def validate_providers(self):
        """Validate that required providers have API keys"""
        errors = []
        
        if self.stt_provider == "deepgram" and not self.deepgram.api_key:
            errors.append("DEEPGRAM_API_KEY is required when using deepgram STT provider")
            
        if self.llm_provider == "openai" and not self.openai.api_key:
            errors.append("OPENAI_API_KEY is required when using openai LLM provider")
            
        if self.tts_provider == "elevenlabs" and not self.elevenlabs.api_key:
            errors.append("ELEVEN_API_KEY is required when using elevenlabs TTS provider")
            
        if errors:
            raise ValueError("\n".join(errors))
        
        return self

# Create settings instance
settings = Settings()

# Validate providers
settings.validate_providers()