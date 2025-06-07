"""
Application configuration and settings.
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings  # Pydantic v2
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "Voice Assistant API"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # CORS
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
            "http://127.0.0.1:3000",
        ],
        env="CORS_ORIGINS"
    )
    
    # API Keys
    deepgram_api_key: Optional[str] = Field(default=None, env="DEEPGRAM_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    eleven_api_key: Optional[str] = Field(default=None, env="ELEVEN_API_KEY")
    deepinfra_api_key: Optional[str] = Field(default=None, env="DEEPINFRA_API_KEY")
    
    # Service Defaults
    default_stt_provider: str = Field(default="deepgram", env="DEFAULT_STT_PROVIDER")
    default_stt_model: str = Field(default="nova-2", env="DEFAULT_STT_MODEL")
    default_llm_provider: str = Field(default="deepinfra", env="DEFAULT_LLM_PROVIDER")
    default_llm_model: str = Field(
        default="meta-llama/Meta-Llama-3.1-70B-Instruct",
        env="DEFAULT_LLM_MODEL"
    )
    default_tts_provider: str = Field(default="deepinfra", env="DEFAULT_TTS_PROVIDER")
    default_tts_model: str = Field(default="hexgrad/Kokoro-82M", env="DEFAULT_TTS_MODEL")
    default_tts_voice: str = Field(default="af_bella", env="DEFAULT_TTS_VOICE")
    
    # Audio Configuration
    default_sample_rate: int = Field(default=16000, env="DEFAULT_SAMPLE_RATE")
    default_channels: int = Field(default=1, env="DEFAULT_CHANNELS")
    
    # Conversation Defaults
    default_system_prompt: str = Field(
        default="You are a helpful assistant. Keep your responses brief and conversational.",
        env="DEFAULT_SYSTEM_PROMPT"
    )
    max_conversation_duration_ms: int = Field(
        default=3600000,  # 1 hour
        env="MAX_CONVERSATION_DURATION_MS"
    )
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_export_interval: int = Field(default=60, env="METRICS_EXPORT_INTERVAL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def validate_api_keys(self) -> dict:
        """Validate which API keys are configured"""
        return {
            "deepgram": bool(self.deepgram_api_key),
            "openai": bool(self.openai_api_key),
            "elevenlabs": bool(self.eleven_api_key),
            "deepinfra": bool(self.deepinfra_api_key)
        }
    
    def get_service_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service"""
        key_map = {
            "deepgram": self.deepgram_api_key,
            "openai": self.openai_api_key,
            "elevenlabs": self.eleven_api_key,
            "deepinfra": self.deepinfra_api_key
        }
        return key_map.get(service)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings