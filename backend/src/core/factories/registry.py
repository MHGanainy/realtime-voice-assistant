from typing import Dict, Type, Any, List  # Added List import
from ..interfaces.stt_base import STTProvider
from ..interfaces.llm_base import LLMProvider
from ..interfaces.tts_base import TTSProvider
import logging

logger = logging.getLogger(__name__)

class ProviderRegistry:
    """
    Registry for provider implementations using Strategy pattern.
    Providers self-register using the @provider decorator.
    """
    
    def __init__(self):
        self._stt_providers: Dict[str, Type[STTProvider]] = {}
        self._llm_providers: Dict[str, Type[LLMProvider]] = {}
        self._tts_providers: Dict[str, Type[TTSProvider]] = {}
        
    def register_stt(self, name: str, provider_class: Type[STTProvider]) -> None:
        """Register an STT provider"""
        self._stt_providers[name] = provider_class
        logger.info(f"Registered STT provider: {name}")
        
    def register_llm(self, name: str, provider_class: Type[LLMProvider]) -> None:
        """Register an LLM provider"""
        self._llm_providers[name] = provider_class
        logger.info(f"Registered LLM provider: {name}")
        
    def register_tts(self, name: str, provider_class: Type[TTSProvider]) -> None:
        """Register a TTS provider"""
        self._tts_providers[name] = provider_class
        logger.info(f"Registered TTS provider: {name}")
        
    def create_stt(self, name: str, config: Dict[str, Any]) -> STTProvider:
        """Create STT provider instance (Factory pattern)"""
        if name not in self._stt_providers:
            raise ValueError(f"STT provider '{name}' not registered")
        return self._stt_providers[name](config)
        
    def create_llm(self, name: str, config: Dict[str, Any]) -> LLMProvider:
        """Create LLM provider instance (Factory pattern)"""
        if name not in self._llm_providers:
            raise ValueError(f"LLM provider '{name}' not registered")
        return self._llm_providers[name](config)
        
    def create_tts(self, name: str, config: Dict[str, Any]) -> TTSProvider:
        """Create TTS provider instance (Factory pattern)"""
        if name not in self._tts_providers:
            raise ValueError(f"TTS provider '{name}' not registered")
        return self._tts_providers[name](config)
    
    def list_providers(self) -> Dict[str, List[str]]:
        """List all registered providers"""
        return {
            "stt": list(self._stt_providers.keys()),
            "llm": list(self._llm_providers.keys()),
            "tts": list(self._tts_providers.keys())
        }

# Global registry instance
registry = ProviderRegistry()

# Factory functions using the registry
def create_stt(name: str = None, config: Dict[str, Any] = None) -> STTProvider:
    """Factory function to create STT provider"""
    from ...config import settings
    name = name or settings.stt_provider
    if config is None:
        config = getattr(settings, name, {})
        if hasattr(config, "dict"):
            config = config.dict()
    return registry.create_stt(name, config)

def create_llm(name: str = None, config: Dict[str, Any] = None) -> LLMProvider:
    """Factory function to create LLM provider"""
    from ...config import settings
    name = name or settings.llm_provider
    if config is None:
        config = getattr(settings, name, {})
        if hasattr(config, "dict"):
            config = config.dict()
    return registry.create_llm(name, config)

def create_tts(name: str = None, config: Dict[str, Any] = None) -> TTSProvider:
    """Factory function to create TTS provider"""
    from ...config import settings
    name = name or settings.tts_provider
    if config is None:
        config = getattr(settings, name, {})
        if hasattr(config, "dict"):
            config = config.dict()
    return registry.create_tts(name, config)