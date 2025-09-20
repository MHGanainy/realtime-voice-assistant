"""
Application configuration and settings.
"""
import os
from typing import List, Optional, Dict, Any
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
    speechify_api_key: Optional[str] = Field(default=None, env="SPEECHIFY_API_KEY")
    together_api_key: Optional[str] = Field(default=None, env="TOGETHER_API_KEY")
    rime_api_key: Optional[str] = Field(default=None, env="RIME_API_KEY")
    riva_api_key: Optional[str] = Field(default=None, env="RIVA_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    assembly_api_key: Optional[str] = Field(default=None, env="ASSEMBLY_API_KEY")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    
    # Google Cloud Credentials
    google_project_id: Optional[str] = Field(default=None, env="GOOGLE_PROJECT_ID")
    google_private_key_id: Optional[str] = Field(default=None, env="GOOGLE_PRIVATE_KEY_ID")
    google_private_key: Optional[str] = Field(default=None, env="GOOGLE_PRIVATE_KEY")
    google_client_email: Optional[str] = Field(default=None, env="GOOGLE_CLIENT_EMAIL")
    google_client_id: Optional[str] = Field(default=None, env="GOOGLE_CLIENT_ID")
    google_auth_uri: Optional[str] = Field(default="https://accounts.google.com/o/oauth2/auth", env="GOOGLE_AUTH_URI")
    google_token_uri: Optional[str] = Field(default="https://oauth2.googleapis.com/token", env="GOOGLE_TOKEN_URI")
    google_auth_provider_x509_cert_url: Optional[str] = Field(
        default="https://www.googleapis.com/oauth2/v1/certs", 
        env="GOOGLE_AUTH_PROVIDER_X509_CERT_URL"
    )
    google_client_x509_cert_url: Optional[str] = Field(default=None, env="GOOGLE_CLIENT_X509_CERT_URL")
    
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
    
    # Backend Integration for Billing
    backend_url: str = Field(
        default="http://localhost:3000",
        env="BACKEND_URL",
        description="MVP backend URL for billing webhooks"
    )
    backend_shared_secret: str = Field(
        default="your-internal-secret-change-in-production",
        env="BACKEND_SHARED_SECRET",
        description="Shared secret for internal API authentication"
    )
    
    # Billing Configuration
    billing_enabled: bool = Field(
        default=True,
        env="BILLING_ENABLED",
        description="Enable/disable billing functionality"
    )
    billing_webhook_timeout: int = Field(
        default=5,
        env="BILLING_WEBHOOK_TIMEOUT",
        description="Timeout for billing webhooks in seconds"
    )
    billing_grace_period_seconds: int = Field(
        default=60,
        env="BILLING_GRACE_PERIOD_SECONDS",
        description="Grace period before terminating conversation due to insufficient credits"
    )
    billing_retry_attempts: int = Field(
        default=3,
        env="BILLING_RETRY_ATTEMPTS",
        description="Number of retry attempts for failed billing webhooks"
    )
    billing_retry_delay_seconds: int = Field(
        default=2,
        env="BILLING_RETRY_DELAY_SECONDS",
        description="Delay between billing webhook retry attempts"
    )
    billing_minute_interval: int = Field(
        default=60,
        env="BILLING_MINUTE_INTERVAL",
        description="Interval in seconds for billing per minute (60 for production, less for testing)"
    )
    billing_warning_threshold: int = Field(
        default=2,
        env="BILLING_WARNING_THRESHOLD",
        description="Credits remaining threshold for warning messages"
    )
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    enable_billing_logs: bool = Field(
        default=True,
        env="ENABLE_BILLING_LOGS",
        description="Enable detailed billing logs"
    )
    billing_log_level: str = Field(
        default="DEBUG",
        env="BILLING_LOG_LEVEL",
        description="Log level specifically for billing operations"
    )
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_export_interval: int = Field(default=60, env="METRICS_EXPORT_INTERVAL")
    enable_billing_metrics: bool = Field(
        default=True,
        env="ENABLE_BILLING_METRICS",
        description="Enable billing-specific metrics collection"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def get_google_credentials_json(self) -> Optional[dict]:
        """Build Google credentials JSON from environment variables"""
        if not all([
            self.google_project_id,
            self.google_private_key_id,
            self.google_private_key,
            self.google_client_email,
            self.google_client_id
        ]):
            return None
        
        return {
            "type": "service_account",
            "project_id": self.google_project_id,
            "private_key_id": self.google_private_key_id,
            "private_key": self.google_private_key.replace('\\n', '\n'),  # Handle escaped newlines
            "client_email": self.google_client_email,
            "client_id": self.google_client_id,
            "auth_uri": self.google_auth_uri,
            "token_uri": self.google_token_uri,
            "auth_provider_x509_cert_url": self.google_auth_provider_x509_cert_url,
            "client_x509_cert_url": self.google_client_x509_cert_url or f"https://www.googleapis.com/robot/v1/metadata/x509/{self.google_client_email.replace('@', '%40')}",
            "universe_domain": "googleapis.com"
        }
        
    def validate_api_keys(self) -> dict:
        """Validate which API keys are configured"""
        return {
            "deepgram": bool(self.deepgram_api_key),
            "openai": bool(self.openai_api_key),
            "elevenlabs": bool(self.eleven_api_key),
            "deepinfra": bool(self.deepinfra_api_key),
            "speechify": bool(self.speechify_api_key),
            "together": bool(self.together_api_key),
            "rime": bool(self.rime_api_key),
            "riva": bool(self.riva_api_key),
            "groq": bool(self.groq_api_key),
            "assembly": bool(self.assembly_api_key),
            "google": bool(self.get_google_credentials_json()),
            "aws_access_key_id": bool(self.aws_access_key_id),
            "aws_secret_access_key": bool(self.aws_secret_access_key),
            "billing_backend": bool(self.backend_url and self.backend_shared_secret)
        }
    
    def get_service_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service"""
        key_map = {
            "deepgram": self.deepgram_api_key,
            "openai": self.openai_api_key,
            "elevenlabs": self.eleven_api_key,
            "deepinfra": self.deepinfra_api_key,
            "speechify": self.speechify_api_key,
            "together": self.together_api_key,
            "rime": self.rime_api_key,
            "riva": self.riva_api_key,
            "groq": self.groq_api_key,
            "assembly": self.assembly_api_key,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key
        }
        return key_map.get(service)
    
    def validate_billing_config(self) -> Dict[str, Any]:
        """Validate billing configuration and return status"""
        validation = {
            "enabled": self.billing_enabled,
            "backend_url_configured": bool(self.backend_url),
            "secret_configured": bool(self.backend_shared_secret),
            "webhook_timeout": self.billing_webhook_timeout,
            "grace_period": self.billing_grace_period_seconds,
            "retry_attempts": self.billing_retry_attempts,
            "retry_delay": self.billing_retry_delay_seconds,
            "minute_interval": self.billing_minute_interval,
            "warning_threshold": self.billing_warning_threshold,
            "logs_enabled": self.enable_billing_logs,
            "billing_log_level": self.billing_log_level,
            "metrics_enabled": self.enable_billing_metrics
        }
        
        # Check if configuration is valid
        validation["is_valid"] = all([
            self.billing_enabled,
            self.backend_url,
            self.backend_shared_secret,
            self.billing_webhook_timeout > 0,
            self.billing_grace_period_seconds >= 0,
            self.billing_minute_interval > 0
        ]) if self.billing_enabled else True
        
        # Add warnings if needed
        warnings = []
        if self.billing_enabled:
            if self.backend_shared_secret == "your-internal-secret-change-in-production":
                warnings.append("Using default internal secret - change in production!")
            if self.billing_minute_interval < 60:
                warnings.append(f"Billing interval is {self.billing_minute_interval}s (less than 1 minute) - testing mode?")
            if not self.backend_url.startswith("https") and "localhost" not in self.backend_url:
                warnings.append("Using non-HTTPS backend URL - ensure this is intentional")
                
        validation["warnings"] = warnings
        
        return validation
    
    def get_billing_webhook_url(self, endpoint: str) -> str:
        """Get full webhook URL for a billing endpoint"""
        base_url = self.backend_url.rstrip('/')
        endpoint = endpoint.lstrip('/')
        return f"{base_url}/api/{endpoint}"
    
    def should_log_billing(self, level: str = "INFO") -> bool:
        """Check if billing logs should be output at given level"""
        if not self.enable_billing_logs:
            return False
            
        levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        billing_level = levels.get(self.billing_log_level.upper(), 10)
        requested_level = levels.get(level.upper(), 20)
        
        return requested_level >= billing_level


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings