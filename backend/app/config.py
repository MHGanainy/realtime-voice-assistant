from pydantic_settings import BaseSettings  # or pydantic.BaseSettings if you downgraded

class Settings(BaseSettings):
    # Deepgram (required for the test you’re doing)
    deepgram_api_key: str
    openai_api_key: str | None = None
    eleven_api_key: str | None = None

    stt_provider: str = "deepgram"
    llm_provider: str = "openai"
    tts_provider: str = "elevenlabs"

    class Config:
        env_file = ".env"

settings = Settings()