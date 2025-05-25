import aiohttp, asyncio, json
from ..interfaces.tts_base import TTS
from ..config import settings

STREAM_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # default voice

class ElevenTTS(TTS):
    async def stream(self, text_iter):
        full_text = ""
        async for piece in text_iter:
            full_text += piece
            # naive chunking: synth every sentence end
            if piece.endswith(('.','?','!')):
                async for audio in self._synth(full_text):
                    yield audio
                full_text = ""
        if full_text:
            async for audio in self._synth(full_text):
                yield audio

    async def _synth(self, text):
        headers = {
            "xi-api-key": settings.eleven_api_key,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json"
        }
        payload = {"text": text, "optimize_streaming_latency":1}
        async with aiohttp.ClientSession() as session:
            async with session.post(STREAM_URL.format(voice_id=VOICE_ID), headers=headers, json=payload) as resp:
                async for chunk in resp.content.iter_chunked(2048):
                    yield chunk