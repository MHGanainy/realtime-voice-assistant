import aiohttp, asyncio, json
from ..interfaces.stt_base import STT
from ..config import settings

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen?punctuate=true&interim_results=true"

class DeepgramSTT(STT):
    async def stream(self, audio_chunks):
        headers = {"Authorization": f"Token {settings.deepgram_api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(DEEPGRAM_URL, headers=headers) as ws:
                async def _sender():
                    async for chunk in audio_chunks:
                        await ws.send_bytes(chunk)
                    await ws.send_str(json.dumps({"type": "CloseStream"}))
                sender = asyncio.create_task(_sender())
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if transcript := data.get("channel", {}).get("alternatives", [{}])[0].get("transcript"):
                            yield transcript, data.get("speech_final", False)
                    if msg.type == aiohttp.WSMsgType.CLOSED:
                        break
                await sender