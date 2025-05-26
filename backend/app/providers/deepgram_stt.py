import aiohttp, asyncio, json
from ..interfaces.stt_base import STT
from ..config import settings

DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2"
    "&punctuate=true"
    "&interim_results=false"          # keep live captions
    "&endpointing=200"              # 3-second VAD
)

class DeepgramSTT(STT):
    async def stream(self, audio_chunks):
        headers = {"Authorization": f"Token {settings.deepgram_api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(DEEPGRAM_URL,
                                          headers=headers,
                                          heartbeat=8) as dg_ws:

                async def _sender():
                    async for chunk in audio_chunks:
                        await dg_ws.send_bytes(chunk)
                    await dg_ws.send_str('{"type":"CloseStream"}')

                sender = asyncio.create_task(_sender())

                async for msg in dg_ws:
                    if msg.type is aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        alt  = data.get("channel", {}).get("alternatives", [{}])[0]
                        text = alt.get("transcript")
                        print(data)
                        if text:            # skip keep-alives
                            yield text, data.get("is_final", False), data.get("speech_final", False)
                    elif msg.type is aiohttp.WSMsgType.CLOSED:
                        break

                await sender
