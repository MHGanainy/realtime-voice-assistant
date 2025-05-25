# backend/app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio, json

from .providers import make_stt          # only STT for now

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    1. Front-end opens the websocket and starts sending 250 ms Opus chunks
    2. We relay those chunks to Deepgram (via DeepgramSTT)
    3. Whenever Deepgram gives us a partial/final transcript we forward it
       • always under "transcript"
       • IF `is_final` → we also echo it back as "response" so the assistant
         column shows exactly the same text.
    """
    await ws.accept()

    stt = make_stt()                     # returns DeepgramSTT instance

    async def audio_source():
        try:
            while True:
                chunk = await ws.receive_bytes()    # mic bytes
                yield chunk
        except WebSocketDisconnect:
            return                                  # client hung up

    # Stream audio → Deepgram → text back to client
    async for transcript, is_final in stt.stream(audio_source()):
        message: dict[str, object] = {
            "transcript": transcript,
            "final": is_final,          # Deepgram flag: this segment is finished
        }
        if is_final:
            # mirror user text as "assistant reply"
            message["response"] = transcript

        await ws.send_json(message)
