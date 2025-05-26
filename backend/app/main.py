from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio, json
from .providers import make_stt

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    stt = make_stt()

    async def audio_source():
        try:
            while True:
                yield await ws.receive_bytes()
        except WebSocketDisconnect:
            return

    buffer: list[str] = []            # holds is_final pieces

    async for text, is_final, speech_final in stt.stream(audio_source()):
        if not is_final:
            # optional live caption (latest interim)
            # await ws.send_json({"transcript": text, "final": False})
            continue

        # ---- this chunk is a *finalised* correction ----
        buffer.append(text)

        if speech_final:               # utterance truly finished
            utterance = " ".join(buffer)
            buffer.clear()

            await ws.send_json({
                "transcript": utterance,
                "final": True,
                "response": utterance,   # mirror for UI demo
            })
            print("USER SAID ▶", utterance)
