from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio, json
from .providers import make_stt, make_llm, make_tts

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    stt = make_stt()
    llm = make_llm()
    tts = make_tts()

    paused = asyncio.Event()

    async def audio_source():
        try:
            while True:
                chunk = await ws.receive_bytes()
                if not paused.is_set():
                    yield chunk
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

            paused.set()

            # ---------- LLM → ElevenLabs TTS pipeline ----------
            reply_parts: list[str] = []

            async def llm_to_tts():
                """Yield tokens to TTS while remembering them for the text reply."""
                async for token in llm.stream(utterance):
                    reply_parts.append(token)
                    yield token               # hand token to TTS as soon as we get it

            # stream TTS audio to the browser
            
            async for audio_chunk in tts.stream(llm_to_tts()):
                await ws.send_bytes(audio_chunk)

            assistant_reply = "".join(reply_parts)     # full sentence for the UI


            await ws.send_json({
                "transcript": utterance,
                "final": True,
                "response": assistant_reply,   # mirror for UI demo
            })
            print("USER SAID ▶", utterance)
            print("ASSISTANT REPLY ▶", assistant_reply)

            paused.clear()
