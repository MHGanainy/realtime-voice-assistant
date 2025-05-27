from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
import logging
from .providers import make_stt, make_llm, make_tts

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
            audio_chunks_sent = 0

            async def llm_to_tts():
                """Yield tokens to TTS while remembering them for the text reply."""
                token_count = 0
                async for token in llm.stream(utterance):
                    reply_parts.append(token)
                    token_count += 1
                    print(f"LLM token {token_count}: '{token}'")
                    yield token               # hand token to TTS as soon as we get it
                print(f"LLM finished streaming {token_count} tokens")

            # stream TTS audio to the browser
            try:
                async for audio_chunk in tts.stream(llm_to_tts()):
                    await ws.send_bytes(audio_chunk)
                    audio_chunks_sent += 1
                    
                print(f"TTS sent {audio_chunks_sent} audio chunks")
            except Exception as e:
                logger.error(f"Error streaming TTS: {e}", exc_info=True)

            assistant_reply = "".join(reply_parts)     # full sentence for the UI

            await ws.send_json({
                "transcript": utterance,
                "final": True,
                "response": assistant_reply,   # mirror for UI demo
            })
            print("USER SAID ▶", utterance)
            print("ASSISTANT REPLY ▶", assistant_reply)

            paused.clear()