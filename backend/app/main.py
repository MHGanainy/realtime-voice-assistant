from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState  # for connection-state checks

from .providers import make_stt, make_llm, make_tts

# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Bidirectional streaming:  browser ⇄ ASR ⇄ LLM ⇄ TTS."""
    await ws.accept()

    stt = make_stt()
    llm = make_llm()
    tts = make_tts()

    # ------------------------- browser → STT audio stream ------------------ #
    async def audio_source() -> asyncio.AsyncIterator[bytes]:
        """Relay whatever the browser sends."""
        try:
            while True:
                chunk = await ws.receive_bytes()
                yield chunk
        except WebSocketDisconnect:
            return

    # --------------------------- WebSocket ping task ----------------------- #
    async def ping_loop() -> None:
        """Ping the client every 10 s so intermediaries keep the WS alive."""
        try:
            while True:
                await asyncio.sleep(10)
                if ws.application_state is not WebSocketState.CONNECTED:
                    break
                # Starlette treats an empty bytes frame as a control ping
                await ws.send_bytes(b"")
        except WebSocketDisconnect:
            pass

    ping_task = asyncio.create_task(ping_loop())

    # --------------------------- main processing loop ---------------------- #
    buffer: list[str] = []  # collect finalised partials

    try:
        async for text, is_final, speech_final in stt.stream(audio_source()):
            if not is_final:
                # Optional live caption:
                # await ws.send_json({"transcript": text, "final": False})
                continue

            # --------------- we got a finalised chunk from STT -------------- #
            buffer.append(text)

            if speech_final:  # utterance really finished
                utterance = " ".join(buffer)
                buffer.clear()

                # 👋 Tell the browser to stop uploading mic data
                await ws.send_json({"command": "pause"})

                # ----------- LLM → ElevenLabs streaming pipeline ------------ #
                reply_parts: list[str] = []
                audio_chunks_sent = 0

                async def llm_to_tts() -> asyncio.AsyncIterator[str]:
                    token_count = 0
                    async for token in llm.stream(utterance):
                        reply_parts.append(token)
                        token_count += 1
                        logger.debug("LLM token %d: %s", token_count, token)
                        yield token  # hand token straight to TTS
                    logger.debug("LLM finished after %d tokens", token_count)

                try:
                    async for audio_chunk in tts.stream(llm_to_tts()):
                        await ws.send_bytes(audio_chunk)
                        audio_chunks_sent += 1
                    logger.debug("TTS sent %d audio chunks", audio_chunks_sent)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error streaming TTS: %s", exc, exc_info=True)

                assistant_reply = "".join(reply_parts)
                await ws.send_json(
                    {
                        "transcript": utterance,
                        "final": True,
                        "response": assistant_reply,
                    }
                )
                logger.info("USER SAID ▶ %s", utterance)
                logger.info("ASSISTANT REPLY ▶ %s", assistant_reply)

                # ✅ Tell the browser it can resume streaming
                await ws.send_json({"command": "resume"})
    finally:
        # make sure the ping task goes away even on disconnect/reload
        ping_task.cancel()
        with suppress(asyncio.CancelledError):
            await ping_task
