from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional, Tuple

from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents
from deepgram.clients.listen import LiveOptions

from ..config import settings
from ..interfaces.stt_base import STT

logger = logging.getLogger(__name__)


class DeepgramSTT(STT):
    """
    Speech-to-Text provider with automatic reconnection & keep-alive.
    """

    # --------------------------------------------------------------------- #
    # object lifecycle
    # --------------------------------------------------------------------- #
    def __init__(self) -> None:
        cfg = DeepgramClientOptions(api_key=settings.deepgram_api_key, options={"keepalive": True})
        self.client = DeepgramClient("", cfg)

        # Runtime state (initialised in .stream)
        self.dg_connection = None
        self._sender_task: Optional[asyncio.Task] = None
        self._result_queue: Optional[asyncio.Queue] = None
        self._last_audio_time: Optional[float] = None
        self._connection_alive = False

        # resilience knobs
        self._max_retries = 5          # initial try + 4 retries
        self._base_delay = 1.0         # seconds – first back-off

    # --------------------------------------------------------------------- #
    # public API – called from main app
    # --------------------------------------------------------------------- #
    async def stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[Tuple[str, bool, bool]]:
        """
        Consume audio chunks and yield (transcript, is_final, speech_final).

        Re-dials Deepgram automatically if the WS is dropped.
        """
        self._result_queue = asyncio.Queue()
        self._connection_alive = True

        # Deepgram model / behaviour options (adjust as needed)
        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            interim_results=False,
            endpointing=300,  # ms VAD timeout
        )

        async def open_connection() -> None:
            """Connect (or reconnect) to Deepgram with exponential back-off."""
            attempt = 0
            delay = self._base_delay
            while attempt <= self._max_retries:
                try:
                    self.dg_connection = self.client.listen.asyncwebsocket.v("1")
                    self._setup_event_handlers()
                    if await self.dg_connection.start(options):
                        logger.info("Deepgram connected on attempt %d", attempt + 1)
                        return
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Deepgram attempt %d failed – %s", attempt + 1, exc)

                attempt += 1
                if attempt > self._max_retries:
                    raise RuntimeError("Deepgram: exhausted reconnect attempts")

                logger.info("Retrying Deepgram in %.1fs …", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)  # cap back-off at 30 s

        async def audio_sender() -> None:
            """Forward microphone chunks to Deepgram."""
            try:
                async for chunk in audio_chunks:
                    if self.dg_connection and self._connection_alive:
                        await self.dg_connection.send(chunk)
                        self._last_audio_time = asyncio.get_running_loop().time()
            except Exception as exc:  # noqa: BLE001
                logger.error("Error sending audio to Deepgram: %s", exc)
            finally:
                self._connection_alive = False
                await self._result_queue.put(None)  # poison pill

        async def keepalive_sender() -> None:
            """Ping Deepgram if no audio has flowed recently."""
            try:
                while self._connection_alive:
                    await asyncio.sleep(5)
                    if (
                        self.dg_connection
                        and self._connection_alive
                        and (
                            self._last_audio_time is None
                            or (asyncio.get_running_loop().time() - self._last_audio_time) > 5
                        )
                    ):
                        logger.debug("Sending Deepgram keep-alive")
                        await self.dg_connection.keep_alive()
            except asyncio.CancelledError:  # normal on shutdown
                pass
            except Exception as exc:  # noqa: BLE001
                logger.error("Keep-alive error: %s", exc)

        # ------------------------- outer connection loop ------------------- #
        await open_connection()

        self._sender_task = asyncio.create_task(audio_sender())
        keepalive_task = asyncio.create_task(keepalive_sender())

        try:
            while True:
                result = await self._result_queue.get()
                if result is None:  # poison pill – socket died
                    # try to reconnect
                    logger.info("Deepgram socket closed – reconnecting …")
                    self._connection_alive = False
                    await open_connection()  # may raise after retries

                    # spin up fresh sender/keep-alive tasks
                    self._sender_task = asyncio.create_task(audio_sender())
                    if keepalive_task.done():
                        keepalive_task = asyncio.create_task(keepalive_sender())
                    continue

                transcript, is_final, speech_final = result
                yield transcript, is_final, speech_final
        finally:
            await self._cleanup()
            keepalive_task.cancel()
            with asyncio.SuppressCancelled():
                await keepalive_task

    # --------------------------------------------------------------------- #
    # private helpers
    # --------------------------------------------------------------------- #
    def _setup_event_handlers(self) -> None:
        """Register Deepgram WS event callbacks."""

        async def on_transcript(dg_self, result, **_) -> None:
            if hasattr(result, "channel") and result.channel.alternatives:
                alt = result.channel.alternatives[0]
                transcript = alt.transcript
                if transcript:
                    is_final = getattr(result, "is_final", False)
                    speech_final = getattr(result, "speech_final", False)
                    logger.debug(
                        "DG result: %s | is_final=%s | speech_final=%s",
                        transcript,
                        is_final,
                        speech_final,
                    )
                    await self._result_queue.put((transcript, is_final, speech_final))

        async def on_error(dg_self, error, **_) -> None:
            logger.error("Deepgram error: %s – will reconnect", error)
            self._connection_alive = False
            await self._result_queue.put(None)  # poison pill

        async def on_close(dg_self, close, **_) -> None:
            logger.info("Deepgram closed: %s", close)
            self._connection_alive = False
            await self._result_queue.put(None)  # poison pill

        self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        self.dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        self.dg_connection.on(LiveTranscriptionEvents.Close, on_close)

    async def _cleanup(self) -> None:
        """Cancel tasks and close the Deepgram socket."""
        if self._sender_task and not self._sender_task.done():
            self._sender_task.cancel()
            with asyncio.SuppressCancelled():
                await self._sender_task

        if self.dg_connection:
            try:
                await self.dg_connection.finish()
            except Exception as exc:  # noqa: BLE001
                logger.error("Error finishing Deepgram connection: %s", exc)
