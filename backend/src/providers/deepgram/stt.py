from typing import AsyncIterator, Dict, Any, Optional
import asyncio
import logging
from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents
from deepgram.clients.listen import LiveOptions
from ...core.interfaces.stt_base import STTProvider
from ...core.events.types import TranscriptEvent, EventType
from ...utils.decorators import provider

logger = logging.getLogger(__name__)

@provider("stt", "deepgram")
class DeepgramSTT(STTProvider):
    """Deepgram STT provider implementation with retry logic"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.dg_connection = None
        self._connection_alive = False
        self._result_queue: Optional[asyncio.Queue] = None
        self._sender_task: Optional[asyncio.Task] = None
        self._last_audio_time: Optional[float] = None
        
        # Retry configuration
        self._max_retries = 5
        self._base_delay = 1.0
        
    async def initialize(self) -> None:
        """Initialize Deepgram client"""
        cfg = DeepgramClientOptions(
            api_key=self.config["api_key"],
            options={"keepalive": True}
        )
        self.client = DeepgramClient("", cfg)
        logger.info("Deepgram client initialized")
        
    async def cleanup(self) -> None:
        """Cleanup Deepgram connection"""
        await self._cleanup()
        
    @property
    def is_connected(self) -> bool:
        return self._connection_alive
        
    async def reconnect(self) -> None:
        """Reconnect to Deepgram"""
        await self._cleanup()
        await self._open_connection()
        
    async def stream(
        self, 
        audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptEvent]:
        """Stream audio and yield transcript events"""
        self._result_queue = asyncio.Queue()
        self._connection_alive = True
        
        # Deepgram options - matching the working code
        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            interim_results=False,  # Enable to get partial results
            endpointing=300,  # ms VAD timeout
        )
        
        # Store options for reconnection
        self._options = options
        
        # Connect to Deepgram
        await self._open_connection()
        
        # Start tasks
        self._sender_task = asyncio.create_task(self._audio_sender(audio_chunks))
        keepalive_task = asyncio.create_task(self._keepalive_sender())
        
        try:
            while True:
                result = await self._result_queue.get()
                
                if result is None:  # Connection closed
                    logger.info("Deepgram socket closed – reconnecting...")
                    self._connection_alive = False
                    
                    # Cancel old sender task
                    if self._sender_task and not self._sender_task.done():
                        self._sender_task.cancel()
                        try:
                            await self._sender_task
                        except asyncio.CancelledError:
                            pass
                    
                    # Reconnect
                    await self._open_connection()
                    
                    # Restart tasks
                    self._sender_task = asyncio.create_task(self._audio_sender(audio_chunks))
                    if keepalive_task.done():
                        keepalive_task = asyncio.create_task(self._keepalive_sender())
                    continue
                
                # Convert tuple result to TranscriptEvent
                transcript, is_final, speech_final = result
                event = TranscriptEvent(
                    source="deepgram_stt",
                    transcript=transcript,
                    is_final=is_final,
                    speech_final=speech_final,
                    data={"transcript": transcript}
                )
                yield event
                
        finally:
            await self._cleanup()
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
            
    async def _open_connection(self) -> None:
        """Connect (or reconnect) to Deepgram with exponential back-off."""
        attempt = 0
        delay = self._base_delay
        
        while attempt <= self._max_retries:
            try:
                self.dg_connection = self.client.listen.asyncwebsocket.v("1")
                self._setup_event_handlers()
                
                if await self.dg_connection.start(self._options):
                    logger.info(f"Deepgram connected on attempt {attempt + 1}")
                    self._connection_alive = True
                    return
                    
            except Exception as exc:
                logger.warning(f"Deepgram attempt {attempt + 1} failed – {exc}")
            
            attempt += 1
            if attempt > self._max_retries:
                raise RuntimeError("Deepgram: exhausted reconnect attempts")
            
            logger.info(f"Retrying Deepgram in {delay:.1f}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)  # cap back-off at 30s
            
    def _setup_event_handlers(self) -> None:
        """Setup Deepgram event handlers - matching working code exactly"""
        
        async def on_transcript(dg_self, result, **_):
            if hasattr(result, "channel") and result.channel.alternatives:
                alt = result.channel.alternatives[0]
                transcript = alt.transcript
                if transcript:
                    is_final = getattr(result, "is_final", False)
                    speech_final = getattr(result, "speech_final", False)
                    logger.debug(
                        f"DG result: {transcript} | is_final={is_final} | speech_final={speech_final}"
                    )
                    await self._result_queue.put((transcript, is_final, speech_final))
                    
        async def on_error(dg_self, error, **_):
            logger.error(f"Deepgram error: {error} – will reconnect")
            self._connection_alive = False
            await self._result_queue.put(None)  # poison pill
            
        async def on_close(dg_self, close, **_):
            logger.info(f"Deepgram closed: {close}")
            self._connection_alive = False
            await self._result_queue.put(None)  # poison pill
        
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        self.dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        self.dg_connection.on(LiveTranscriptionEvents.Close, on_close)
        
    async def _audio_sender(self, audio_chunks: AsyncIterator[bytes]) -> None:
        """Forward microphone chunks to Deepgram."""
        try:
            async for chunk in audio_chunks:
                if self.dg_connection and self._connection_alive:
                    await self.dg_connection.send(chunk)
                    self._last_audio_time = asyncio.get_running_loop().time()
                    
        except Exception as exc:
            logger.error(f"Error sending audio to Deepgram: {exc}")
        finally:
            self._connection_alive = False
            await self._result_queue.put(None)  # poison pill
            
    async def _keepalive_sender(self) -> None:
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
        except Exception as exc:
            logger.error(f"Keep-alive error: {exc}")
            
    async def _cleanup(self) -> None:
        """Cancel tasks and close the Deepgram socket."""
        if self._sender_task and not self._sender_task.done():
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
                
        if self.dg_connection:
            try:
                await self.dg_connection.finish()
            except Exception as exc:
                logger.error(f"Error finishing Deepgram connection: {exc}")