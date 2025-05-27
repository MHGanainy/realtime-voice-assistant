import asyncio
import json
import logging
from typing import AsyncIterator, Optional, Tuple

from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents
from deepgram.clients.listen import LiveOptions

from ..interfaces.stt_base import STT
from ..config import settings

# Logger setup
logger = logging.getLogger(__name__)


class DeepgramSTT(STT):
    """
    Speech-to-Text provider using Deepgram SDK for WebSocket streaming.
    """

    def __init__(self):
        """Initialize the Deepgram client with API key from settings."""
        # Initialize Deepgram client
        config = DeepgramClientOptions(
            api_key=settings.deepgram_api_key,
            options={"keepalive": True}
        )
        self.client = DeepgramClient("", config)
        self.dg_connection = None
        self._audio_queue = None
        self._result_queue = None
        self._sender_task = None
        self._last_audio_time = None
        self._connection_alive = False
        
    async def stream(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[Tuple[str, bool, bool]]:
        """
        Stream audio chunks to Deepgram and yield transcription results.
        
        Args:
            audio_chunks: Async iterator of audio bytes
            
        Yields:
            Tuple of (transcript, is_final, speech_final)
        """
        # Create queues for communication
        self._audio_queue = asyncio.Queue()
        self._result_queue = asyncio.Queue()
        self._connection_alive = True
        
        # Initialize WebSocket connection
        self.dg_connection = self.client.listen.asyncwebsocket.v("1")
        
        # Set up event handlers
        self._setup_event_handlers()
        
        # Configure Deepgram options
        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            interim_results=False,  # Set to False as in your original code
            endpointing=200,  # 200ms VAD timeout (similar to your endpointing=200)
            # encoding="linear16",
            # sample_rate=16000,
            # channels=1
        )
        
        try:
            # Start the connection
            if await self.dg_connection.start(options):
                logger.info("Deepgram WebSocket connected successfully")
                
                # Start sender task
                self._sender_task = asyncio.create_task(self._audio_sender(audio_chunks))
                
                # Start keepalive task
                keepalive_task = asyncio.create_task(self._keepalive_sender())
                
                # Yield results as they come
                while self._connection_alive:
                    try:
                        result = await asyncio.wait_for(
                            self._result_queue.get(), 
                            timeout=5.0  # Reduced timeout, we'll use keepalive instead
                        )
                        
                        if result is None:  # Poison pill
                            break
                            
                        transcript, is_final, speech_final = result
                        yield transcript, is_final, speech_final
                        
                    except asyncio.TimeoutError:
                        # This is now normal during long TTS playback
                        # The keepalive will maintain the connection
                        continue
            else:
                logger.error("Failed to connect to Deepgram")
                raise Exception("Failed to connect to Deepgram WebSocket")
                
        except Exception as e:
            logger.error(f"Error in Deepgram stream: {e}")
            raise
        finally:
            self._connection_alive = False
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
            await self._cleanup()
    
    def _setup_event_handlers(self):
        """Set up event handlers for the Deepgram WebSocket connection."""
        
        async def on_message(dg_self, result, **kwargs):
            """Handle transcription results."""
            try:
                # Extract the transcript from the result
                if hasattr(result, 'channel') and result.channel.alternatives:
                    alternative = result.channel.alternatives[0]
                    transcript = alternative.transcript
                    
                    # Get the is_final flag (this indicates if the transcript is finalized)
                    is_final = getattr(result, 'is_final', False)
                    
                    # Get speech_final flag (indicates end of utterance)
                    speech_final = getattr(result, 'speech_final', False)
                    
                    # Debug log
                    logger.debug(f"Received: '{transcript}' is_final={is_final} speech_final={speech_final}")
                    
                    # Only process non-empty transcripts
                    if transcript:
                        # Put result in queue
                        await self._result_queue.put((transcript, is_final, speech_final))
                        
            except Exception as e:
                logger.error(f"Error processing Deepgram message: {e}")
        
        async def on_error(dg_self, error, **kwargs):
            """Handle errors from Deepgram."""
            logger.error(f"Deepgram error: {error}")
            self._connection_alive = False
            # Put None to signal end of stream
            await self._result_queue.put(None)
        
        async def on_close(dg_self, close, **kwargs):
            """Handle connection close."""
            logger.info("Deepgram connection closed")
            self._connection_alive = False
            # Put None to signal end of stream
            await self._result_queue.put(None)
        
        # Register event handlers
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        self.dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        self.dg_connection.on(LiveTranscriptionEvents.Close, on_close)
    
    async def _audio_sender(self, audio_chunks: AsyncIterator[bytes]):
        """Send audio chunks to Deepgram WebSocket."""
        try:
            async for chunk in audio_chunks:
                if self.dg_connection and self._connection_alive:
                    await self.dg_connection.send(chunk)
                    self._last_audio_time = asyncio.get_event_loop().time()
            
            # Send close stream message
            if self.dg_connection and self._connection_alive:
                await self.dg_connection.finish()
                
        except Exception as e:
            logger.error(f"Error sending audio to Deepgram: {e}")
        finally:
            self._connection_alive = False
            # Signal end of results
            await self._result_queue.put(None)
    
    async def _keepalive_sender(self):
        """Send keepalive messages to maintain connection during silence."""
        try:
            while self._connection_alive:
                await asyncio.sleep(5.0)  # Send keepalive every 5 seconds
                
                if self.dg_connection and self._connection_alive:
                    # Check if we haven't sent audio recently
                    current_time = asyncio.get_event_loop().time()
                    if self._last_audio_time is None or (current_time - self._last_audio_time) > 5.0:
                        # The SDK's keepalive method maintains the connection
                        logger.debug("Sending keepalive to maintain connection")
                        await self.dg_connection.keep_alive()
                        
        except Exception as e:
            logger.error(f"Error in keepalive sender: {e}")
    
    async def _cleanup(self):
        """Clean up resources."""
        try:
            # Cancel sender task if running
            if self._sender_task and not self._sender_task.done():
                self._sender_task.cancel()
                try:
                    await self._sender_task
                except asyncio.CancelledError:
                    pass
            
            # Finish the connection
            if self.dg_connection:
                try:
                    await self.dg_connection.finish()
                except Exception as e:
                    logger.error(f"Error finishing connection: {e}")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")