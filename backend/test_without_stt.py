# backend/test_without_stt.py
"""
Test script to verify other services work without STT
"""
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import TextFrame, LLMMessagesFrame, EndFrame
from pipecat.services.openai import OpenAILLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

class TextInjector(FrameProcessor):
    """Inject text as if it came from STT"""
    
    async def inject_text(self, text: str):
        """Simulate user speaking"""
        print(f"Injecting text: {text}")
        frame = TextFrame(text=text)
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)

async def test_pipeline():
    """Test the pipeline without STT"""
    
    # Create services
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo"
    )
    
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVEN_API_KEY"),
        voice_id="EXAVITQu4vr4xnSDxMaL"
    )
    
    # Create text injector
    text_injector = TextInjector()
    
    # Audio collector to capture TTS output
    audio_chunks = []
    
    class AudioCollector(FrameProcessor):
        async def process_frame(self, frame, direction):
            if hasattr(frame, 'audio') and frame.audio:
                audio_chunks.append(len(frame.audio))
                print(f"Collected audio chunk: {len(frame.audio)} bytes")
            await self.push_frame(frame, direction)
    
    audio_collector = AudioCollector()
    
    # Build pipeline
    pipeline = Pipeline([
        text_injector,
        llm,
        tts,
        audio_collector
    ])
    
    # Initial messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep responses very short."
        }
    ]
    
    # Create task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True)
    )
    
    # Initialize with system messages
    await task.queue_frames([LLMMessagesFrame(messages)])
    
    # Run pipeline
    runner = PipelineRunner()
    
    async def run_test():
        # Wait a bit for initialization
        await asyncio.sleep(0.5)
        
        # Inject test text
        await text_injector.inject_text("Hello, can you hear me?")
        
        # Wait for response
        await asyncio.sleep(5)
        
        print(f"\nTotal audio chunks collected: {len(audio_chunks)}")
        if audio_chunks:
            print(f"Total audio bytes: {sum(audio_chunks)}")
        
        # End the pipeline
        await task.queue_frame(EndFrame())
    
    # Run both tasks
    await asyncio.gather(
        runner.run(task),
        run_test()
    )

if __name__ == "__main__":
    print("Testing pipeline without STT...")
    print("This will simulate a user saying 'Hello, can you hear me?'")
    print("-" * 50)
    asyncio.run(test_pipeline())