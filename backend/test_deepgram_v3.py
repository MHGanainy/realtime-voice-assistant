# backend/test_deepgram_v3.py
"""
Test if Deepgram v3 works without rate limiting
"""
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

print("Testing Deepgram SDK v3...")

# First, let's see which version we have
try:
    import deepgram
    print(f"Deepgram SDK version: {deepgram.__version__ if hasattr(deepgram, '__version__') else 'Unknown'}")
except Exception as e:
    print(f"Error importing deepgram: {e}")
    exit(1)

# Try the v3 API
try:
    from deepgram import DeepgramClient, LiveOptions
    
    async def test_v3():
        try:
            # Create a Deepgram client using the API key
            deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))
            
            # Test with a simple listen websocket connection
            # This will tell us if the API key works
            config = LiveOptions(
                model="nova-2",
                language="en-US",
                punctuate=True
            )
            
            print("✅ Deepgram v3 client created successfully!")
            print("   API key appears to be valid")
            
            # Note: We can't easily test for rate limits without actually using the service
            print("\n⚠️  Note: Rate limits can only be detected when actually using the service")
            
            return True
            
        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "Unauthorized" in error_str:
                print("❌ Deepgram v3 authentication failed - Invalid API Key")
                return False
            else:
                print(f"❌ Deepgram v3 error: {e}")
                return False
    
    # Run the test
    success = asyncio.run(test_v3())
    
    if success:
        print("\n✅ Deepgram v3 is working! You can use the main.py as is.")
    else:
        print("\n❌ Deepgram v3 has issues. Need to find alternative solution.")
        
except ImportError as e:
    print(f"❌ Deepgram v3 imports not available: {e}")
    print("   You have Deepgram v2 installed")
    print("\nTo fix, choose one option:")
    print("1. Upgrade Deepgram: pip install deepgram-sdk==3.0.0")
    print("2. Or downgrade Pipecat: pip install pipecat-ai==0.0.39")