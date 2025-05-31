# backend/verify_services.py
"""
Verify all services work with correct versions
"""
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

print("Verifying Services with Correct Versions")
print("=" * 50)

# Test Deepgram SDK v2
print("\n1. Testing Deepgram v2:")
try:
    from deepgram import Deepgram
    
    async def test_deepgram():
        dg = Deepgram(os.getenv("DEEPGRAM_API_KEY"))
        
        # Test connection with projects endpoint
        try:
            projects = await dg.projects.list()
            print("   ✅ Deepgram connected successfully")
            if hasattr(projects, 'projects') and projects.projects:
                print(f"   Found {len(projects.projects)} project(s)")
        except Exception as e:
            if "429" in str(e):
                print("   ❌ Deepgram is rate limited (429 error)")
                print("   You need to wait or get a new API key")
            else:
                print(f"   ❌ Deepgram error: {e}")
    
    asyncio.run(test_deepgram())
    
except Exception as e:
    print(f"   ❌ Failed to import Deepgram: {e}")

# Test OpenAI
print("\n2. Testing OpenAI:")
try:
    from openai import OpenAI
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Test with v1.3.0 API (correct syntax)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Say 'OpenAI works'"}
        ],
        max_tokens=10
    )
    print(f"   ✅ OpenAI works: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"   ❌ OpenAI error: {e}")

# Test ElevenLabs
print("\n3. Testing ElevenLabs:")
try:
    from elevenlabs import generate, set_api_key
    
    set_api_key(os.getenv("ELEVEN_API_KEY"))
    
    # Generate a tiny audio sample
    audio = generate(
        text="Test",
        voice="EXAVITQu4vr4xnSDxMaL"
    )
    
    print(f"   ✅ ElevenLabs works: Generated {len(audio)} bytes of audio")
    
except Exception as e:
    if "quota" in str(e).lower():
        print("   ❌ ElevenLabs quota exceeded - check your account")
    else:
        print(f"   ❌ ElevenLabs error: {e}")

print("\n" + "=" * 50)
print("Next steps:")
print("1. If Deepgram shows rate limit, you need a new API key")
print("2. If all services work, run: python src/main.py")
print("3. If issues persist, check the error messages above")