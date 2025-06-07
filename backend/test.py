# quick_deepinfra_test.py
import asyncio
import aiohttp
import os
from dotenv import load_dotenv
import json

load_dotenv()

async def test_deepinfra_api():
    """Quick test of DeepInfra TTS API"""
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        print("❌ DEEPINFRA_API_KEY not found in .env")
        return
    
    print("✅ API key found")
    
    # Test the API directly
    url = "https://api.deepinfra.com/v1/text-to-speech/af_bella"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "text": "Hello, testing DeepInfra API.",
        "model_id": "hexgrad/Kokoro-82M",
        "output_format": "wav"
    }
    
    print(f"\n📡 Calling API: {url}")
    print(f"📝 Payload: {json.dumps(payload, indent=2)}")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                print(f"\n📊 Response status: {response.status}")
                print(f"📋 Response headers: {dict(response.headers)}")
                
                if response.status == 200:
                    audio_data = await response.read()
                    print(f"✅ Success! Received {len(audio_data)} bytes of audio")
                    
                    # Save to file
                    with open("test_api_direct.wav", "wb") as f:
                        f.write(audio_data)
                    print("💾 Saved to test_api_direct.wav")
                else:
                    error_text = await response.text()
                    print(f"❌ Error: {error_text}")
                    
        except Exception as e:
            print(f"❌ Exception: {e}")


async def test_with_elevenlabs_client():
    """Test using ElevenLabs client"""
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        print("❌ DEEPINFRA_API_KEY not found")
        return
    
    try:
        from elevenlabs import ElevenLabs
        
        print("\n🔧 Testing with ElevenLabs client...")
        
        client = ElevenLabs(
            api_key=api_key,
            base_url="https://api.deepinfra.com/",
        )
        
        # Test basic conversion
        audio = client.text_to_speech.convert(
            voice_id="af_bella",
            output_format="mp3",
            text="Testing with ElevenLabs client.",
            model_id="hexgrad/Kokoro-82M",
        )
        
        with open("test_elevenlabs_client.mp3", "wb") as f:
            f.write(audio)
        
        print("✅ ElevenLabs client test successful!")
        
    except ImportError:
        print("⚠️  ElevenLabs client not installed. Run: pip install elevenlabs")
    except Exception as e:
        print(f"❌ ElevenLabs client error: {e}")


if __name__ == "__main__":
    print("🚀 DeepInfra TTS API Quick Test\n")
    asyncio.run(test_deepinfra_api())
    asyncio.run(test_with_elevenlabs_client())