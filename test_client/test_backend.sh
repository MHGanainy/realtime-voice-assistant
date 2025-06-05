#!/bin/bash

echo "🎤 Voice Assistant Full Pipeline Test"
echo "====================================="
echo ""
echo "Prerequisites:"
echo "  1. Backend must be running with API keys configured"
echo "  2. You need microphone and speaker access"
echo ""
echo "What will happen:"
echo "  1. Connect to the backend and get a session"
echo "  2. Stream your microphone audio"
echo "  3. Show transcriptions as you speak"
echo "  4. Play assistant responses through speakers"
echo ""
echo "Press Ctrl+C to exit if needed"
echo ""

# Check if backend is running
if ! curl -s http://localhost:8000/api/health > /dev/null; then
    echo "❌ Backend is not running! Start it with:"
    echo "   cd backend && python -m uvicorn src.main:app --reload"
    exit 1
fi

echo "✅ Backend is running!"
echo ""

# Install requirements if needed
pip install -r requirements.txt > /dev/null 2>&1

# Run the voice assistant client
# python audio_recorder.py

python interactive_voice_assistant.py