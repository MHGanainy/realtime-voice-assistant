# backend/find_modules.py
"""
Find the correct import paths for Pipecat modules
"""
import os
import pkgutil
import pipecat

print("Pipecat Module Structure")
print("=" * 50)

# Find Pipecat installation path
pipecat_path = os.path.dirname(pipecat.__file__)
print(f"Pipecat installed at: {pipecat_path}")

# List all submodules
print("\nAvailable Pipecat modules:")
for importer, modname, ispkg in pkgutil.walk_packages(pipecat.__path__, prefix='pipecat.'):
    if 'vad' in modname or 'silero' in modname or 'audio' in modname:
        print(f"  {'[PKG]' if ispkg else '[MOD]'} {modname}")

# Try to find VAD modules
print("\n" + "=" * 50)
print("Looking for VAD/Silero modules...")

possible_paths = [
    "pipecat.vad.silero",
    "pipecat.audio.vad.silero",
    "pipecat.processors.vad.silero",
    "pipecat.services.vad.silero",
    "pipecat.vad",
]

for path in possible_paths:
    try:
        module = __import__(path, fromlist=[''])
        print(f"✅ Found: {path}")
        # List what's in the module
        items = [x for x in dir(module) if not x.startswith('_')]
        if 'SileroVADAnalyzer' in items:
            print(f"   Contains SileroVADAnalyzer!")
        print(f"   Available: {items[:5]}...")
    except ImportError:
        print(f"❌ Not found: {path}")

# Check what's in the transports module for VAD
print("\n" + "=" * 50)
print("Checking transport parameters...")
try:
    from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
    import inspect
    
    print("FastAPIWebsocketParams fields:")
    sig = inspect.signature(FastAPIWebsocketParams)
    for param in sig.parameters:
        print(f"  - {param}")
except Exception as e:
    print(f"Error: {e}")