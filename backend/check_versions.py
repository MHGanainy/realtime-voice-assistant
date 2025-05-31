# backend/check_versions.py
"""
Check installed package versions
"""
import subprocess
import sys

print("Python version:", sys.version)
print("\nInstalled packages:")
print("-" * 50)

packages = ['pipecat-ai', 'deepgram-sdk', 'openai', 'elevenlabs']

for package in packages:
    try:
        result = subprocess.run(['pip', 'show', package], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            version_line = [line for line in lines if line.startswith('Version:')][0]
            print(f"{package}: {version_line.split(': ')[1]}")
        else:
            print(f"{package}: Not installed")
    except Exception as e:
        print(f"{package}: Error checking - {e}")

print("\nChecking Pipecat's Deepgram import...")
try:
    import pipecat
    print(f"Pipecat location: {pipecat.__file__}")
    
    # Check what Deepgram service expects
    with open("/opt/anaconda3/envs/backend/lib/python3.11/site-packages/pipecat/services/deepgram.py", "r") as f:
        lines = f.readlines()
        print("\nDeepgram imports in Pipecat:")
        for i, line in enumerate(lines[30:40]):  # Lines around the import
            print(f"  {i+30}: {line.rstrip()}")
except Exception as e:
    print(f"Error: {e}")