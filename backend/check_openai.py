# backend/check_openai.py
"""
Check OpenAI compatibility with Pipecat
"""
import subprocess
import sys

print("Checking OpenAI compatibility...")
print("=" * 50)

# Check current OpenAI version
result = subprocess.run(['pip', 'show', 'openai'], capture_output=True, text=True)
if result.returncode == 0:
    lines = result.stdout.strip().split('\n')
    version_line = [line for line in lines if line.startswith('Version:')][0]
    print(f"Current OpenAI version: {version_line.split(': ')[1]}")

print("\nChecking what Pipecat expects from OpenAI...")
try:
    # Read the openai.py file from pipecat
    with open("/opt/anaconda3/envs/backend/lib/python3.11/site-packages/pipecat/services/openai.py", "r") as f:
        lines = f.readlines()
        print("\nOpenAI imports in Pipecat (lines 45-55):")
        for i, line in enumerate(lines[45:55]):
            print(f"  {i+45}: {line.rstrip()}")
except Exception as e:
    print(f"Error reading file: {e}")

print("\nChecking available OpenAI classes...")
try:
    import openai
    print(f"OpenAI module location: {openai.__file__}")
    print("\nAvailable in openai module:")
    relevant_items = [item for item in dir(openai) if 'Async' in item or 'Client' in item or 'Httpx' in item]
    for item in relevant_items:
        print(f"  - {item}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 50)
print("Recommendations:")
print("1. Try: pip install openai==1.12.0")
print("2. Or downgrade Pipecat: pip install pipecat-ai==0.0.39")
print("3. Or use the latest compatible versions of everything")