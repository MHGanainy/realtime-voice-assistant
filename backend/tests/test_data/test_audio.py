import numpy as np

def generate_test_tone(frequency=440, duration=1.0, sample_rate=16000):
    """Generate a test tone (sine wave) as PCM data"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    pcm = (wave * 32767).astype(np.int16)
    return pcm.tobytes()

def generate_silence(duration=1.0, sample_rate=16000):
    """Generate silence as PCM data"""
    samples = int(sample_rate * duration)
    return bytes(samples * 2)  # 2 bytes per sample