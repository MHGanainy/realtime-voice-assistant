import { CONFIG } from '../config';

export const encodeAudioFrame = (audioData, frameType) => {
  if (!frameType) return null;

  try {
    const message = frameType.create({
      audio: {
        audio: audioData,
        sampleRate: CONFIG.MIC_SAMPLE_RATE,
        numChannels: CONFIG.CHANNELS,
      }
    });
    return frameType.encode(message).finish();
  } catch (error) {
    console.error("Encode error:", error);
    return null;
  }
};

export const decodeFrame = (data, frameType) => {
  if (!frameType) return null;

  try {
    const decoded = frameType.decode(new Uint8Array(data));
    return frameType.toObject(decoded, {
      bytes: Uint8Array,
      longs: Number,
      defaults: true,
    });
  } catch (error) {
    console.error("Decode error:", error);
    return null;
  }
};

export const playAudioQueue = async (refs, updateState) => {
  const { audioQueue, audioContext } = refs.current;
  
  // Check if already playing
  if (refs.current.isPlaying || !audioQueue.length || !audioContext) return;

  refs.current.isPlaying = true;
  
  // Only update state once at the beginning
  if (!refs.current.isAssistantSpeaking) {
    refs.current.isAssistantSpeaking = true;
    updateState({ isAssistantSpeaking: true, isMicMuted: true });
  }

  // Initialize next start time if needed
  if (refs.current.nextStartTime < audioContext.currentTime) {
    refs.current.nextStartTime = audioContext.currentTime + 0.05;
  }

  while (audioQueue.length > 0 && refs.current.audioContext) {
    const { bytes: audioData, rate } = audioQueue.shift();

    try {
      if (refs.current.audioContext.state === 'closed') break;

      const aligned = audioData.byteOffset & 1 ? new Uint8Array(audioData) : audioData;
      const int16 = new Int16Array(aligned.buffer, aligned.byteOffset, aligned.byteLength >> 1);
      const float32 = new Float32Array(int16.length);
      
      for (let i = 0; i < int16.length; i++) {
        float32[i] = int16[i] / 32768;
      }

      const buffer = refs.current.audioContext.createBuffer(1, float32.length, rate);
      buffer.getChannelData(0).set(float32);

      const source = refs.current.audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(refs.current.audioContext.destination);

      // Schedule playback for seamless audio
      source.start(refs.current.nextStartTime);
      
      // Calculate next start time
      const duration = buffer.duration;
      refs.current.nextStartTime += duration;

      // Wait for this chunk to finish
      await new Promise(resolve => {
        source.onended = resolve;
      });

    } catch (error) {
      console.error("Playback error:", error);
      break;
    }
  }

  refs.current.isPlaying = false;
  refs.current.isAssistantSpeaking = false;
  refs.current.nextStartTime = 0;
  updateState({ isAssistantSpeaking: false, isMicMuted: false, status: "Ready for next input" });
};

export const setupAudioWorklet = async (stream, refs, encodeCallback) => {
  const audioContext = new AudioContext({ sampleRate: CONFIG.MIC_SAMPLE_RATE });
  refs.current.audioContext = audioContext;

  // Ensure audio context is running
  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }

  const processorCode = `
    class AudioProcessor extends AudioWorkletProcessor {
      constructor() {
        super();
        this.buffer = [];
      }

      process(inputs) {
        const input = inputs[0]?.[0];
        if (!input) return true;
        
        for (const sample of input) {
          this.buffer.push(sample);
          
          if (this.buffer.length >= ${CONFIG.CHUNK_SIZE}) {
            const int16 = new Int16Array(${CONFIG.CHUNK_SIZE});
            for (let i = 0; i < ${CONFIG.CHUNK_SIZE}; i++) {
              const s = Math.max(-1, Math.min(1, this.buffer[i]));
              int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            
            this.port.postMessage({ type: 'audio', data: int16.buffer });
            this.buffer = [];
          }
        }
        return true;
      }
    }
    registerProcessor('audio-processor', AudioProcessor);
  `;

  const blob = new Blob([processorCode], { type: 'application/javascript' });
  await audioContext.audioWorklet.addModule(URL.createObjectURL(blob));

  const source = audioContext.createMediaStreamSource(stream);
  const processor = new AudioWorkletNode(audioContext, 'audio-processor');
  
  processor.port.onmessage = (event) => {
    if (event.data.type === 'audio' && 
        !refs.current.isAssistantSpeaking && 
        refs.current.websocket?.readyState === WebSocket.OPEN) {
      const encoded = encodeCallback(new Uint8Array(event.data.data));
      if (encoded) {
        refs.current.websocket.send(encoded);
      }
    }
  };

  source.connect(processor);
  // DON'T connect processor to destination - this causes feedback!
  
  refs.current.processor = processor;
};