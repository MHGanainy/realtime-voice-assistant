/**
 * AudioPlayer - Handles audio playback with low latency
 * Now with dedicated Opus playback support
 */

export class AudioPlayer extends EventTarget {
  constructor(options = {}) {
    super();
    
    // Configuration
    this.sampleRate = 48000;
    this.channelCount = 1;
    this.bufferSize = options.bufferSize || 4096; // ~85ms at 48kHz
    
    // State
    this.context = null;
    this.isPlaying = false;
    this.playbackQueue = [];
    this.currentSource = null;
    
    // Metrics
    this.metrics = {
      packetsPlayed: 0,
      underruns: 0,
      totalLatency: 0
    };
  }

  async initialize() {
    // Create audio context with low latency hint
    this.context = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: this.sampleRate,
      latencyHint: 'interactive'
    });

    // Resume context if suspended (for browser autoplay policies)
    if (this.context.state === 'suspended') {
      await this.context.resume();
    }

    console.log(`AudioPlayer initialized: ${this.context.sampleRate}Hz, state: ${this.context.state}`);
  }

  async play(audioData) {
    if (!this.context) {
      await this.initialize();
    }

    try {
      const audioBuffer = await this._createAudioBuffer(audioData);
      await this._playAudioBuffer(audioBuffer);
      
      this.metrics.packetsPlayed++;
      this.emit('playbackStarted');
    } catch (error) {
      console.error('Playback error:', error);
      this.metrics.underruns++;
      this.emit('playbackError', error);
    }
  }

  async playDecodedOpus(pcmData) {
    // Specialized method for playing decoded Opus audio
    // This ensures we handle the exact format from our decoder
    if (!this.context) {
      await this.initialize();
    }

    try {
      // PCM data is already in the correct format (Int16 ArrayBuffer)
      const audioBuffer = await this._createAudioBuffer(pcmData);
      await this._playAudioBuffer(audioBuffer, true);
      
      this.metrics.packetsPlayed++;
      console.log(`Playing decoded Opus: ${pcmData.byteLength} bytes`);
    } catch (error) {
      console.error('Opus playback error:', error);
      this.metrics.underruns++;
    }
  }

  async _createAudioBuffer(audioData) {
    // Determine the format of the audio data
    let floatArray;
    
    if (audioData instanceof ArrayBuffer) {
      // Convert ArrayBuffer to Float32Array
      const int16Array = new Int16Array(audioData);
      floatArray = new Float32Array(int16Array.length);
      
      // Convert 16-bit PCM to float32 [-1, 1]
      for (let i = 0; i < int16Array.length; i++) {
        floatArray[i] = int16Array[i] / 32768;
      }
    } else if (audioData instanceof Float32Array) {
      floatArray = audioData;
    } else {
      throw new Error('Unsupported audio data format');
    }
    
    // Create AudioBuffer
    const audioBuffer = this.context.createBuffer(
      this.channelCount,
      floatArray.length,
      this.sampleRate
    );
    
    // Copy data to audio buffer
    audioBuffer.copyToChannel(floatArray, 0);
    
    return audioBuffer;
  }

  async _playAudioBuffer(audioBuffer, isOpus = false) {
    return new Promise((resolve) => {
      // Stop any currently playing audio if this is Opus (echo)
      if (isOpus && this.currentSource) {
        this.currentSource.stop();
        this.currentSource = null;
      }

      // Create buffer source
      const source = this.context.createBufferSource();
      source.buffer = audioBuffer;
      
      // Connect to destination
      source.connect(this.context.destination);
      
      // Track playback state
      source.onended = () => {
        this.isPlaying = false;
        this.currentSource = null;
        this.emit('playbackEnded');
        resolve();
      };
      
      // Start playback immediately
      const startTime = this.context.currentTime;
      source.start(startTime);
      
      this.isPlaying = true;
      this.currentSource = source;
      
      // Calculate latency
      const latency = (startTime - this.context.currentTime) * 1000;
      this.metrics.totalLatency += latency;
      
      console.log(`Audio playback started: ${audioBuffer.duration.toFixed(3)}s, latency: ${latency.toFixed(1)}ms`);
    });
  }

  stop() {
    if (this.currentSource) {
      try {
        this.currentSource.stop();
        this.currentSource = null;
        this.isPlaying = false;
        this.emit('playbackStopped');
      } catch (error) {
        // Source might have already stopped
        console.warn('Error stopping audio:', error);
      }
    }
  }

  async close() {
    this.stop();
    
    if (this.context && this.context.state !== 'closed') {
      await this.context.close();
      this.context = null;
    }
  }

  getMetrics() {
    const avgLatency = this.metrics.packetsPlayed > 0 
      ? this.metrics.totalLatency / this.metrics.packetsPlayed 
      : 0;
    
    return {
      ...this.metrics,
      averageLatency: avgLatency,
      contextState: this.context?.state || 'uninitialized'
    };
  }

  // EventTarget helpers
  emit(event, data) {
    this.dispatchEvent(new CustomEvent(event, { detail: data }));
  }

  on(event, handler) {
    this.addEventListener(event, (e) => handler(e.detail));
  }

  off(event, handler) {
    this.removeEventListener(event, handler);
  }
}