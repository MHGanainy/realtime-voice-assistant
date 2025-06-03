/**
 * VoiceRoom - Main abstraction for voice conversations
 * Enhanced with real WebCodecs AudioDecoder for Opus decoding
 */

export class VoiceRoom extends EventTarget {
  constructor(options) {
    super();
    
    this.roomId = options.roomId;
    this.userId = options.userId;
    this.wsUrl = options.wsUrl || this._getDefaultWsUrl();
    
    // Audio components
    this.capture = options.capture;
    this.player = options.player;
    
    // State
    this.socket = null;
    this.isConnected = false;
    this.isTTSPlaying = false;
    this.participants = new Set();
    
    // Opus decoding state
    this.opusDecoder = null;
    this.isDecoderReady = false;
    this.decoderConfig = null;
    
    // Quality metrics
    this.enableQualityMetrics = false;
    this._originalAudioBuffer = null;
    
    // Metrics
    this.metrics = {
      captureToSendLatency: 0,
      packetsent: 0,
      packetReceived: 0,
      bytesent: 0,
      byteReceived: 0,
      lastPingTime: 0,
      rtt: 0,
      compressionRatio: 0,
      audioQuality: { snr: 0, correlation: 0 },
      decoderState: 'unconfigured'
    };
    
    // Reconnection
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 30000;
    this.reconnectAttempts = 0;
    this.shouldReconnect = true;
    
    // Bind methods
    this._handleAudioChunk = this._handleAudioChunk.bind(this);
    this._handleVoiceStart = this._handleVoiceStart.bind(this);
    this._handleVoiceEnd = this._handleVoiceEnd.bind(this);
  }

  _getDefaultWsUrl() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = import.meta.env.VITE_WS_URL || `${wsProtocol}//${window.location.host}`;
    return `${wsHost}/ws/room/${this.roomId}`;
  }

  async join() {
    this.shouldReconnect = true;
    await this._connect();
  }

  async _connect() {
    try {
      // Create WebSocket connection
      this.socket = new WebSocket(this.wsUrl);
      this.socket.binaryType = 'arraybuffer';
      
      // Setup event handlers
      this.socket.onopen = async () => {
        this.isConnected = true;
        this.reconnectAttempts = 0;
        console.log(`Connected to room ${this.roomId}`);
        
        // Send join message
        this._sendJson({
          type: 'join',
          userId: this.userId
        });
        
        // Initialize audio after successful connection
        await this._initializeAudio();
        
        // Initialize Opus decoder with real WebCodecs
        await this._initializeOpusDecoder();
        
        // Start metrics collection
        this._startMetricsCollection();
        
        this.emit('connected');
      };
      
      this.socket.onmessage = (event) => {
        this._handleMessage(event);
      };
      
      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };
      
      this.socket.onclose = () => {
        this.isConnected = false;
        console.log('WebSocket closed');
        
        // Cleanup
        this._cleanup();
        
        this.emit('disconnected');
        
        // Attempt reconnection
        if (this.shouldReconnect) {
          this._scheduleReconnect();
        }
      };
      
    } catch (error) {
      console.error('Failed to connect:', error);
      this.emit('error', error);
      
      if (this.shouldReconnect) {
        this._scheduleReconnect();
      }
    }
  }

  async _initializeAudio() {
    // Initialize audio capture
    if (this.capture) {
      await this.capture.initialize();
      await this.capture.start();
      
      // Listen for audio events
      this.capture.on('audioChunk', this._handleAudioChunk);
      this.capture.on('voiceStart', this._handleVoiceStart);
      this.capture.on('voiceEnd', this._handleVoiceEnd);
      this.capture.on('level', (level) => {
        this.emit('micLevel', level);
      });
    }
    
    // Initialize audio player
    if (this.player) {
      await this.player.initialize();
    }
  }

  async _initializeOpusDecoder() {
    if (!('AudioDecoder' in window)) {
      console.warn('WebCodecs AudioDecoder not available, falling back to raw PCM');
      this.metrics.decoderState = 'unavailable';
      return;
    }

    try {
      // First check if Opus is supported
      const { supported } = await AudioDecoder.isConfigSupported({
        codec: 'opus',
        sampleRate: 48000,
        numberOfChannels: 1
      });

      if (!supported) {
        console.warn('Opus codec not supported by AudioDecoder');
        this.isDecoderReady = false;
        this.metrics.decoderState = 'unsupported';
        return;
      }

      // Create decoder with output and error handlers
      this.opusDecoder = new AudioDecoder({
        output: (audioData) => this._handleDecodedAudio(audioData),
        error: (error) => {
          console.error('Opus decoder error:', error);
          this.metrics.decoderState = 'error';
          this.emit('decoderError', error);
        }
      });

      // Configure for Opus
      const config = {
        codec: 'opus',
        sampleRate: 48000,
        numberOfChannels: 1
      };

      await this.opusDecoder.configure(config);
      
      // Store config for reference
      this.decoderConfig = config;
      this.opusDecoder.codecConfig = config;

      this.isDecoderReady = true;
      this.metrics.decoderState = 'configured';
      console.log('Opus decoder initialized');
      
    } catch (error) {
      console.error('Failed to initialize Opus decoder:', error);
      this.isDecoderReady = false;
      this.metrics.decoderState = 'error';
    }
  }

  _detectOpusFrame(audioData) {
    // Check if this is an Opus frame with our header format
    if (audioData.byteLength < 8) {
      return { isOpusFrame: false, sequence: null, timestamp: null, opusData: null };
    }

    const view = new DataView(audioData);
    const sequence = view.getUint32(0, false); // Big endian
    const timestamp = view.getUint32(4, false);

    // Sanity checks for valid frame
    // Sequence numbers should be reasonable (not huge)
    // Timestamp should be a valid 32-bit value
    if (sequence < 1000000 && timestamp > 0) {
      // Extract Opus data
      const opusData = new Uint8Array(audioData.slice(8));
      
      // Additional check: Opus frames are typically small
      if (opusData.length > 0 && opusData.length < 500) {
        return {
          isOpusFrame: true,
          sequence,
          timestamp,
          opusData
        };
      }
    }

    return { isOpusFrame: false, sequence: null, timestamp: null, opusData: null };
  }

  _handleMessage(event) {
    if (typeof event.data === 'string') {
      // JSON message
      const message = JSON.parse(event.data);
      this._handleJsonMessage(message);
    } else {
      // Binary audio data
      this._handleAudioData(event.data);
    }
  }

  _handleJsonMessage(message) {
    switch (message.type) {
      case 'joined':
        this.participants = new Set(message.participants);
        this.emit('joined', { roomId: message.roomId });
        break;
        
      case 'participant_joined':
        this.participants.add(message.userId);
        this.emit('participantJoined', { userId: message.userId });
        break;
        
      case 'participant_left':
        this.participants.delete(message.userId);
        this.emit('participantLeft', { userId: message.userId });
        break;
        
      case 'participant_speaking':
        this.emit('participantSpeaking', {
          userId: message.userId,
          speaking: message.speaking
        });
        break;
        
      case 'pong':
        const rtt = Date.now() - message.timestamp;
        this.metrics.rtt = rtt;
        this.emit('ping', { rtt });
        break;
        
      default:
        console.warn('Unknown message type:', message.type);
    }
  }

  async _handleAudioData(audioData) {
    this.metrics.packetReceived++;
    this.metrics.byteReceived += audioData.byteLength;
    
    // Detect if this is an Opus frame
    const frameInfo = this._detectOpusFrame(audioData);
    
    if (frameInfo.isOpusFrame) {
      console.log(`Received Opus frame: seq=${frameInfo.sequence}, size=${frameInfo.opusData.length} bytes`);
      
      // Calculate compression ratio
      const uncompressedSize = 1920; // 960 samples * 2 bytes
      this.metrics.compressionRatio = 1 - (frameInfo.opusData.length / uncompressedSize);
      
      // Decode Opus data using real WebCodecs
      if (this.isDecoderReady && this.opusDecoder.state === 'configured') {
        try {
          const chunk = new EncodedAudioChunk({
            type: 'key',
            timestamp: frameInfo.timestamp * 1000, // Convert to microseconds
            data: frameInfo.opusData
          });
          
          this.opusDecoder.decode(chunk);
        } catch (error) {
          console.error('Failed to decode Opus frame:', error);
          this.metrics.decoderState = 'decode_error';
        }
      } else {
        console.warn('Opus decoder not ready, skipping frame');
      }
    } else {
      // Raw PCM audio - play directly
      console.log(`Received raw PCM: ${audioData.byteLength} bytes`);
      if (this.player) {
        await this.player.play(audioData);
      }
    }
    
    this.emit('audioReceived', { 
      size: audioData.byteLength,
      isOpus: frameInfo.isOpusFrame,
      compressionRatio: this.metrics.compressionRatio
    });
  }

  async _handleDecodedAudio(audioData) {
    console.log(`Decoded audio: ${audioData.numberOfFrames} frames at ${audioData.sampleRate}Hz`);
    
    // Convert AudioData to ArrayBuffer for playback
    const pcmBuffer = new ArrayBuffer(audioData.numberOfFrames * 2); // 16-bit samples
    const pcmView = new Int16Array(pcmBuffer);
    
    // Extract audio data
    const floatData = new Float32Array(audioData.numberOfFrames);
    audioData.copyTo(floatData, { planeIndex: 0 });
    
    // Convert float32 to int16
    for (let i = 0; i < floatData.length; i++) {
      pcmView[i] = Math.max(-32768, Math.min(32767, floatData[i] * 32768));
    }
    
    // Clean up AudioData
    audioData.close();
    
    // Measure quality if enabled
    if (this.enableQualityMetrics && this._originalAudioBuffer) {
      const quality = this._measureAudioQuality(this._originalAudioBuffer, pcmBuffer);
      this.metrics.audioQuality = quality;
      console.log(`Audio quality - SNR: ${quality.snr.toFixed(1)}dB, Correlation: ${quality.correlation.toFixed(3)}`);
    }
    
    // Play the decoded audio
    if (this.player) {
      if (this.player.playDecodedOpus) {
        await this.player.playDecodedOpus(pcmBuffer);
      } else {
        await this.player.play(pcmBuffer);
      }
    }
    
    // Mark as playing for interrupt detection
    this.isTTSPlaying = true;
    setTimeout(() => {
      this.isTTSPlaying = false;
    }, audioData.duration / 1000); // Convert microseconds to milliseconds
  }

  _measureAudioQuality(originalBuffer, decodedBuffer) {
    // Ensure buffers have even byte lengths (required for Int16Array)
    if (!originalBuffer || originalBuffer.byteLength % 2 !== 0) {
      console.warn('Original buffer has invalid length:', originalBuffer?.byteLength);
      return { snr: 0, correlation: 0 };
    }
    if (!decodedBuffer || decodedBuffer.byteLength % 2 !== 0) {
      console.warn('Decoded buffer has invalid length:', decodedBuffer?.byteLength);
      return { snr: 0, correlation: 0 };
    }
    
    const original = new Int16Array(originalBuffer);
    const decoded = new Int16Array(decodedBuffer);
    
    // Ensure same length
    const length = Math.min(original.length, decoded.length);
    
    if (length === 0) {
      return { snr: 0, correlation: 0 };
    }
    
    // Calculate SNR (Signal-to-Noise Ratio)
    let signal = 0;
    let noise = 0;
    
    for (let i = 0; i < length; i++) {
      signal += original[i] * original[i];
      const diff = original[i] - decoded[i];
      noise += diff * diff;
    }
    
    // Avoid division by zero
    let snr = 0;
    if (signal > 0 && noise > 0) {
      snr = 10 * Math.log10(signal / noise);
    } else if (signal > 0 && noise === 0) {
      snr = Infinity; // Perfect reconstruction
    }
    
    // Calculate correlation coefficient
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    
    for (let i = 0; i < length; i++) {
      sumX += original[i];
      sumY += decoded[i];
      sumXY += original[i] * decoded[i];
      sumX2 += original[i] * original[i];
      sumY2 += decoded[i] * decoded[i];
    }
    
    const denominator = Math.sqrt((length * sumX2 - sumX * sumX) * (length * sumY2 - sumY * sumY));
    const correlation = denominator === 0 ? 0 : (length * sumXY - sumX * sumY) / denominator;
    
    return { 
      snr: isFinite(snr) ? snr : 0, 
      correlation: Math.abs(correlation) 
    };
  }

  _handleAudioChunk(chunk) {
    if (!this.isConnected || !this.socket) return;
    
    // Store original for quality comparison
    if (this.enableQualityMetrics && chunk.data) {
      // Ensure we store a properly aligned buffer
      const buffer = chunk.data.buffer.slice(chunk.data.byteOffset, chunk.data.byteOffset + chunk.data.byteLength);
      if (buffer.byteLength % 2 === 0) {
        this._originalAudioBuffer = buffer;
      } else {
        // Pad to even length if necessary
        const paddedBuffer = new ArrayBuffer(buffer.byteLength + 1);
        new Uint8Array(paddedBuffer).set(new Uint8Array(buffer));
        this._originalAudioBuffer = paddedBuffer;
      }
    }
    
    // Calculate latency
    const now = performance.now();
    if (chunk.timestamp) {
      this.metrics.captureToSendLatency = now - chunk.timestamp;
    }
    
    // Send audio data
    this.socket.send(chunk.data);
    
    // Update metrics
    this.metrics.packetsent++;
    this.metrics.bytesent += chunk.data.byteLength;
  }

  _handleVoiceStart() {
    if (this.isTTSPlaying) {
      // User is interrupting TTS
      this._sendJson({ type: 'interrupt' });
      this.isTTSPlaying = false;
      this.player?.stop();
    } else {
      // Normal voice activity
      this._sendJson({
        type: 'voice_activity',
        activity: 'start'
      });
    }
    
    this.emit('voiceStart');
  }

  _handleVoiceEnd() {
    this._sendJson({
      type: 'voice_activity',
      activity: 'end'
    });
    
    this.emit('voiceEnd');
  }

  _sendJson(data) {
    if (this.isConnected && this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(data));
    }
  }

  _cleanup() {
    // Stop audio capture event listeners
    if (this.capture) {
      this.capture.off('audioChunk', this._handleAudioChunk);
      this.capture.off('voiceStart', this._handleVoiceStart);
      this.capture.off('voiceEnd', this._handleVoiceEnd);
    }
    
    // Close Opus decoder properly
    if (this.opusDecoder) {
      // Flush any pending decodes
      if (this.opusDecoder.state === 'configured') {
        this.opusDecoder.flush().catch(e => {
          console.warn('Error flushing decoder:', e);
        });
      }
      
      // Close decoder
      if (this.opusDecoder.state !== 'closed') {
        this.opusDecoder.close();
      }
      
      this.opusDecoder = null;
      this.isDecoderReady = false;
      this.metrics.decoderState = 'closed';
    }
  }

  _scheduleReconnect() {
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts),
      this.maxReconnectDelay
    );
    
    this.reconnectAttempts++;
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      if (this.shouldReconnect) {
        this._connect();
      }
    }, delay);
  }

  _startMetricsCollection() {
    // Send ping every 10 seconds
    this._metricsInterval = setInterval(() => {
      this.collectMetrics();
      
      // Send ping for RTT measurement
      if (this.isConnected) {
        this._sendJson({
          type: 'ping',
          timestamp: Date.now()
        });
      }
    }, 10000);
  }

  collectMetrics() {
    const metrics = {
      ...this.metrics,
      timestamp: Date.now(),
      connected: this.isConnected,
      participants: this.participants.size,
      decoderReady: this.isDecoderReady,
      decoderState: this.opusDecoder?.state || 'none'
    };
    
    this.emit('metrics', metrics);
    return metrics;
  }

  async leave() {
    this.shouldReconnect = false;
    
    // Stop audio
    await this.capture?.stop();
    
    // Cleanup
    this._cleanup();
    
    // Close WebSocket
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    
    // Clear intervals
    clearInterval(this._metricsInterval);
    
    this.emit('left');
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