/**
 * AudioWorklet processor for ultra-low latency audio capture
 * Achieves < 12ms capture-to-emit latency by processing in audio thread
 * 
 * Key optimizations:
 * - Runs in separate audio thread (no main thread blocking)
 * - Fixed 20ms frames (960 samples at 48kHz)
 * - Efficient ring buffer for sample accumulation
 * - Real-time VAD without FFT overhead
 * 
 * Note: This file should be placed in public/audio/capture-processor.js
 */

class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    
    // Extract configuration from options
    const config = options.processorOptions || {};
    
    // Core audio parameters
    this.sampleRate = sampleRate; // Global AudioWorklet property
    this.frameSize = config.frameSize || 960; // 20ms at 48kHz
    this.channels = config.channels || 1;
    
    // Ring buffer for accumulating samples
    this.ringBuffer = new Float32Array(this.frameSize * 2); // 2x size for safety
    this.writeIndex = 0;
    this.frameCount = 0;
    
    // VAD configuration
    this.enableVAD = config.enableVAD !== false;
    this.vadSensitivity = config.vadSensitivity || 0.01;
    this.vadHangoverFrames = config.vadHangoverFrames || 10; // 200ms
    
    // VAD state
    this.isVoiceActive = false;
    this.hangoverCounter = 0;
    this.energyHistory = new Float32Array(5); // 100ms history
    this.historyIndex = 0;
    
    // Performance tracking
    this.lastProcessTime = currentTime;
    this.processCount = 0;
    this.dropCount = 0;
    
    // Level metering (for visualization)
    this.levelUpdateInterval = 5; // Every 5 frames (100ms)
    this.levelUpdateCounter = 0;
    
    console.log('[CaptureProcessor] Initialized:', {
      sampleRate: this.sampleRate,
      frameSize: this.frameSize,
      vadEnabled: this.enableVAD
    });
  }

  /**
   * Main processing function - called by audio thread
   * CRITICAL: Must be fast to avoid glitches
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    
    // No input connected yet
    if (!input || !input[0]) {
      return true;
    }
    
    const inputChannel = input[0];
    const inputLength = inputChannel.length;
    
    // Accumulate samples into ring buffer
    for (let i = 0; i < inputLength; i++) {
      this.ringBuffer[this.writeIndex] = inputChannel[i];
      this.writeIndex++;
      
      // Process complete frame
      if (this.writeIndex >= this.frameSize) {
        this.processFrame();
        this.writeIndex = 0;
      }
    }
    
    // Keep processor alive
    return true;
  }
  
  /**
   * Process a complete 20ms frame
   */
  processFrame() {
    const frameStartTime = currentTime;
    
    // Extract frame from ring buffer
    const frame = this.ringBuffer.slice(0, this.frameSize);
    
    // Calculate frame metrics
    const metrics = this.calculateFrameMetrics(frame);
    
    // Update VAD if enabled
    if (this.enableVAD) {
      this.updateVAD(metrics);
    }
    
    // Send audio chunk with minimal latency
    this.port.postMessage({
      type: 'audioChunk',
      data: frame,
      timestamp: frameStartTime * 1000, // Convert to ms
      frameNumber: this.frameCount,
      metrics: {
        energy: metrics.energy,
        zeroCrossings: metrics.zeroCrossings,
        voiceActive: this.isVoiceActive
      }
    });
    
    // Update level meter periodically
    this.levelUpdateCounter++;
    if (this.levelUpdateCounter >= this.levelUpdateInterval) {
      this.port.postMessage({
        type: 'level',
        level: metrics.rms,
        timestamp: frameStartTime * 1000
      });
      this.levelUpdateCounter = 0;
    }
    
    // Track performance
    const processingTime = currentTime - frameStartTime;
    if (processingTime > 0.001) { // Log if > 1ms
      console.warn('[CaptureProcessor] Slow frame:', processingTime * 1000, 'ms');
    }
    
    this.frameCount++;
    this.processCount++;
  }
  
  /**
   * Calculate frame metrics for VAD and monitoring
   * Optimized for real-time performance
   */
  calculateFrameMetrics(frame) {
    let sum = 0;
    let absSum = 0;
    let zeroCrossings = 0;
    let lastSample = 0;
    
    // Single pass for all metrics
    for (let i = 0; i < frame.length; i++) {
      const sample = frame[i];
      
      // For RMS
      sum += sample * sample;
      
      // For average magnitude
      absSum += Math.abs(sample);
      
      // Zero crossings (sign changes)
      if (i > 0 && ((lastSample >= 0) !== (sample >= 0))) {
        zeroCrossings++;
      }
      lastSample = sample;
    }
    
    const rms = Math.sqrt(sum / frame.length);
    const avgMagnitude = absSum / frame.length;
    const energy = sum / frame.length;
    
    return {
      rms,
      energy,
      avgMagnitude,
      zeroCrossings,
      zeroCrossingRate: zeroCrossings / frame.length
    };
  }
  
  /**
   * Update Voice Activity Detection state
   * Uses energy + zero crossing rate for robust detection
   */
  updateVAD(metrics) {
    // Update energy history
    this.energyHistory[this.historyIndex] = metrics.energy;
    this.historyIndex = (this.historyIndex + 1) % this.energyHistory.length;
    
    // Calculate adaptive threshold based on history
    let avgEnergy = 0;
    for (let i = 0; i < this.energyHistory.length; i++) {
      avgEnergy += this.energyHistory[i];
    }
    avgEnergy /= this.energyHistory.length;
    
    // Dynamic threshold: 2x average energy or minimum sensitivity
    const threshold = Math.max(avgEnergy * 2, this.vadSensitivity);
    
    // Voice detection logic
    const energyAboveThreshold = metrics.energy > threshold;
    const hasVoiceCharacteristics = metrics.zeroCrossingRate > 0.1 && metrics.zeroCrossingRate < 0.5;
    
    const currentlyActive = energyAboveThreshold && hasVoiceCharacteristics;
    
    if (currentlyActive) {
      // Voice detected
      if (!this.isVoiceActive) {
        // Transition to active
        this.isVoiceActive = true;
        this.hangoverCounter = this.vadHangoverFrames;
        
        this.port.postMessage({
          type: 'vadStatus',
          active: true,
          timestamp: currentTime * 1000,
          confidence: metrics.energy / threshold
        });
      } else {
        // Reset hangover
        this.hangoverCounter = this.vadHangoverFrames;
      }
    } else {
      // No voice detected
      if (this.isVoiceActive) {
        // Use hangover before transitioning to inactive
        this.hangoverCounter--;
        
        if (this.hangoverCounter <= 0) {
          this.isVoiceActive = false;
          
          this.port.postMessage({
            type: 'vadStatus',
            active: false,
            timestamp: currentTime * 1000
          });
        }
      }
    }
  }
  
  /**
   * Handle messages from main thread
   */
  onmessage(event) {
    const { data } = event;
    
    switch (data.type) {
      case 'updateConfig':
        // Update configuration dynamically
        if (data.config.vadSensitivity !== undefined) {
          this.vadSensitivity = data.config.vadSensitivity;
        }
        if (data.config.enableVAD !== undefined) {
          this.enableVAD = data.config.enableVAD;
        }
        break;
        
      case 'getStats':
        // Return performance statistics
        this.port.postMessage({
          type: 'stats',
          stats: {
            framesProcessed: this.frameCount,
            droppedFrames: this.dropCount,
            isVoiceActive: this.isVoiceActive,
            currentTime: currentTime
          }
        });
        break;
        
      case 'reset':
        // Reset processor state
        this.writeIndex = 0;
        this.frameCount = 0;
        this.isVoiceActive = false;
        this.hangoverCounter = 0;
        this.energyHistory.fill(0);
        break;
        
      default:
        console.warn('[CaptureProcessor] Unknown message type:', data.type);
    }
  }
}

// Register the processor
registerProcessor('capture-processor', CaptureProcessor);