// src/services/vadProcessor.js
/**
 * Client-side Voice Activity Detection using WebRTC VAD
 * Reduces unnecessary network traffic and improves responsiveness
 */
export class VADProcessor {
    constructor(options = {}) {
      this.options = {
        sampleRate: options.sampleRate || 16000,
        frameDuration: options.frameDuration || 30, // ms
        mode: options.mode || 2, // 0-3, higher = more aggressive
        minSpeechFrames: options.minSpeechFrames || 3,
        minSilenceFrames: options.minSilenceFrames || 10,
        preSpeechPadFrames: options.preSpeechPadFrames || 5,
        ...options
      };
      
      this.frameSize = (this.options.sampleRate * this.options.frameDuration) / 1000;
      this.buffer = new Float32Array(this.frameSize);
      this.bufferIndex = 0;
      
      // State tracking
      this.isSpeaking = false;
      this.speechFrameCount = 0;
      this.silenceFrameCount = 0;
      this.preSpeechBuffer = [];
      this.vadActive = false;
      
      // Callbacks
      this.onSpeechStart = options.onSpeechStart || (() => {});
      this.onSpeechEnd = options.onSpeechEnd || (() => {});
      this.onVoiceActivity = options.onVoiceActivity || (() => {});
      this.onProcessedAudio = options.onProcessedAudio || (() => {});
      
      // Energy-based VAD parameters
      this.energyThreshold = 0.001;  // Lower threshold for more sensitivity
      this.dynamicEnergyThreshold = this.energyThreshold;
      this.energyHistory = [];
      this.maxEnergyHistory = 50;
      
      // Debug
      this.frameCount = 0;
      console.log('VAD initialized with settings:', {
        sampleRate: this.options.sampleRate,
        frameDuration: this.options.frameDuration,
        frameSize: this.frameSize,
        mode: this.options.mode
      });
    }
    
    processAudio(inputData) {
      // If input is smaller than frame size, buffer it
      if (inputData.length < this.frameSize) {
        // Add to buffer
        for (let i = 0; i < inputData.length; i++) {
          this.buffer[this.bufferIndex++] = inputData[i];
          
          if (this.bufferIndex >= this.frameSize) {
            // Process complete frame
            this.processFrame(this.buffer.slice(0, this.frameSize));
            this.bufferIndex = 0;
          }
        }
      } else {
        // Process input in frame-sized chunks
        for (let i = 0; i < inputData.length; i += this.frameSize) {
          const frameEnd = Math.min(i + this.frameSize, inputData.length);
          const frame = inputData.slice(i, frameEnd);
          
          if (frame.length === this.frameSize) {
            this.processFrame(frame);
          } else {
            // Buffer incomplete frame
            for (let j = 0; j < frame.length; j++) {
              this.buffer[this.bufferIndex++] = frame[j];
            }
          }
        }
      }
    }
    
    processFrame(frame) {
      this.frameCount++;
      
      const isSpeech = this.detectVoiceActivity(frame);
      const energy = this.calculateEnergy(frame);
      
      // Debug logging every 100 frames
      if (this.frameCount % 100 === 0) {
        console.log(`VAD Frame ${this.frameCount}: energy=${energy.toFixed(4)}, threshold=${this.dynamicEnergyThreshold.toFixed(4)}, isSpeech=${isSpeech}, speechFrames=${this.speechFrameCount}`);
      }
      
      // Update state based on detection
      if (isSpeech) {
        this.speechFrameCount++;
        this.silenceFrameCount = 0;
        
        if (!this.isSpeaking && this.speechFrameCount >= this.options.minSpeechFrames) {
          console.log('VAD: Starting speech after', this.speechFrameCount, 'speech frames');
          this.startSpeech();
        }
      } else {
        this.silenceFrameCount++;
        this.speechFrameCount = 0;
        
        if (this.isSpeaking && this.silenceFrameCount >= this.options.minSilenceFrames) {
          console.log('VAD: Ending speech after', this.silenceFrameCount, 'silence frames');
          this.endSpeech();
        }
      }
      
      // Store frame in pre-speech buffer
      if (!this.isSpeaking) {
        this.preSpeechBuffer.push(new Float32Array(frame));
        if (this.preSpeechBuffer.length > this.options.preSpeechPadFrames) {
          this.preSpeechBuffer.shift();
        }
      }
      
      // Notify about voice activity
      this.onVoiceActivity({
        isSpeech,
        isSpeaking: this.isSpeaking,
        energy: energy,
        threshold: this.dynamicEnergyThreshold
      });
      
      // If we're speaking, pass the audio through
      if (this.isSpeaking) {
        this.onProcessedAudio(frame);
      }
      
      return isSpeech;
    }
    
    detectVoiceActivity(frame) {
      const energy = this.calculateEnergy(frame);
      
      // Update energy history
      this.energyHistory.push(energy);
      if (this.energyHistory.length > this.maxEnergyHistory) {
        this.energyHistory.shift();
      }
      
      // Update dynamic threshold
      this.updateDynamicThreshold();
      
      // Basic energy-based detection
      const isEnergyBased = energy > this.dynamicEnergyThreshold;
      
      // For debugging: if mode is 0, only use energy
      if (this.options.mode === 0) {
        return isEnergyBased;
      }
      
      // Additional checks
      const hasFrequencyContent = this.checkFrequencyContent(frame);
      const hasZeroCrossings = this.checkZeroCrossings(frame);
      
      // Debug log for first few detections
      if (this.frameCount < 10 && isEnergyBased) {
        console.log('VAD Detection:', {
          energy: energy.toFixed(4),
          threshold: this.dynamicEnergyThreshold.toFixed(4),
          hasFrequency: hasFrequencyContent,
          hasZeroCrossings: hasZeroCrossings,
          mode: this.options.mode
        });
      }
      
      // Combine detections based on mode
      switch (this.options.mode) {
        case 0: // Very permissive
          return isEnergyBased;
        case 1: // Permissive
          return isEnergyBased && (hasFrequencyContent || hasZeroCrossings);
        case 2: // Balanced
          return isEnergyBased && hasFrequencyContent;
        case 3: // Aggressive
          return isEnergyBased && hasFrequencyContent && hasZeroCrossings;
        default:
          return isEnergyBased;
      }
    }
    
    calculateEnergy(frame) {
      let sum = 0;
      for (let i = 0; i < frame.length; i++) {
        sum += frame[i] * frame[i];
      }
      return Math.sqrt(sum / frame.length);
    }
    
    updateDynamicThreshold() {
      if (this.energyHistory.length < 10) return;
      
      // Calculate statistics
      const sorted = [...this.energyHistory].sort((a, b) => a - b);
      const percentile20 = sorted[Math.floor(sorted.length * 0.2)];
      const percentile80 = sorted[Math.floor(sorted.length * 0.8)];
      
      // Set threshold between noise floor and speech level
      this.dynamicEnergyThreshold = percentile20 + (percentile80 - percentile20) * 0.3;
      this.dynamicEnergyThreshold = Math.max(this.dynamicEnergyThreshold, this.energyThreshold);
    }
    
    checkFrequencyContent(frame) {
      // Simple frequency analysis using autocorrelation
      let maxCorr = 0;
      const minPeriod = Math.floor(this.options.sampleRate / 400); // 400 Hz
      const maxPeriod = Math.floor(this.options.sampleRate / 80);  // 80 Hz
      
      for (let period = minPeriod; period < maxPeriod && period < frame.length; period++) {
        let correlation = 0;
        for (let i = 0; i < frame.length - period; i++) {
          correlation += frame[i] * frame[i + period];
        }
        correlation /= frame.length - period;
        maxCorr = Math.max(maxCorr, Math.abs(correlation));
      }
      
      return maxCorr > 0.3; // Threshold for voiced speech
    }
    
    checkZeroCrossings(frame) {
      let crossings = 0;
      for (let i = 1; i < frame.length; i++) {
        if ((frame[i] >= 0) !== (frame[i - 1] >= 0)) {
          crossings++;
        }
      }
      
      const crossingRate = crossings / frame.length;
      return crossingRate > 0.1 && crossingRate < 0.5; // Typical range for speech
    }
    
    startSpeech() {
      this.isSpeaking = true;
      this.vadActive = true;
      
      // Include pre-speech buffer
      const preSpeechData = this.preSpeechBuffer.reduce((acc, frame) => {
        return Float32Array.from([...acc, ...frame]);
      }, new Float32Array(0));
      
      this.onSpeechStart({
        preSpeechData,
        timestamp: Date.now()
      });
      
      this.preSpeechBuffer = [];
    }
    
    endSpeech() {
      this.isSpeaking = false;
      this.vadActive = false;
      
      this.onSpeechEnd({
        timestamp: Date.now(),
        duration: this.speechFrameCount * this.options.frameDuration
      });
      
      this.speechFrameCount = 0;
      this.silenceFrameCount = 0;
    }
    
    reset() {
      this.isSpeaking = false;
      this.speechFrameCount = 0;
      this.silenceFrameCount = 0;
      this.preSpeechBuffer = [];
      this.vadActive = false;
      this.energyHistory = [];
      this.bufferIndex = 0;
      this.buffer.fill(0);
    }
  }