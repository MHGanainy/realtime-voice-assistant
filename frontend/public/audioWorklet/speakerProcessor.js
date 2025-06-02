class SpeakerProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.bufferQueue = [];
      this.isPlaying = false;
      this.currentBuffer = null;
      this.currentBufferIndex = 0;
      
      this.port.onmessage = (event) => {
        if (event.data.type === 'audio') {
          this.bufferQueue.push(event.data.buffer);
          if (!this.isPlaying) {
            this.isPlaying = true;
          }
        } else if (event.data.type === 'clear') {
          this.bufferQueue = [];
          this.currentBuffer = null;
          this.currentBufferIndex = 0;
          this.isPlaying = false;
        }
      };
    }
  
    process(inputs, outputs, parameters) {
      const output = outputs[0];
      if (!output || !output[0]) return true;
      
      const outputChannel = output[0];
      
      if (!this.isPlaying) {
        // Output silence
        outputChannel.fill(0);
        return true;
      }
      
      let outputIndex = 0;
      
      while (outputIndex < outputChannel.length) {
        // Get next buffer if needed
        if (!this.currentBuffer || this.currentBufferIndex >= this.currentBuffer.length) {
          if (this.bufferQueue.length > 0) {
            this.currentBuffer = this.bufferQueue.shift();
            this.currentBufferIndex = 0;
          } else {
            // No more buffers, fill with silence
            for (let i = outputIndex; i < outputChannel.length; i++) {
              outputChannel[i] = 0;
            }
            this.isPlaying = false;
            break;
          }
        }
        
        // Copy samples from current buffer
        const remainingOutput = outputChannel.length - outputIndex;
        const remainingBuffer = this.currentBuffer.length - this.currentBufferIndex;
        const samplesToWrite = Math.min(remainingOutput, remainingBuffer);
        
        for (let i = 0; i < samplesToWrite; i++) {
          outputChannel[outputIndex++] = this.currentBuffer[this.currentBufferIndex++];
        }
      }
      
      return true;
    }
  }
  
  registerProcessor('speaker-processor', SpeakerProcessor);
  