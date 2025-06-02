class MicProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.bufferSize = 512;
      this.buffer = new Float32Array(this.bufferSize);
      this.bufferIndex = 0;
    }
  
    convertFloat32ToS16PCM(float32Array) {
      const int16Array = new Int16Array(float32Array.length);
      for (let i = 0; i < float32Array.length; i++) {
        const v = Math.max(-1, Math.min(1, float32Array[i]));
        int16Array[i] = v < 0 ? v * 32768 : v * 32767;
      }
      return int16Array;
    }
  
    process(inputs, outputs, parameters) {
      const input = inputs[0];
      if (!input || !input[0]) return true;
  
      const inputData = input[0];
      
      // Buffer input data
      for (let i = 0; i < inputData.length; i++) {
        this.buffer[this.bufferIndex++] = inputData[i];
        
        if (this.bufferIndex >= this.bufferSize) {
          // Convert and send buffered data
          const pcmInt16 = this.convertFloat32ToS16PCM(this.buffer);
          const pcmBytes = new Uint8Array(pcmInt16.buffer);
          
          // Send to main thread
          this.port.postMessage({
            type: 'audio',
            data: pcmBytes
          });
          
          this.bufferIndex = 0;
        }
      }
  
      return true;
    }
  }
  
  registerProcessor('mic-processor', MicProcessor);