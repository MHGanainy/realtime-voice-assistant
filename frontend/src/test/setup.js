// Mock Web Audio API
global.AudioContext = class MockAudioContext {
    constructor() {
      this.sampleRate = 48000;
      this.currentTime = 0;
      this.destination = {};
      this.state = 'running';
    }
  
    createMediaStreamSource() {
      return { connect: () => {} };
    }
  
    createBufferSource() {
      return {
        buffer: null,
        connect: () => {},
        start: () => {},
        onended: null
      };
    }
  
    createBuffer(channels, frames, sampleRate) {
      return {
        copyToChannel: () => {},
        getChannelData: () => new Float32Array(frames)
      };
    }
  
    close() {
      this.state = 'closed';
      return Promise.resolve();
    }
  };
  
  global.AudioWorkletNode = class MockAudioWorkletNode {
    constructor() {
      this.port = {
        postMessage: () => {},
        onmessage: null
      };
    }
  
    disconnect() {}
  };
  
  // Mock MediaDevices
  global.navigator.mediaDevices = {
    getUserMedia: () => Promise.resolve({
      getTracks: () => [{ stop: () => {} }]
    })
  };
  
  // Mock performance.now() if not available
  if (!global.performance) {
    global.performance = {
      now: () => Date.now()
    };
  }