// Get WebSocket URL from environment variable or use default
const getWebSocketUrl = () => {
    // In Vite, environment variables are accessed via import.meta.env
    const envUrl = import.meta.env?.VITE_WS_URL;
    
    if (envUrl) {
      return envUrl;
    }
    
    // Default to localhost for development
    return 'ws://localhost:8000';
  };
  
  // Get API URL from WebSocket URL
  const getApiUrl = () => {
    const wsUrl = getWebSocketUrl();
    // Convert ws:// to http:// or wss:// to https://
    return wsUrl.replace(/^ws/, 'http');
  };
  
  export const CONFIG = {
    MIC_SAMPLE_RATE: 16000,
    CHANNELS: 1,
    CHUNK_SIZE: 320,
    WS_BASE_URL: getWebSocketUrl(),
    API_BASE_URL: getApiUrl(),
    EXIT_WORDS: ['goodbye', 'bye', 'exit', 'quit']
  };
  
  export const EVENT_STYLES = {
    'connection': { icon: '🔌', color: '#60a5fa' },
    'conversation': { icon: '💬', color: '#34d399' },
    'transcription': { icon: '📝', color: '#a78bfa' },
    'audio': { icon: '🔊', color: '#fbbf24' },
    'turn': { icon: '🔄', color: '#f472b6' },
    'metrics': { icon: '📊', color: '#94a3b8' },
    'error': { icon: '❌', color: '#ef4444' },
    'pipeline': { icon: '⚙️', color: '#06b6d4' },
    'global': { icon: '🌐', color: '#6366f1' },
    'system': { icon: '💻', color: '#8b5cf6' },
    'test': { icon: '🧪', color: '#ec4899' }
  };
  
  export const PIPECAT_PROTO = `
  syntax = "proto3";
  package pipecat;
  
  message TextFrame {
    uint64 id = 1;
    string name = 2;
    string text = 3;
  }
  
  message AudioRawFrame {
    uint64 id = 1;
    string name = 2;
    bytes audio = 3;
    uint32 sample_rate = 4;
    uint32 num_channels = 5;
    optional uint64 pts = 6;
  }
  
  message TranscriptionFrame {
    uint64 id = 1;
    string name = 2;
    string text = 3;
    string user_id = 4;
    string timestamp = 5;
  }
  
  message MessageFrame {
      string data = 1;
  }
  
  message Frame {
    oneof frame {
      TextFrame text = 1;
      AudioRawFrame audio = 2;
      TranscriptionFrame transcription = 3;
      MessageFrame message = 4;
    }
  }
  `;