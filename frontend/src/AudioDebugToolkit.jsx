// frontend/src/AudioDebugToolkit.jsx
import { useState, useRef, useEffect } from 'react';

export default function AudioDebugToolkit({ onClose }) {
  const [micPermission, setMicPermission] = useState('checking');
  const [selectedMic, setSelectedMic] = useState('');
  const [microphones, setMicrophones] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [isPlayingTestSound, setIsPlayingTestSound] = useState(false);
  const [browserCompatibility, setBrowserCompatibility] = useState({});
  const [audioFormats, setAudioFormats] = useState({});
  const [networkLatency, setNetworkLatency] = useState(null);
  const [websocketSupport, setWebsocketSupport] = useState(null);
  
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const micStreamRef = useRef(null);
  const animationFrameRef = useRef(null);
  const oscillatorRef = useRef(null);

  // Check browser compatibility on mount
  useEffect(() => {
    checkBrowserCompatibility();
    checkAudioFormats();
    checkWebSocketSupport();
    checkMicrophonePermission();
    return () => {
      stopMicrophoneTest();
    };
  }, []);

  const checkBrowserCompatibility = () => {
    const features = {
      getUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
      webAudio: typeof AudioContext !== 'undefined' || typeof webkitAudioContext !== 'undefined',
      webSockets: typeof WebSocket !== 'undefined',
      protobuf: true, // Assuming protobuf.js is loaded
      es6: true, // If the code is running, ES6 is supported
      audioWorklet: typeof AudioWorkletNode !== 'undefined',
      mediaRecorder: typeof MediaRecorder !== 'undefined',
      speechRecognition: 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window,
    };

    // Browser detection
    const userAgent = navigator.userAgent;
    let browserName = 'Unknown';
    let browserVersion = '';
    
    if (userAgent.indexOf('Chrome') > -1) {
      browserName = 'Chrome';
      browserVersion = userAgent.match(/Chrome\/(\d+)/)?.[1] || '';
    } else if (userAgent.indexOf('Safari') > -1 && userAgent.indexOf('Chrome') === -1) {
      browserName = 'Safari';
      browserVersion = userAgent.match(/Version\/(\d+)/)?.[1] || '';
    } else if (userAgent.indexOf('Firefox') > -1) {
      browserName = 'Firefox';
      browserVersion = userAgent.match(/Firefox\/(\d+)/)?.[1] || '';
    } else if (userAgent.indexOf('Edge') > -1) {
      browserName = 'Edge';
      browserVersion = userAgent.match(/Edge\/(\d+)/)?.[1] || '';
    }

    setBrowserCompatibility({
      ...features,
      browserName,
      browserVersion,
      userAgent: userAgent.substring(0, 100) + '...',
    });
  };

  const checkAudioFormats = () => {
    const audio = new Audio();
    const formats = {
      mp3: audio.canPlayType('audio/mpeg'),
      wav: audio.canPlayType('audio/wav'),
      ogg: audio.canPlayType('audio/ogg'),
      webm: audio.canPlayType('audio/webm'),
      aac: audio.canPlayType('audio/aac'),
      opus: audio.canPlayType('audio/ogg; codecs="opus"'),
    };
    setAudioFormats(formats);
  };

  const checkWebSocketSupport = async () => {
    try {
      const testWs = new WebSocket('ws://localhost:8765');
      const timeout = setTimeout(() => {
        testWs.close();
        setWebsocketSupport('timeout');
      }, 3000);

      testWs.onopen = () => {
        clearTimeout(timeout);
        setWebsocketSupport('connected');
        testWs.close();
        // Test latency
        testNetworkLatency();
      };

      testWs.onerror = () => {
        clearTimeout(timeout);
        setWebsocketSupport('error');
      };
    } catch (err) {
      setWebsocketSupport('error');
    }
  };

  const testNetworkLatency = async () => {
    const times = [];
    for (let i = 0; i < 5; i++) {
      const start = performance.now();
      try {
        await fetch('/');
        times.push(performance.now() - start);
      } catch (err) {
        // Ignore errors
      }
    }
    if (times.length > 0) {
      const avgLatency = times.reduce((a, b) => a + b, 0) / times.length;
      setNetworkLatency(Math.round(avgLatency));
    }
  };

  const checkMicrophonePermission = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const mics = devices.filter(device => device.kind === 'audioinput');
      setMicrophones(mics);
      
      if (mics.length > 0) {
        setSelectedMic(mics[0].deviceId);
      }

      // Check permission state
      if (navigator.permissions && navigator.permissions.query) {
        const result = await navigator.permissions.query({ name: 'microphone' });
        setMicPermission(result.state);
        result.addEventListener('change', () => {
          setMicPermission(result.state);
        });
      } else {
        // Try to get user media to check permission
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          stream.getTracks().forEach(track => track.stop());
          setMicPermission('granted');
        } catch (err) {
          setMicPermission('denied');
        }
      }
    } catch (err) {
      console.error('Error checking microphone:', err);
      setMicPermission('error');
    }
  };

  const startMicrophoneTest = async () => {
    try {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: selectedMic,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });
      
      micStreamRef.current = stream;
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      
      source.connect(analyserRef.current);
      
      setIsRecording(true);
      updateAudioLevel();
    } catch (err) {
      console.error('Error starting microphone test:', err);
      alert('Failed to access microphone: ' + err.message);
    }
  };

  const stopMicrophoneTest = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(track => track.stop());
      micStreamRef.current = null;
    }
    
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    setIsRecording(false);
    setAudioLevel(0);
  };

  const updateAudioLevel = () => {
    if (!analyserRef.current) return;
    
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(dataArray);
    
    const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
    setAudioLevel(average);
    
    animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
  };

  const playTestSound = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    setIsPlayingTestSound(true);
    
    // Create a simple tone
    oscillatorRef.current = audioContextRef.current.createOscillator();
    const gainNode = audioContextRef.current.createGain();
    
    oscillatorRef.current.connect(gainNode);
    gainNode.connect(audioContextRef.current.destination);
    
    oscillatorRef.current.frequency.value = 440; // A4 note
    gainNode.gain.value = 0.3;
    
    oscillatorRef.current.start();
    
    // Stop after 1 second
    setTimeout(() => {
      oscillatorRef.current.stop();
      setIsPlayingTestSound(false);
    }, 1000);
  };

  const exportDebugInfo = () => {
    const debugInfo = {
      timestamp: new Date().toISOString(),
      browserCompatibility,
      audioFormats,
      microphones: microphones.map(mic => ({
        label: mic.label,
        deviceId: mic.deviceId ? 'present' : 'missing',
      })),
      micPermission,
      websocketSupport,
      networkLatency,
      userAgent: navigator.userAgent,
      screen: {
        width: window.screen.width,
        height: window.screen.height,
        pixelRatio: window.devicePixelRatio,
      },
    };
    
    const blob = new Blob([JSON.stringify(debugInfo, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `audio-debug-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case true:
      case 'granted':
      case 'connected':
      case 'probably':
        return '✅';
      case false:
      case 'denied':
      case 'error':
      case '':
        return '❌';
      case 'prompt':
      case 'timeout':
      case 'maybe':
        return '⚠️';
      default:
        return '❓';
    }
  };

  return (
    <div style={styles.overlay}>
      <div style={styles.modal}>
        <div style={styles.header}>
          <h2 style={styles.title}>🛠️ Audio Debug Toolkit</h2>
          <button style={styles.closeButton} onClick={onClose}>×</button>
        </div>
        
        <div style={styles.content}>
          {/* Browser Compatibility */}
          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>Browser Compatibility</h3>
            <div style={styles.infoGrid}>
              <div style={styles.infoRow}>
                <span>Browser:</span>
                <span>{browserCompatibility.browserName} {browserCompatibility.browserVersion}</span>
              </div>
              {Object.entries(browserCompatibility).map(([key, value]) => {
                if (key === 'browserName' || key === 'browserVersion' || key === 'userAgent') return null;
                return (
                  <div key={key} style={styles.infoRow}>
                    <span>{key}:</span>
                    <span>{getStatusIcon(value)} {value ? 'Supported' : 'Not Supported'}</span>
                  </div>
                );
              })}
            </div>
          </section>

          {/* Audio Formats */}
          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>Audio Format Support</h3>
            <div style={styles.infoGrid}>
              {Object.entries(audioFormats).map(([format, support]) => (
                <div key={format} style={styles.infoRow}>
                  <span>{format.toUpperCase()}:</span>
                  <span>{getStatusIcon(support)} {support || 'Not Supported'}</span>
                </div>
              ))}
            </div>
          </section>

          {/* Network & WebSocket */}
          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>Network & WebSocket</h3>
            <div style={styles.infoGrid}>
              <div style={styles.infoRow}>
                <span>WebSocket Status:</span>
                <span>{getStatusIcon(websocketSupport)} {websocketSupport || 'Checking...'}</span>
              </div>
              <div style={styles.infoRow}>
                <span>Network Latency:</span>
                <span>{networkLatency ? `${networkLatency}ms` : 'Testing...'}</span>
              </div>
            </div>
          </section>

          {/* Microphone Test */}
          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>Microphone Test</h3>
            <div style={styles.infoGrid}>
              <div style={styles.infoRow}>
                <span>Permission:</span>
                <span>{getStatusIcon(micPermission)} {micPermission}</span>
              </div>
              <div style={styles.infoRow}>
                <span>Devices Found:</span>
                <span>{microphones.length}</span>
              </div>
            </div>
            
            {microphones.length > 0 && (
              <select 
                style={styles.select}
                value={selectedMic}
                onChange={(e) => setSelectedMic(e.target.value)}
                disabled={isRecording}
              >
                {microphones.map(mic => (
                  <option key={mic.deviceId} value={mic.deviceId}>
                    {mic.label || `Microphone ${mic.deviceId.substring(0, 8)}`}
                  </option>
                ))}
              </select>
            )}
            
            <button
              style={{...styles.button, ...(isRecording ? styles.recordingButton : {})}}
              onClick={isRecording ? stopMicrophoneTest : startMicrophoneTest}
              disabled={micPermission !== 'granted'}
            >
              {isRecording ? '⏹ Stop Test' : '🎤 Start Test'}
            </button>
            
            {isRecording && (
              <div style={styles.levelMeter}>
                <div style={styles.levelLabel}>Audio Level:</div>
                <div style={styles.levelBar}>
                  <div 
                    style={{
                      ...styles.levelFill,
                      width: `${(audioLevel / 255) * 100}%`,
                      backgroundColor: audioLevel > 200 ? '#ef4444' : audioLevel > 100 ? '#f59e0b' : '#22c55e',
                    }}
                  />
                </div>
              </div>
            )}
          </section>

          {/* Audio Playback Test */}
          <section style={styles.section}>
            <h3 style={styles.sectionTitle}>Audio Playback Test</h3>
            <button
              style={styles.button}
              onClick={playTestSound}
              disabled={isPlayingTestSound}
            >
              {isPlayingTestSound ? '🔊 Playing...' : '🔈 Play Test Sound'}
            </button>
            <p style={styles.helpText}>
              You should hear a 440Hz tone (A4 note) for 1 second
            </p>
          </section>

          {/* Export Debug Info */}
          <section style={styles.section}>
            <button style={styles.exportButton} onClick={exportDebugInfo}>
              📥 Export Debug Info
            </button>
          </section>
        </div>
      </div>
    </div>
  );
}

const styles = {
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  modal: {
    backgroundColor: '#1a1a1a',
    borderRadius: '8px',
    width: '90%',
    maxWidth: '600px',
    maxHeight: '90vh',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    border: '1px solid #333',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '1rem 1.5rem',
    borderBottom: '1px solid #333',
  },
  title: {
    margin: 0,
    fontSize: '1.5rem',
    color: '#fff',
  },
  closeButton: {
    background: 'transparent',
    border: 'none',
    color: '#888',
    fontSize: '2rem',
    cursor: 'pointer',
    padding: 0,
    width: '2rem',
    height: '2rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  content: {
    padding: '1.5rem',
    overflowY: 'auto',
    flex: 1,
  },
  section: {
    marginBottom: '2rem',
  },
  sectionTitle: {
    fontSize: '1.125rem',
    marginBottom: '1rem',
    color: '#fff',
  },
  infoGrid: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  infoRow: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '0.5rem',
    backgroundColor: '#222',
    borderRadius: '4px',
    fontSize: '0.875rem',
  },
  select: {
    width: '100%',
    padding: '0.5rem',
    marginTop: '0.5rem',
    backgroundColor: '#222',
    border: '1px solid #444',
    borderRadius: '4px',
    color: '#fff',
    fontSize: '0.875rem',
  },
  button: {
    width: '100%',
    padding: '0.75rem',
    marginTop: '0.75rem',
    backgroundColor: '#4a9eff',
    border: 'none',
    borderRadius: '4px',
    color: '#fff',
    fontSize: '1rem',
    cursor: 'pointer',
    transition: 'background 0.2s',
  },
  recordingButton: {
    backgroundColor: '#dc2626',
  },
  exportButton: {
    width: '100%',
    padding: '0.75rem',
    backgroundColor: '#22c55e',
    border: 'none',
    borderRadius: '4px',
    color: '#fff',
    fontSize: '1rem',
    cursor: 'pointer',
  },
  levelMeter: {
    marginTop: '1rem',
  },
  levelLabel: {
    fontSize: '0.875rem',
    color: '#888',
    marginBottom: '0.25rem',
  },
  levelBar: {
    width: '100%',
    height: '20px',
    backgroundColor: '#222',
    borderRadius: '4px',
    overflow: 'hidden',
  },
  levelFill: {
    height: '100%',
    transition: 'width 0.1s',
  },
  helpText: {
    fontSize: '0.75rem',
    color: '#888',
    marginTop: '0.5rem',
    textAlign: 'center',
  },
};