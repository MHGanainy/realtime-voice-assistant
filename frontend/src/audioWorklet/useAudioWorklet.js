// src/audioWorklet/useAudioWorklet.js
import { useRef, useCallback, useEffect } from 'react';

export const useAudioWorklet = ({ 
  sampleRate = 16000,
  outputSampleRate = 24000,
  onAudioData
}) => {
  const audioContextRef = useRef(null);
  const micNodeRef = useRef(null);
  const speakerNodeRef = useRef(null);
  const streamRef = useRef(null);
  const isRecordingRef = useRef(false);

  const initializeAudioWorklet = useCallback(async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        latencyHint: 'interactive',
        sampleRate: outputSampleRate,
      });

      // Load worklet modules from public directory
      await audioContextRef.current.audioWorklet.addModule('/audioWorklet/micProcessor.js');
      await audioContextRef.current.audioWorklet.addModule('/audioWorklet/speakerProcessor.js');
    }
  }, [outputSampleRate]);

  const startRecording = useCallback(async () => {
    try {
      await initializeAudioWorklet();
      
      // Get microphone stream
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: sampleRate,
          channelCount: 1,
          autoGainControl: true,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      
      streamRef.current = stream;
      
      // Create nodes
      const source = audioContextRef.current.createMediaStreamSource(stream);
      micNodeRef.current = new AudioWorkletNode(audioContextRef.current, 'mic-processor');
      speakerNodeRef.current = new AudioWorkletNode(audioContextRef.current, 'speaker-processor');
      
      // Handle mic data
      micNodeRef.current.port.onmessage = (event) => {
        if (event.data.type === 'audio' && onAudioData) {
          onAudioData(event.data.data);
        }
      };
      
      // Connect nodes
      source.connect(micNodeRef.current);
      speakerNodeRef.current.connect(audioContextRef.current.destination);
      
      isRecordingRef.current = true;
      
      return {
        audioContext: audioContextRef.current,
        speakerNode: speakerNodeRef.current
      };
    } catch (error) {
      console.error('Failed to start recording:', error);
      throw error;
    }
  }, [sampleRate, onAudioData, initializeAudioWorklet]);

  const stopRecording = useCallback(() => {
    if (micNodeRef.current) {
      micNodeRef.current.disconnect();
      micNodeRef.current = null;
    }
    
    if (speakerNodeRef.current) {
      speakerNodeRef.current.port.postMessage({ type: 'clear' });
      speakerNodeRef.current.disconnect();
      speakerNodeRef.current = null;
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    isRecordingRef.current = false;
  }, []);

  const queueAudioForPlayback = useCallback((audioData) => {
    if (speakerNodeRef.current && audioContextRef.current) {
      // Decode audio data and send to speaker worklet
      audioContextRef.current.decodeAudioData(
        audioData.buffer.slice(0),
        (buffer) => {
          const channelData = buffer.getChannelData(0);
          speakerNodeRef.current.port.postMessage({
            type: 'audio',
            buffer: channelData
          });
        },
        (error) => {
          console.error('Error decoding audio:', error);
        }
      );
    }
  }, []);

  useEffect(() => {
    return () => {
      stopRecording();
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, [stopRecording]);

  return {
    startRecording,
    stopRecording,
    queueAudioForPlayback,
    isRecording: isRecordingRef.current
  };
};