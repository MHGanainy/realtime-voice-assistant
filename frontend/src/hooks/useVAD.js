// src/hooks/useVAD.js
import { useRef, useCallback, useEffect } from 'react';
import { VADProcessor } from '../services/vadProcessor';

export const useVAD = ({
  sampleRate = 16000,
  onSpeechStart,
  onSpeechEnd,
  onVoiceActivity,
  onProcessedAudio,
  mode = 2
}) => {
  const vadRef = useRef(null);
  const isActiveRef = useRef(false);
  
  useEffect(() => {
    vadRef.current = new VADProcessor({
      sampleRate,
      mode,
      onSpeechStart: (data) => {
        console.log('VAD: Speech started');
        onSpeechStart?.(data);
      },
      onSpeechEnd: (data) => {
        console.log('VAD: Speech ended');
        onSpeechEnd?.(data);
      },
      onVoiceActivity: (data) => {
        onVoiceActivity?.(data);
      },
      onProcessedAudio: (data) => {
        onProcessedAudio?.(data);
      }
    });
    
    return () => {
      vadRef.current?.reset();
    };
  }, [sampleRate, mode, onSpeechStart, onSpeechEnd, onVoiceActivity, onProcessedAudio]);
  
  const processAudioData = useCallback((audioData) => {
    if (vadRef.current && isActiveRef.current) {
      vadRef.current.processAudio(audioData);
    }
  }, []);
  
  const startVAD = useCallback(() => {
    isActiveRef.current = true;
    vadRef.current?.reset();
  }, []);
  
  const stopVAD = useCallback(() => {
    isActiveRef.current = false;
    vadRef.current?.reset();
  }, []);
  
  const setMode = useCallback((newMode) => {
    if (vadRef.current) {
      vadRef.current.options.mode = newMode;
    }
  }, []);
  
  return {
    processAudioData,
    startVAD,
    stopVAD,
    setMode,
    isActive: isActiveRef.current
  };
};