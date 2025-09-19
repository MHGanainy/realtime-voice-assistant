import { CONFIG } from '../config';

export const startEventPing = (refs) => {
  const pingInterval = setInterval(() => {
    if (refs.current.eventWebsocket?.readyState === WebSocket.OPEN) {
      refs.current.eventWebsocket.send(JSON.stringify({
        type: 'ping',
        timestamp: new Date().toISOString()
      }));
    } else {
      clearInterval(pingInterval);
    }
  }, 30000);
  
  refs.current.eventPingInterval = pingInterval;
};

export const connectEventWebSocket = (sessionId, refs, updateState, addEventLog, handleEventMessage) => {
  if (!sessionId) return;

  const eventWs = new WebSocket(`${CONFIG.WS_BASE_URL}/ws/events?session_id=${sessionId}`);
  
  eventWs.onopen = () => {
    updateState({ eventWsConnected: true });
    addEventLog('system', 'Event stream connected', { type: 'info' });
    startEventPing(refs);
  };

  eventWs.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleEventMessage(data);
    } catch (error) {
      addEventLog('system', 'Failed to parse event message', { 
        type: 'error',
        data: { error: error.message }
      });
    }
  };

  eventWs.onerror = (error) => {
    addEventLog('system', 'Event stream error', { type: 'error' });
  };

  eventWs.onclose = (event) => {
    updateState({ eventWsConnected: false });
    addEventLog('system', 'Event stream disconnected', { 
      type: 'warning',
      data: { code: event.code, reason: event.reason }
    });
    
    if (refs.current.eventPingInterval) {
      clearInterval(refs.current.eventPingInterval);
      refs.current.eventPingInterval = null;
    }
    
    refs.current.eventWebsocket = null;
  };

  refs.current.eventWebsocket = eventWs;
};

export const handleEventMessage = (message, addEventLog, refs) => {
  const { type, event: eventName, data, timestamp } = message;

  switch (type) {
    case 'event':
      if (eventName) {
        addEventLog(eventName, data || {}, { timestamp });
      }
      break;

    case 'welcome':
      addEventLog('system:welcome', data || {}, { type: 'success' });
      
      if (refs.current.eventWebsocket?.readyState === WebSocket.OPEN) {
        refs.current.eventWebsocket.send(JSON.stringify({
          type: 'subscribe',
          subscription: {
            patterns: ["*"],
            include_historical: true
          }
        }));
      }
      break;

    case 'historical_start':
      addEventLog('system:historical:start', { count: data?.count || 0 }, { type: 'info' });
      break;

    case 'historical_end':
      addEventLog('system:historical:end', { count: data?.count || 0 }, { type: 'success' });
      break;

    case 'subscription_updated':
      addEventLog('system:subscription:updated', data || {}, { type: 'info' });
      break;

    case 'stats':
      addEventLog('system:stats', data || {}, { type: 'info' });
      break;

    case 'heartbeat':
    case 'pong':
      break;

    default:
      addEventLog('system:unknown', { type, ...message }, { type: 'warning' });
  }
};

export const createConversationWebSocket = (sessionId, params, refs, updateState, handleMessage, addEventLog) => {
  const queryParams = new URLSearchParams({
    session_id: sessionId,
    stt_provider: params.sttProvider || 'deepinfra',
    stt_model: params.sttModel || 'nova-2',
    llm_provider: params.llmProvider || 'openai',
    llm_model: params.llmModel || 'gpt-3.5-turbo',
    tts_provider: params.ttsProvider || 'deepinfra',
    tts_model: params.ttsModel || 'hexgrad/Kokoro-82M',
    tts_voice: params.ttsVoice || 'af_bella',
    system_prompt: params.systemPrompt,
    enable_interruptions: (params.enableInterruptions ?? true).toString(),
    vad_enabled: 'true',
    enable_processors: params.enableProcessors.toString()
  });
  
  // Add JWT token if provided (authentication)
  if (params.jwtToken) {
    queryParams.append('token', params.jwtToken);
  }
  
  // Add correlation token if provided (transcript tracking)
  if (params.correlationToken) {
    queryParams.append('correlation_token', params.correlationToken);
  }
  
  // Log warning if missing tokens
  if (!params.jwtToken || !params.correlationToken) {
    console.warn('Missing required tokens:', {
      hasJWT: !!params.jwtToken,
      hasCorrelation: !!params.correlationToken
    });
  }
  
  const ws = new WebSocket(`${CONFIG.WS_BASE_URL}/ws/conversation?${queryParams}`);
  ws.binaryType = 'arraybuffer';
  
  ws.onopen = () => {
    updateState({ 
      isConnected: true, 
      isRecording: true, 
      status: `Connected! Speak naturally... (Processors ${params.enableProcessors ? 'enabled' : 'disabled'})` 
    });
    
    // Log the connection settings
    addEventLog('system:connection:settings', {
      processors_enabled: params.enableProcessors,
      interruptions_enabled: params.enableInterruptions ?? true,
      correlation_token: params.correlationToken,
      has_jwt_token: !!params.jwtToken,
      stt_provider: params.sttProvider,
      stt_model: params.sttModel,
      llm_provider: params.llmProvider,
      llm_model: params.llmModel,
      tts_provider: params.ttsProvider,
      tts_model: params.ttsModel,
      tts_voice: params.ttsVoice,
      system_prompt_preview: params.systemPrompt.substring(0, 50) + '...',
      message: params.enableProcessors 
        ? 'Conversation tracking and metrics enabled' 
        : 'Low-latency mode - no tracking or metrics'
    }, { type: 'info' });
  };
  
  ws.onmessage = handleMessage;
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    updateState({ status: 'Connection error' });
  };
  ws.onclose = (event) => {
    updateState({ 
      isConnected: false, 
      isRecording: false, 
      status: `Disconnected: ${event.reason || 'Connection closed'}` 
    });
  };
  
  return ws;
};

export const requestEventStats = (refs) => {
  if (refs.current.eventWebsocket?.readyState === WebSocket.OPEN) {
    refs.current.eventWebsocket.send(JSON.stringify({
      type: 'get_stats'
    }));
  }
};