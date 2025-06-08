// TTS Provider Configurations
export const TTS_PROVIDERS = {
    deepinfra: {
      name: 'DeepInfra',
      provider: 'deepinfra',
      model: 'hexgrad/Kokoro-82M',
      voice: 'af_bella',
      displayName: 'DeepInfra - Kokoro (af_bella)',
      description: 'Using Kokoro-82M model with af_bella voice'
    },
    elevenlabs: {
      name: 'ElevenLabs',
      provider: 'elevenlabs',
      model: 'eleven_flash_v2_5',
      voice: '21m00Tcm4TlvDq8ikWAM',
      displayName: 'ElevenLabs - Flash v2.5',
      description: 'Using eleven_flash_v2_5 model'
    }
  };
  
  // Default TTS provider
  export const DEFAULT_TTS_PROVIDER = 'deepinfra';
  
  // Helper function to get TTS config by provider key
  export const getTTSConfig = (provider) => {
    return TTS_PROVIDERS[provider] || TTS_PROVIDERS[DEFAULT_TTS_PROVIDER];
  };
  
  // Get array of providers for dropdown
  export const getTTSProviderOptions = () => {
    return Object.entries(TTS_PROVIDERS).map(([key, config]) => ({
      value: key,
      label: config.displayName,
      ...config
    }));
  };


export const LLM_PROVIDERS = {
  openai: {
    name: 'OpenAI',
    provider: 'openai',
    model: 'gpt-3.5-turbo',
    displayName: 'OpenAI - GPT-3.5 Turbo',
    description: 'Fast and efficient GPT-3.5 Turbo model'
  },
  deepinfra: {
    name: 'DeepInfra',
    provider: 'deepinfra',
    model: 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    displayName: 'DeepInfra - Llama 3.1 70B',
    description: 'Powerful Meta Llama 3.1 70B Instruct model'
  }
};

// Default LLM provider
export const DEFAULT_LLM_PROVIDER = 'openai';

// Helper function to get LLM config by provider key
export const getLLMConfig = (provider) => {
  return LLM_PROVIDERS[provider] || LLM_PROVIDERS[DEFAULT_LLM_PROVIDER];
};

// Get array of providers for dropdown
export const getLLMProviderOptions = () => {
  return Object.entries(LLM_PROVIDERS).map(([key, config]) => ({
    value: key,
    label: config.displayName,
    ...config
  }));
};



// STT Provider Configurations
export const STT_PROVIDERS = {
  deepinfra: {
    name: 'Deepgram',
    provider: 'deepgram',
    model: 'nova-2',
    displayName: 'Deepgram - Nova 2',
    description: 'Fast and accurate Nova 2 speech recognition'
  },
  openai: {
    name: 'OpenAI',
    provider: 'openai',
    model: 'gpt-4o-transcribe',
    displayName: 'OpenAI - GPT-4 Transcribe',
    description: 'Advanced GPT-4 based transcription'
  }
};

// Default STT provider
export const DEFAULT_STT_PROVIDER = 'deepinfra';

// Helper function to get STT config by provider key
export const getSTTConfig = (provider) => {
  return STT_PROVIDERS[provider] || STT_PROVIDERS[DEFAULT_STT_PROVIDER];
};

// Get array of providers for dropdown
export const getSTTProviderOptions = () => {
  return Object.entries(STT_PROVIDERS).map(([key, config]) => ({
    value: key,
    label: config.displayName,
    ...config
  }));
};