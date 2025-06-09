// TTS Provider Configurations
export const TTS_PROVIDERS = {
  openai_mini: {
    name: 'OpenAI',
    provider: 'openai',
    model: 'gpt-4o-mini-tts',
    voice: 'nova',
    displayName: 'OpenAI - GPT-4o Mini TTS (Nova)',
    description: 'Faster generation, improved prosody, recommended for most use cases'
  },
  openai_standard: {
    name: 'OpenAI',
    provider: 'openai',
    model: 'tts-1',
    voice: 'alloy',
    displayName: 'OpenAI - TTS-1 (Alloy)',
    description: 'Original TTS model with standard quality speech'
  },
  deepgram: {
    name: 'Deepgram',
    provider: 'deepgram',
    model: 'aura-2',
    voice: 'aura-2-helena-en',
    displayName: 'Deepgram - Aura Helena',
    description: 'Natural sounding Aura v2 Helena voice'
  },
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
  },

  speechify: {
    name: 'Speechify',
    provider: 'speechify',
    model: 'simba-english',
    voice: 'kristy',
    displayName: 'Speechify - Simba English (Kristy)',
    description: 'Natural English voice using Simba English model'
  },
  rime: {
    name: 'Rime',
    provider: 'rime',
    model: 'mistv2',
    voice: 'cove',
    displayName: 'Rime - Mistv2 (cove)',
    description: 'Natural English voice using Rime Mistv2 model'
  },
  riva: {
    name: 'Riva',
    provider: 'riva',
    model: 'radtts-hifigan-tts',
    voice: "English-US-RadTTS.Female-1",
    displayName: 'Riva - Radtts Hifigan TTS',
    description: 'Riva Radtts Hifigan TTS model'
  },
  fastpitch: {
    name: 'Riva',
    provider: 'riva',
    model: 'fastpitch-hifigan-tts',
    voice: 'English-US.Female-1',
    displayName: 'Riva - Fastpitch Hifigan TTS',
    description: 'Riva Fastpitch Hifigan TTS model'
  },
  groq: {
    name: 'Groq',
    provider: 'groq',
    model: 'playai-tts',
    voice: 'Celeste-PlayAI',
    displayName: 'Groq - PlayAI TTS',
    description: 'Groq PlayAI TTS model'
  }
};

// Default TTS provider
export const DEFAULT_TTS_PROVIDER = 'openai_mini';

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
},
together: {
  name: 'Together',
  provider: 'together',
  model: 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
  displayName: 'Together - Llama 3.3 70B Instruct Turbo',
  description: 'Powerful Together Llama 3.3 70B Instruct Turbo model'
},
groq: {
  name: 'Groq',
  provider: 'groq',
  model: 'llama-3.3-70b-versatile',
  displayName: 'Groq - Llama 3.3 70B Versatile',
  description: 'Powerful Groq Llama 3.3 70B Versatile model'
},
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
deepgram: {
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
},
riva: {
  name: 'Riva',
  provider: 'riva',
  model: 'parakeet-ctc-1.1b-asr',
  displayName: 'Riva - Parakeet CTC 1.1B ASR',
  description: 'Riva Parakeet CTC 1.1B ASR model'
},
groq: {
  name: 'Groq',
  provider: 'groq',
  model: 'distil-whisper-large-v3-en',
  displayName: 'Groq - Distil Whisper Large V3 EN',
  description: 'Groq Distil Whisper Large V3 EN model'
}
};

// Default STT provider
export const DEFAULT_STT_PROVIDER = 'deepgram';

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