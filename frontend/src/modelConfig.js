// TTS Provider Configurations
export const TTS_PROVIDERS = {
  // Inworld TTS Voices with specific speeds
  inworld_craig: {
    name: 'Inworld',
    provider: 'inworld',
    model: 'inworld-tts-1',
    voice: 'Craig',
    speed: 1.2,
    displayName: 'Inworld - Craig (Male, Speed 1.2)',
    description: 'Male voice with faster speech rate for dynamic conversations'
  },
  inworld_edward: {
    name: 'Inworld',
    provider: 'inworld',
    model: 'inworld-tts-1',
    voice: 'Edward',
    speed: 1.0,
    displayName: 'Inworld - Edward (Male, Speed 1.0)',
    description: 'Male voice with normal speech rate, clear and professional'
  },
  inworld_olivia: {
    name: 'Inworld',
    provider: 'inworld',
    model: 'inworld-tts-1',
    voice: 'Olivia',
    speed: 1.0,
    displayName: 'Inworld - Olivia (Female, Speed 1.0)',
    description: 'Female voice with normal speech rate, warm and friendly'
  },
  inworld_wendy: {
    name: 'Inworld',
    provider: 'inworld',
    model: 'inworld-tts-1',
    voice: 'Wendy',
    speed: 1.2,
    displayName: 'Inworld - Wendy (Female, Speed 1.2)',
    description: 'Female voice with faster speech rate, energetic and engaging'
  },
  inworld_priya: {
    name: 'Inworld',
    provider: 'inworld',
    model: 'inworld-tts-1',
    voice: 'Priya',
    speed: 1.0,
    displayName: 'Inworld - Priya (Female, Asian accent, Speed 1.0)',
    description: 'Female voice with Asian accent, normal speech rate'
  },
  
  // Existing OpenAI voices
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
  
  // Existing Google voices
  google_chirp_hd_charon: {
    name: 'Google Cloud',
    provider: 'google',
    model: 'chirp-hd',
    voice: 'en-US-Chirp3-HD-Charon',
    displayName: 'Google - Chirp HD Charon (Default)',
    description: 'Low-latency streaming voice (~$16/million chars)'
  },
  google_standard_female_a: {
    name: 'Google Cloud',
    provider: 'google',
    model: 'standard',
    voice: 'en-US-Standard-A',
    displayName: 'Google - Standard A',
    description: 'Cost-effective standard voice ($4/million chars)'
  },
  google_standard_male_b: {
    name: 'Google Cloud',
    provider: 'google',
    model: 'standard',
    voice: 'en-US-Standard-B',
    displayName: 'Google - Standard B',
    description: 'Cost-effective standard voice ($4/million chars)'
  },
  google_wavenet_female_a: {
    name: 'Google Cloud',
    provider: 'google',
    model: 'wavenet',
    voice: 'en-US-Wavenet-A',
    displayName: 'Google - WaveNet A',
    description: 'Premium neural voice with natural inflection ($16/million chars)'
  },
  google_wavenet_male_b: {
    name: 'Google Cloud',
    provider: 'google',
    model: 'wavenet',
    voice: 'en-US-Wavenet-B',
    displayName: 'Google - WaveNet B',
    description: 'Premium neural voice with natural inflection ($16/million chars)'
  },
  google_neural2_female_a: {
    name: 'Google Cloud',
    provider: 'google',
    model: 'neural2',
    voice: 'en-US-Neural2-A',
    displayName: 'Google - Neural2 A',
    description: 'Latest neural technology for high-quality speech ($16/million chars)'
  },
};

// Default TTS provider - change to Inworld Edward as default
export const DEFAULT_TTS_PROVIDER = 'inworld_edward';

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

// LLM Provider Configurations (keeping existing)
export const LLM_PROVIDERS = {
  openai: {
    name: 'OpenAI',
    provider: 'openai',
    model: 'gpt-4o-mini',
    displayName: 'OpenAI - GPT-4o Mini',
    description: 'Fast and efficient GPT-4o Mini model'
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
export const DEFAULT_LLM_PROVIDER = 'groq';

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

// STT Provider Configurations (keeping existing)
export const STT_PROVIDERS = {
  deepgram: {
    name: 'Deepgram',
    provider: 'deepgram',
    model: 'nova-2',
    displayName: 'Deepgram - Nova 2',
    description: 'Fast and accurate Nova 2 speech recognition'
  },
  assemblyai: {
    name: 'AssemblyAI',
    provider: 'assemblyai',
    model: 'universal-streaming',
    displayName: 'AssemblyAI - Universal Streaming',
    description: 'High-accuracy streaming transcription with low latency'
  }
};

// Default STT provider
export const DEFAULT_STT_PROVIDER = 'assemblyai';

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