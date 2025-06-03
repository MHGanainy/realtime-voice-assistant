import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
  },
  esbuild: {
    jsxInject: `import React from 'react'`
  },
  // Ensure AudioWorklet files are properly served
  publicDir: 'public',
  build: {
    rollupOptions: {
      output: {
        // Ensure capture-processor.js is copied to output
        assetFileNames: (assetInfo) => {
          if (assetInfo.name === 'capture-processor.js') {
            return 'audio/[name][extname]';
          }
          return 'assets/[name]-[hash][extname]';
        }
      }
    }
  },
  // Handle AudioWorklet files specially in dev mode
  optimizeDeps: {
    exclude: ['capture-processor.js']
  }
})