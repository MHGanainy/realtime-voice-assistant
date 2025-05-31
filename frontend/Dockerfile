# Multi-stage build for React frontend
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json ./

# Install dependencies - use npm install for flexibility
RUN npm install

# Copy source code
COPY . .

# Build argument for WebSocket URL (pointing to your backend)
ARG VITE_WS_URL=wss://backend-va-production.up.railway.app
ENV VITE_WS_URL=$VITE_WS_URL

# Build the application
RUN npm run build

# Production stage using nginx
FROM nginx:alpine

# Copy built files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY default.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]