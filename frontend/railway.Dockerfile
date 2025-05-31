# Multi-stage build for React frontend
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build argument for WebSocket URL
ARG VITE_WS_URL=wss://backend-va-production.up.railway.app/ws
ENV VITE_WS_URL=$VITE_WS_URL

# Build the application
RUN npm run build

# Production stage using nginx
FROM nginx:alpine

# Install gettext for envsubst
RUN apk add --no-cache gettext

# Copy built files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration template
COPY default.conf /etc/nginx/templates/default.conf.template

# Create a startup script to handle PORT variable
RUN echo '#!/bin/sh' > /docker-entrypoint.sh && \
    echo 'envsubst '\''$PORT'\'' < /etc/nginx/templates/default.conf.template > /etc/nginx/conf.d/default.conf' >> /docker-entrypoint.sh && \
    echo 'exec nginx -g "daemon off;"' >> /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh

# Expose port (Railway will override this)
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:${PORT:-80}/ || exit 1

# Start nginx with PORT substitution
ENTRYPOINT ["/docker-entrypoint.sh"]