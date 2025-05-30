# Railway Frontend Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

# Build argument for WebSocket URL - Railway will provide this
ARG VITE_WS_URL
ENV VITE_WS_URL=$VITE_WS_URL

RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration for Railway
COPY railway.nginx.conf /etc/nginx/conf.d/default.conf

# Railway sets PORT env var, but nginx needs it in config
ENV PORT=80
EXPOSE $PORT

# Start nginx
CMD ["nginx", "-g", "daemon off;"]