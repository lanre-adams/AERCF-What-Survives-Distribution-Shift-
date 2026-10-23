# ── Stage 1: Build React app ─────────────────────────────────────────────────
FROM node:18-alpine AS builder

WORKDIR /app

# Install dependencies first (layer-cache friendly)
COPY package.json ./
# react-scripts 5 requires ajv@^8 on Node 18; install it explicitly first
RUN npm install --legacy-peer-deps ajv@^8 && \
    npm install --legacy-peer-deps

# Copy source and build
COPY public/ ./public/
COPY src/ ./src/

# Point the API at the nginx reverse proxy path
ENV REACT_APP_API_URL=/api

RUN npm run build

# ── Stage 2: Serve with nginx ─────────────────────────────────────────────────
FROM nginx:1.27-alpine

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy the React build
COPY --from=builder /app/build /usr/share/nginx/html

# nginx config is mounted via docker-compose; expose port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
