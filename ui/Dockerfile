FROM node:20-alpine

WORKDIR /app

# Install deps before copying source so this layer is cached
COPY ui/package*.json ./
RUN npm ci --omit=dev

COPY ui/ .

# picture_samples used by the Challenge game; placed alongside server.js so
# SAMPLES_DIR=/app/picture_samples resolves correctly inside the container
COPY picture_samples/ picture_samples/

EXPOSE 3000

CMD ["node", "server.js"]
