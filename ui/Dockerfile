FROM node:20-alpine

WORKDIR /app

# Install deps before copying source so this layer is cached
COPY ui/package*.json ./
RUN npm ci --omit=dev

COPY ui/ .

EXPOSE 3000

CMD ["node", "server.js"]
