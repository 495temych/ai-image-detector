FROM node:20-slim
WORKDIR /app

# Install deps first for layer caching (context = repo root, so path is ui/)
COPY ui/package*.json ./
RUN npm ci --production

# Copy UI source into WORKDIR so server.js is at /app/server.js
COPY ui/ .

# image_samples lives one level above server.js both locally (ui/../image_samples)
# and here (/app/../image_samples = /image_samples), so path.join(__dirname, '..', 'image_samples')
# resolves correctly without any env-var override.
COPY image_samples/ /image_samples/

EXPOSE 3000
CMD ["node", "server.js"]
