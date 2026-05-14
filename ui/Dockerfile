FROM node:20-slim
WORKDIR /app

# Install deps first for layer caching (context = repo root, so path is ui/)
COPY ui/package*.json ./
RUN npm ci --production

# Copy UI source into WORKDIR so server.js is at /app/server.js
COPY ui/ .

# picture_samples lives one level above server.js both locally (ui/../picture_samples)
# and here (/app/../picture_samples = /picture_samples), so path.join(__dirname, '..', 'picture_samples')
# resolves correctly without any env-var override.
COPY picture_samples/ /picture_samples/

EXPOSE 3000
CMD ["node", "server.js"]
