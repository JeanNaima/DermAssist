FROM node:20-alpine

WORKDIR /app

COPY package*.json ./

RUN npm ci --omit=dev

COPY config ./config
COPY controllers ./controllers
COPY models ./models
COPY routes ./routes
COPY services ./services
COPY index.js ./

EXPOSE 3000

CMD ["node", "index.js"]