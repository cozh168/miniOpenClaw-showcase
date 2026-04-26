FROM node:20-alpine

ENV NEXT_TELEMETRY_DISABLED=1

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend /app
RUN npm run build

EXPOSE 7788

CMD ["npm", "run", "start", "--", "--hostname", "0.0.0.0"]
