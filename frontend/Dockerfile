# frontend/Dockerfile
FROM node:18 AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Çıktıyı tek bir klasöre topla (build veya dist hangisi varsa)
RUN if [ -d "dist" ]; then mv dist web_build; \
    elif [ -d "build" ]; then mv build web_build; \
    else echo "No build output found (dist/ or build/)"; ls -la; exit 1; fi

FROM nginx:alpine
COPY --from=build /app/web_build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
