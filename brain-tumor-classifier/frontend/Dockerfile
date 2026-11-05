# =============================================================================
# Image du site (React + Vite) servi par nginx.
#   docker build -t brain-mri-front ./frontend --build-arg VITE_API_URL=http://localhost:8000
# L'adresse de l'API est intégrée au site à la construction (VITE_API_URL).
# =============================================================================

# ---- Étape 1 : construction du site ----------------------------------------
FROM node:24-alpine AS build
WORKDIR /app

# Dépendances d'abord (couche mise en cache tant que package-lock.json ne change pas)
COPY package.json package-lock.json ./
RUN npm ci

# Adresse de l'API vue par le navigateur
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}

COPY . .
RUN npm run build

# ---- Étape 2 : nginx qui sert dist/ -----------------------------------------
FROM nginx:1.30-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
