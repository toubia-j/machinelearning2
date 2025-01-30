# machinelearning2/Dockerfile

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code du projet dans /app
COPY . .

# Exposer les ports (5000 et 5001)
EXPOSE 5000
EXPOSE 5001

