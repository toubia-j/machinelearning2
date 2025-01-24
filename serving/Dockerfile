FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY ../requirements.txt /app/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le contenu de "serving" et "scripts" dans le conteneur
COPY . .

# Spécifier la commande à exécuter au démarrage du conteneur
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
