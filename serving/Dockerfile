# serving/Dockerfile

# 1. On part d'une image Python
FROM python:3.9-slim

# 2. On fixe le répertoire de travail
WORKDIR /app

# 3. Copier le requirements.txt et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier le code (api.py, model.py, etc.)
COPY . .

# 5. Exposer le port
EXPOSE 8000

# 6. Lancer l'API avec Uvicorn (FastAPI) ou autre
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
