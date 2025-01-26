# webapp/Dockerfile

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose le port Flask (5000 par défaut)
EXPOSE 5000

CMD ["python", "web.py"]
