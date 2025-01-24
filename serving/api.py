from fastapi import FastAPI
from scripts.train_model import train_model  # Assurez-vous que 'train_model' est bien importé
from sklearn.model_selection import train_test_split
from scripts.train_model import load_data, preprocess_data, normalize_data
import os

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prévision de modèle!"}
# Route pour entraîner le modèle


@app.post("/train_model/")
async def train_model():
    # Logique pour entraîner ton modèle ici, en appelant la fonction train_model définie dans ton script
    # Exemple : 
    # model = train_model(X_train, y_train)
    return {"message": "Modèle entraîné avec succès!"}