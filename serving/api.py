
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests

# On importe la logique ML depuis model.py
from .model import recommender_function

# Créer l’application FastAPI
api_app = FastAPI()

# Modèle Pydantic pour la requête
class BookDescription(BaseModel):
    description: str

@api_app.get("/")
def root():
    return {"message": "Bienvenue sur l’API de recommandation"}

@api_app.post("/recommend/")
async def recommend_books(book: BookDescription) -> dict:
    try:
        # Appel à la fonction de recommandation (définie dans model.py)
        results = recommender_function(book.description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"recommendations": results}
