# app/web.py
import os
from flask import Flask, request, render_template
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, "templates")

app = Flask(__name__, template_folder=templates_dir)

FASTAPI_URL = "http://127.0.0.1:8000"  # L’URL où tourne FastAPI

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # Rendu de la page d’accueil

@app.route("/results", methods=["POST"])
def results():
    description = request.form["description"]
    try:
        # On appelle l’API FastAPI /recommend/
        response = requests.post(
            f"{FASTAPI_URL}/recommend/",
            json={"description": description}
        )
        response.raise_for_status()
        recommendations = response.json().get("recommendations", [])
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

    return render_template("results.html", recommendations=recommendations)

if __name__ == "__main__":
    # Lancement du serveur Flask sur le port 5000
    app.run(debug=True, port=5000)
