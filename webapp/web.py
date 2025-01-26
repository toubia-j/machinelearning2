import os
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/results", methods=["POST"])
def results():
    description = request.form.get("description", "")
    try:
        # On poste la description à l'endpoint /recommend/ de FastAPI
        response = requests.post(f"{API_URL}/recommend/", json={"description": description})
        response.raise_for_status()
        data = response.json()
        recommendations = data.get("recommendations", [])
    except requests.exceptions.RequestException as e:
        return f"Error calling API: {e}"

    return render_template("results.html", recommendations=recommendations)

if __name__ == "__main__":
    # Important : host=0.0.0.0 pour que Docker l'expose
    app.run(host="0.0.0.0", port=5000, debug=True)
