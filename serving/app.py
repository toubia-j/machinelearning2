from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from classes import Preprocessor 

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Ajoutez cette ligne

def get_probas(model, processed_data, mlb, threshold=0.05):
    """Filtre les probabilités avec seuil de 5%"""
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)
            # Combinaison des labels avec leurs probabilités
            prob_dict = {label: float(prob) for label, prob in zip(mlb.classes_, probabilities[0])}
            # Filtrage et tri décroissant
            return {
                k: v for k, v in sorted(prob_dict.items(), 
                key=lambda item: item[1], 
                reverse=True) 
                if v >= threshold
            }
        return {}
    except Exception as e:
        app.logger.error(f"Erreur de probabilités: {str(e)}")
        return {}

# Charger le modèle une fois au démarrage
def load_resources():
    global model, preprocessor, mlb
    try:
        model = joblib.load('saved_models/model.pkl')
        preprocessor = joblib.load('saved_models/preprocessor.pkl')
        mlb = joblib.load('saved_models/mlb.pkl')
        print("Modèle chargé :", type(model))
        print("Préprocesseur chargé :", type(preprocessor))
        print("Classes disponibles :", mlb.classes_)
    except Exception as e:
        app.logger.error(f"Error loading resources: {str(e)}")
        raise

# Chargement initial des ressources
load_resources()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'description' not in data:
            return jsonify({'error': 'Missing description field'}), 400
            
        description = data['description']
        
        # Conversion explicite en Series pandas
        processed = preprocessor.transform(pd.Series([description]))
        
        # Prédiction
        prediction = model.predict(processed)
        labels = mlb.inverse_transform(prediction)
        
        # Probabilités
        probabilities = get_probas(model, processed, mlb, threshold=0.02)
        
        return jsonify({
            'prediction': list(labels[0]),
            'probabilities': probabilities
        })
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)