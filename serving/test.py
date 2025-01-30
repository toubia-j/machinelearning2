from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from flask_cors import CORS
import os
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from classes import Preprocessor 
app = Flask(__name__)
CORS(app)  # Autorise les requêtes cross-origin

# -------------------------------------------------------
# 1) Préparation : charger tous les modèles au démarrage
# -------------------------------------------------------
# On peut stocker chaque triplet (model, preprocessor, mlb) dans un dict
# Clé = nom "logique" du modèle, Valeur = (model, preprocessor, mlb)
MODELS = {}

def load_resources_all():
    """
    Charge plusieurs modèles et leurs ressources associées (preprocessor, mlb).
    Adapte les noms/fichiers à tes propres besoins.
    """
    # Dictionnaire de chemins => { "nom_modele": ("model_file", "preproc_file", "mlb_file") }
    models_info = {
        "baseline": (
            "saved_models/model.pkl",
            "saved_models/preprocessor.pkl",
            "saved_models/mlb.pkl",
        ),
        "svm_20250126_222927": (
            "saved_models/svm_20250126_222927.pkl",
            "saved_models/preprocessor.pkl",
            "saved_models/mlb.pkl",
        ),
        "logistic_reg_20250126_222927": (
            "saved_models/logistic_reg_20250126_222927.pkl",
            "saved_models/preprocessor.pkl",
            "saved_models/mlb.pkl",
        )
    }

    global MODELS
    for model_key, (model_path, preproc_path, mlb_path) in models_info.items():
        try:
            # Vérifie l'existence des fichiers
            if not all(os.path.exists(p) for p in [model_path, preproc_path, mlb_path]):
                app.logger.warning(f"Fichier(s) manquant(s) pour {model_key}, vérifie les chemins.")
                continue
            
            # Charge le modèle scikit-learn
            model = joblib.load(model_path)
            preprocessor = joblib.load(preproc_path)
            mlb = joblib.load(mlb_path)

            MODELS[model_key] = (model, preprocessor, mlb)
            app.logger.info(f"✓ Modèle '{model_key}' chargé avec succès.")
        except Exception as e:
            app.logger.error(f"Erreur lors du chargement de {model_key}: {e}")

# -------------------------------------------------------
# 2) Fonction utilitaire pour retourner les probabilités
# -------------------------------------------------------
def get_probas(model, processed_data, mlb, threshold=0.05):
    """
    Calcule les probabilités pour chaque label et
    renvoie un dict { label: prob } filtré par threshold.
    """
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)
            # Combinaison label-proba (probabilities[0] => vecteur de probabilités pour la 1ère ligne)
            prob_dict = {
                label: float(prob)
                for label, prob in zip(mlb.classes_, probabilities[0])
            }
            # Filtrage et tri décroissant
            filtered_sorted = {
                k: v for k, v in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                if v >= threshold
            }
            return filtered_sorted
        return {}
    except Exception as e:
        app.logger.error(f"Erreur de probabilités: {str(e)}")
        return {}

# -------------------------------------------------------
# 3) Charger toutes les ressources au démarrage
# -------------------------------------------------------
load_resources_all()

@app.route('/')
def home():
    """Affiche la page HTML."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Route qui applique la prédiction sur TOUS les modèles
    et renvoie un JSON du type :
    {
      "results": {
        "baseline": {
          "prediction": [...],
          "probabilities": {...}
        },
        "svm_20250126_222927": {
          "prediction": [...],
          "probabilities": {...}
        },
        ...
      }
    }
    """
    try:
        data = request.get_json()
        if not data or 'description' not in data:
            return jsonify({'error': 'Missing "description" field'}), 400
            
        description = data['description']
        if not MODELS:
            return jsonify({'error': 'Aucun modèle n\'est chargé.'}), 500
        
        # Pour chaque modèle, on génère les prédictions
        all_results = {}
        for model_key, (model, preprocessor, mlb) in MODELS.items():
            try:
                # Conversion en Series, transformation TF-IDF
                processed = preprocessor.transform(pd.Series([description]))
                
                # Prédiction binaire
                prediction_array = model.predict(processed)  # ex: [[1, 0, 1, ...]]
                labels = mlb.inverse_transform(prediction_array)
                
                # Probabilités
                probabilities = get_probas(model, processed, mlb, threshold=0.02)
                
                all_results[model_key] = {
                    'prediction': list(labels[0]),
                    'probabilities': probabilities
                }

            except Exception as e:
                app.logger.error(f"Erreur de prédiction avec le modèle {model_key}: {e}")
                all_results[model_key] = {
                    'prediction': [],
                    'probabilities': {},
                    'error': str(e)
                }
        
        return jsonify({'results': all_results})
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
