from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from flask_cors import CORS
import os
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from classes import Preprocessor 
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


MODELS = {}


# -----------------------------------------------------------------
# 2) Fonctions de prédiction
# -----------------------------------------------------------------
def predict_with_embedding(model, embedding_model, label_encoder, description):
    """
    Transforme 'description' en embeddings via 'embedding_model',
    puis fait model.predict(...) et reconvertit l'index → label via label_encoder.
    Renvoie un dict : {
      "prediction": [str],      # label(s) ou juste un label unique
      "probabilities": {label: proba, ...}
    }
    """
    # 1) Générer l'embedding du synopsis (selon ton code)
    #    ex: if embedding_model est un SentenceTransformer, on fait:
    #    synopsis_embedding = embedding_model.encode([description])
    #    ou si c'est un pipeline scikit, on fait embedding_model.transform([...])
    synopsis_embedding = embedding_model.encode([description])  # A adapter

    # 2) Prédire l'étiquette encodée
    genre_encoded = model.predict(synopsis_embedding)  # shape: (1,)

    # 3) Convertir vers label(s) textuels
    prediction_label = label_encoder.inverse_transform(genre_encoded)  # shape: (1,)
    
    # 4) Facultatif: calculer les top probas
    probabilities = {}
    if hasattr(model, "predict_proba"):
        proba_array = model.predict_proba(synopsis_embedding)[0]  # shape: (n_classes,)
        # On récupère la liste de toutes les classes
        classes_encoded = np.arange(len(label_encoder.classes_))  # [0..N-1]
        # On mappe each class_encoded -> label -> proba
        for c_encoded, p_val in zip(classes_encoded, proba_array):
            c_label = label_encoder.inverse_transform([c_encoded])[0]
            probabilities[c_label] = float(p_val)

        # On peut trier par proba décroissante
        probabilities = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))

    return {
        "prediction": list(prediction_label),
        "probabilities": probabilities
    }


def predict_with_tfidf(model, preprocessor, mlb, description, threshold=0.02):
    """
    Exemple pour un modèle TF-IDF multi-label (SVM, Logistic, etc.).
    """
    processed = preprocessor.transform(pd.Series([description]))
    prediction_array = model.predict(processed)  # ex: [[1,0,1,...]]
    labels = mlb.inverse_transform(prediction_array)
    
    # Probabilités si existant
    probabilities = {}
    if hasattr(model, "predict_proba"):
        # shape: (1, nb_labels)
        prob_matrix = model.predict_proba(processed)
        # On associe mlb.classes_ => prob
        prob_dict = {label: float(prob_matrix[0][i]) for i, label in enumerate(mlb.classes_)}
        # filtrer/ trier
        probabilities = {
            k: v for k, v in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            if v >= threshold
        }
    return {
        "prediction": list(labels[0]),
        "probabilities": probabilities
    }


# -----------------------------------------------------------------
# 3) Charger les ressources au démarrage
# -----------------------------------------------------------------
def load_resources_all():
    """
    Exemple de chargement de multiples modèles :
      - 2 basés sur embeddings (KNN / RF)
      - 2 basés sur TF-IDF (SVM / logistic_reg)
    Adapte les chemins selon tes fichiers .pkl existants
    """
    global MODELS

    # Chemins KNN / RF
    knn_path = "saved_models/KNeighborsClassifier.pkl"
    rf_path  = "saved_models/RandomForestClassifier.pkl"

    # Chemin encodeur (embedding) => "label_encodeur.pkl"
    #   ATTENTION : si c'est vraiment un LabelEncoder() scikit, 
    #   ou un SentenceTransformer() (embedding) ? 
    #   D'après ton code, on dirait un sentence_transformer.
    #   On suppose c'est un pickle d'un HF embedding. 
    embedding_model_path = "saved_models/description_embeddings.pkl"  

    # Chemin du label_encoder (pour retransformer la sortie model -> nom de catégorie)
    label_encoder_path = "saved_models/label_encoder.pkl"

    # Chemins d'exemple SVM / Logistic TF-IDF
    svm_path = "saved_models/svm_20250126_222927.pkl"
    logistic_path = "saved_models/logistic_reg_20250126_222927.pkl"
    preprocessor_path = "saved_models/preprocessor.pkl"
    mlb_path = "saved_models/mlb.pkl"

    # 1) Charger KNN
    if os.path.exists(knn_path) and os.path.exists(embedding_model_path) and os.path.exists(label_encoder_path):
        knn_model = joblib.load(knn_path)
        embedding_model = joblib.load(embedding_model_path)
        label_enc = joblib.load(label_encoder_path)

        MODELS["knn"] = {
            "type": "embedding",
            "model": knn_model,
            "embedding_model": embedding_model,
            "label_encoder": label_enc
        }

    # 2) Charger RF
    if os.path.exists(rf_path) and os.path.exists(embedding_model_path) and os.path.exists(label_encoder_path):
        rf_model = joblib.load(rf_path)
        embedding_model = joblib.load(embedding_model_path)
        label_enc = joblib.load(label_encoder_path)

        MODELS["rf"] = {
            "type": "embedding",
            "model": rf_model,
            "embedding_model": embedding_model,
            "label_encoder": label_enc
        }

    # 3) Charger SVM / Logistic
    if os.path.exists(svm_path) and os.path.exists(preprocessor_path) and os.path.exists(mlb_path):
        svm_model = joblib.load(svm_path)
        preproc = joblib.load(preprocessor_path)
        mlb = joblib.load(mlb_path)

        MODELS["svm_20250126"] = {
            "type": "tfidf",
            "model": svm_model,
            "preprocessor": preproc,
            "mlb": mlb
        }

    if os.path.exists(logistic_path) and os.path.exists(preprocessor_path) and os.path.exists(mlb_path):
        log_model = joblib.load(logistic_path)
        preproc = joblib.load(preprocessor_path)
        mlb = joblib.load(mlb_path)

        MODELS["logistic_20250126"] = {
            "type": "tfidf",
            "model": log_model,
            "preprocessor": preproc,
            "mlb": mlb
        }

    print("Chargement terminé. Modèles disponibles :", list(MODELS.keys()))


load_resources_all()


# -----------------------------------------------------------------
# 4) Routes Flask
# -----------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Applique la prédiction à TOUS les modèles chargés (embedding ou tfidf)
    et renvoie un JSON du type :
    {
      "results": {
        "knn": { "prediction": [...], "probabilities": {...} },
        "rf":  { ... },
        "svm_20250126": { ... },
        ...
      }
    }
    """
    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({'error': 'Missing "description" field'}), 400
    
    description = data['description']
    if not MODELS:
        return jsonify({'error': 'Aucun modèle n\'est chargé.'}), 500

    all_results = {}
    for model_key, info in MODELS.items():
        try:
            model_type = info["type"]
            if model_type == "embedding":
                # KNN / RF => besoin embedding + label_encoder
                model = info["model"]
                emb_model = info["embedding_model"]
                lbl_enc = info["label_encoder"]
                result = predict_with_embedding(model, emb_model, lbl_enc, description)
                all_results[model_key] = result

            elif model_type == "tfidf":
                # SVM / Logistic => besoin preprocessor + mlb
                model = info["model"]
                preproc = info["preprocessor"]
                mlb = info["mlb"]
                result = predict_with_tfidf(model, preproc, mlb, description)
                all_results[model_key] = result
            else:
                all_results[model_key] = {"error": f"Type inconnu: {model_type}"}
        except Exception as e:
            all_results[model_key] = {"error": str(e)}
    #print(all_results)
    return jsonify({"results": all_results})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
