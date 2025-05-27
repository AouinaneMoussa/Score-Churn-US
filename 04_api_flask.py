# api_exploitation.py

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import logging # Pour un meilleur logging

# --- Configuration et Chargement des Modèles (au démarrage de l'application) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Répertoire du script api_exploitation.py
MODELS_DIR = os.path.join(BASE_DIR, "models") # models est un sous-dossier

MODEL_PATH = os.path.join(MODELS_DIR, "churn_xgb_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "churn_preprocessor.pkl")
ORIGINAL_COLUMNS_PATH = os.path.join(MODELS_DIR, "X_original_columns.pkl")
IMPUTATION_VALUES_PATH = os.path.join(MODELS_DIR, "imputation_values.joblib") # Nouveau chemin

MODEL_VERSION = "1.0.1" # Mettez à jour la version si vous le souhaitez

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_original_columns = joblib.load(ORIGINAL_COLUMNS_PATH)
    imputation_values_map = joblib.load(IMPUTATION_VALUES_PATH) # Charger les valeurs d'imputation
    logging.info(f"✅ Modèle (v{MODEL_VERSION}), preprocessor, colonnes originales et valeurs d'imputation chargés.")
    logging.info(f"Valeurs d'imputation disponibles: {imputation_values_map}")
except FileNotFoundError as e:
    logging.error(f"❌ Erreur de chargement de fichier : {e}. Vérifiez les chemins dans {MODELS_DIR}.")
    raise SystemExit(f"Impossible de charger les artefacts du modèle : {e}")
except Exception as e:
    logging.error(f"❌ Erreur lors du chargement des artefacts : {e}")
    raise SystemExit(f"Erreur critique lors du chargement des artefacts : {e}")

cat_fix = {
    'PreferredLoginDevice': {'Phone': 'Mobile', 'Mobile Phone': 'Mobile'},
    'PreferredPaymentMode': {'CC': 'Credit Card', 'COD': 'Cash on Delivery', 'E wallet': 'E-Wallet'},
    'PreferedOrderCat': {'Mobile Phone': 'Mobile'}
}

cat_cols_for_preprocessor = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                             'PreferedOrderCat', 'MaritalStatus', 'CityTier', 'Complain']
num_cols_for_preprocessor = [col for col in X_original_columns if col not in cat_cols_for_preprocessor]

# --- Initialisation de l'application Flask ---
app = Flask(__name__)

# --- Logique de Prédiction ---
def get_churn_prediction_details(data_input: dict) -> tuple[dict, int]:
    try:
        input_dict = {}
        missing_required_fields = []

        for col_name in X_original_columns:
            value = data_input.get(col_name)

            if value is None: # Gérer les None explicitement avant conversion de type
                input_dict[col_name] = np.nan # Sera traité par l'imputation ci-dessous
                # Vous pourriez vouloir vérifier si un champ requis est None ici, selon votre contrat d'API
                # Par exemple, si 'Tenure' est toujours requis:
                # if col_name == 'Tenure': # Remplacez par vos champs réellement requis
                #    missing_required_fields.append(col_name)
                continue # Passe à la colonne suivante

            # Conversion de type et gestion spécifique
            if col_name in num_cols_for_preprocessor:
                try:
                    input_dict[col_name] = float(value)
                except (ValueError, TypeError):
                    return {"error": f"Invalid numeric value for {col_name}: '{value}'"}, 400
            elif col_name == 'CityTier':
                try:
                    input_dict[col_name] = int(str(value)) # CityTier est traité comme cat. mais peut venir en int/str
                except (ValueError, TypeError):
                    return {"error": f"Invalid value for CityTier: '{value}'. Expected '1', '2', or '3'."}, 400
            elif col_name == 'Complain':
                complain_str = str(value).lower() # Convertir en minuscule pour plus de flexibilité
                if "1" in complain_str or complain_str == "true":
                    input_dict[col_name] = 1
                elif "0" in complain_str or complain_str == "false":
                    input_dict[col_name] = 0
                else:
                    # Si la valeur n'est ni 0 ni 1 après conversion, elle est invalide
                    # ou pourrait être traitée comme NaN pour imputation si cela a du sens
                     return {"error": f"Invalid value for Complain: '{value}'. Expected '0 (Non)', '1 (Oui)', 0, 1, 'true', or 'false'."}, 400
            else: # Colonnes catégorielles textuelles
                input_dict[col_name] = str(value)

        # if missing_required_fields:
        #    return {"error": f"Missing required fields: {', '.join(missing_required_fields)}"}, 400

        df = pd.DataFrame([input_dict], columns=X_original_columns)

        for col, mapping in cat_fix.items():
            if col in df.columns:
                df[col] = df[col].replace(mapping)

        # 4. Imputation des valeurs manquantes avec les valeurs du training
        imputed_cols_log = []
        fallback_imputed_cols_log = []

        for col in X_original_columns:
            if col in df.columns and df[col].isnull().any():
                if col in imputation_values_map:
                    fill_value = imputation_values_map[col]
                    df[col].fillna(fill_value, inplace=True)
                    imputed_cols_log.append(f"'{col}' avec (training) '{fill_value}'")
                else:
                    # Ce cas arrive si l'input API a un NaN pour un champ qui était toujours rempli dans le train
                    # (et donc n'a pas de valeur d'imputation sauvegardée pour cette colonne).
                    fallback_value = 0 if col in num_cols_for_preprocessor else "Unknown_API_Input"
                    df[col].fillna(fallback_value, inplace=True)
                    fallback_imputed_cols_log.append(f"'{col}' avec (fallback) '{fallback_value}'")
                    logging.warning(f"Colonne '{col}' avec NaN dans l'input mais pas de valeur d'imputation du training. Remplie avec la valeur de repli '{fallback_value}'.")
            elif col not in df.columns: # Devrait être impossible si X_original_columns est bien utilisé
                 logging.error(f"Logique d'erreur: la colonne '{col}' de X_original_columns n'est pas dans le DataFrame créé.")
                 return {"error": f"Internal error: Missing expected model feature during DataFrame creation: {col}"}, 500


        if imputed_cols_log:
            logging.info(f"Imputation (valeurs du training): {', '.join(imputed_cols_log)}")
        if fallback_imputed_cols_log:
            logging.warning(f"Imputation (valeurs de repli): {', '.join(fallback_imputed_cols_log)}")


        for col in cat_cols_for_preprocessor:
            if col in df.columns:
                df[col] = df[col].astype('category')
            else: # Ne devrait pas arriver
                 return {"error": f"Internal error: Missing categorical column for preprocessing: {col}"}, 500

        df_processed = preprocessor.transform(df)

        proba_churn = model.predict_proba(df_processed)[0][1]
        prediction_val = model.predict(df_processed)[0]

        # Dans la fonction get_churn_prediction_details de api_exploitation.py

        # ... (calcul de proba_churn et prediction_val) ...

        prediction_label = "Le client risque de churner" if prediction_val == 1 else "Le client semble fidèle"
        
        # Conversion explicite en types Python natifs pour la sérialisation JSON
        is_churn_risk = bool(prediction_val == 1) # Assure un booléen Python
        churn_probability_python_float = float(round(proba_churn * 100, 2)) # Convertit en float Python

        response_payload = {
            "predictionLabel": prediction_label,
            "churnProbability": churn_probability_python_float, # Utiliser la valeur convertie
            "isChurnRisk": is_churn_risk,
            "modelVersion": MODEL_VERSION
        }
        return response_payload, 200

    except Exception as e:
        logging.error(f"Erreur interne lors de la prédiction : {e}", exc_info=True)
        return {"error": "Erreur interne du serveur lors de la prédiction.", "details": str(e)}, 500

# --- Définition de l'Endpoint API ---
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    input_data = request.get_json()
    if not input_data:
        return jsonify({"error": "No input data provided"}), 400
    
    # Vérifier que tous les X_original_columns sont présents ou peuvent être None
    # for col_name in X_original_columns:
    #    if col_name not in input_data:
    #        # Décidez si c'est une erreur ou si None/NaN est implicite et sera imputé
    #        # Pour être strict :
    #        # return jsonify({"error": f"Missing required field in input: {col_name}"}), 400
    #        pass # On laisse la logique d'imputation gérer les None plus tard

    response_data, status_code = get_churn_prediction_details(input_data)
    return jsonify(response_data), status_code

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "modelVersion": MODEL_VERSION, "imputationValuesLoaded": bool(imputation_values_map)}), 200

# --- Exécution de l'application ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    # Pour la production, utilisez un serveur WSGI comme Gunicorn:
    # gunicorn --workers 4 --bind 0.0.0.0:5000 api_exploitation:app
    app.run(host='0.0.0.0', port=port, debug=False) # debug=False pour la prod