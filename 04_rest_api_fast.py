# api_exploitation_fastapi.py
# (Note: If you saved this file as fast_api.py, the uvicorn.run command at the bottom needs to reflect that)

import os
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field, field_validator, model_validator

# --- Configuration and Global Variables ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "churn_xgb_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "churn_preprocessor.pkl")
ORIGINAL_COLUMNS_PATH = os.path.join(MODELS_DIR, "X_original_columns.pkl")
IMPUTATION_VALUES_PATH = os.path.join(MODELS_DIR, "imputation_values.joblib")

MODEL_VERSION = "1.1.0" # Updated version for FastAPI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load models and other artifacts at startup
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_original_columns: List[str] = joblib.load(ORIGINAL_COLUMNS_PATH)
    imputation_values_map: Dict[str, Any] = joblib.load(IMPUTATION_VALUES_PATH)
    logger.info(f"✅ Modèle (v{MODEL_VERSION}), preprocessor, colonnes originales et valeurs d'imputation chargés.")
    logger.info(f"Valeurs d'imputation disponibles: {imputation_values_map}")
except FileNotFoundError as e:
    logger.error(f"❌ Erreur de chargement de fichier : {e}. Vérifiez les chemins dans {MODELS_DIR}.")
    raise SystemExit(f"Impossible de charger les artefacts du modèle : {e}")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement des artefacts : {e}", exc_info=True)
    raise SystemExit(f"Erreur critique lors du chargement des artefacts : {e}")

cat_fix = {
    'PreferredLoginDevice': {'Phone': 'Mobile', 'Mobile Phone': 'Mobile'},
    'PreferredPaymentMode': {'CC': 'Credit Card', 'COD': 'Cash on Delivery', 'E wallet': 'E-Wallet'},
    'PreferedOrderCat': {'Mobile Phone': 'Mobile'} # Note: 'PreferedOrderCat' is the name in X_original_columns
}

cat_cols_for_preprocessor = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                             'PreferedOrderCat', 'MaritalStatus', 'CityTier', 'Complain']
num_cols_for_preprocessor = [col for col in X_original_columns if col not in cat_cols_for_preprocessor]


# --- Pydantic Models for Request and Response ---
class ChurnInput(BaseModel):
    Tenure: Optional[float] = Field(None, example=10.0)
    PreferredLoginDevice: Optional[str] = Field(None, example="Mobile")
    CityTier: Optional[str] = Field(None, example="1", description="City tier as string '1', '2', or '3'") # Will be converted to int in logic
    WarehouseToHome: Optional[float] = Field(None, example=15.0)
    PreferredPaymentMode: Optional[str] = Field(None, example="Credit Card")
    Gender: Optional[str] = Field(None, example="Male")
    HourSpendOnApp: Optional[float] = Field(None, example=3.0)
    NumberOfDeviceRegistered: Optional[int] = Field(None, example=3)
    PreferedOrderCat: Optional[str] = Field(None, example="Mobile") # Matches X_original_columns
    SatisfactionScore: Optional[int] = Field(None, example=3)
    MaritalStatus: Optional[str] = Field(None, example="Married")
    NumberOfAddress: Optional[int] = Field(None, example=2)
    Complain: Optional[str] = Field(None, example="0 (Non)", description="Complaint status, e.g., '0 (Non)', '1 (Oui)', '0', '1'") # Will be converted to int
    OrderAmountHikeFromlastYear: Optional[float] = Field(None, example=15.0)
    CouponUsed: Optional[float] = Field(None, example=1.0) # Kept as float as per original numeric processing
    OrderCount: Optional[float] = Field(None, example=2.0)   # Kept as float
    DaySinceLastOrder: Optional[float] = Field(None, example=5.0) # Kept as float
    CashbackAmount: Optional[float] = Field(None, example=150.0)

    @model_validator(mode='before')
    @classmethod
    def ensure_all_original_cols_present_or_none(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return values


class ChurnOutput(BaseModel):
    predictionLabel: str
    churnProbability: float
    isChurnRisk: bool
    modelVersion: str

class HealthOutput(BaseModel):
    status: str
    modelVersion: str
    imputationValuesLoaded: bool

# --- FastAPI Application Instance ---
app = FastAPI(
    title="Churn Prediction API",
    description="API to predict customer churn based on input features.",
    version=MODEL_VERSION
)

# --- Core Prediction Logic ---
def process_prediction_request(data_input_model: ChurnInput) -> Dict[str, Any]:
    """
    Processes the input data, performs preprocessing, prediction, and returns results.
    Raises HTTPException for client or server errors.
    """
    try:
        data_input_dict = data_input_model.model_dump(exclude_unset=False)
        processed_input_dict = {}

        for col_name in X_original_columns:
            value = data_input_dict.get(col_name)
            if value is None:
                processed_input_dict[col_name] = np.nan
                continue
            if col_name in num_cols_for_preprocessor:
                try:
                    processed_input_dict[col_name] = float(value)
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail=f"Invalid numeric value for {col_name}: '{value}'")
            elif col_name == 'CityTier':
                try:
                    processed_input_dict[col_name] = int(str(value))
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail=f"Invalid value for CityTier: '{value}'. Expected '1', '2', or '3'.")
            elif col_name == 'Complain':
                complain_str_val = str(value).lower()
                if "1" in complain_str_val or complain_str_val == "true":
                    processed_input_dict[col_name] = 1
                elif "0" in complain_str_val or complain_str_val == "false":
                    processed_input_dict[col_name] = 0
                else:
                    raise HTTPException(status_code=400, detail=f"Invalid value for Complain: '{value}'. Expected '0 (Non)', '1 (Oui)', '0', '1', 'true', or 'false'.")
            else:
                processed_input_dict[col_name] = str(value)
        
        df = pd.DataFrame([processed_input_dict], columns=X_original_columns)

        for col, mapping in cat_fix.items():
            if col in df.columns:
                df[col] = df[col].replace(mapping)

        imputed_cols_log = []
        fallback_imputed_cols_log = []
        for col in X_original_columns:
            if col in df.columns and df[col].isnull().any():
                if col in imputation_values_map:
                    fill_value = imputation_values_map[col]
                    df[col].fillna(fill_value, inplace=True)
                    imputed_cols_log.append(f"'{col}' with (training) '{fill_value}'")
                else:
                    fallback_value = 0 if col in num_cols_for_preprocessor else "Unknown_API_Input"
                    df[col].fillna(fallback_value, inplace=True)
                    fallback_imputed_cols_log.append(f"'{col}' with (fallback) '{fallback_value}'")
                    logger.warning(f"Colonne '{col}' avec NaN dans l'input mais pas de valeur d'imputation du training. Remplie avec '{fallback_value}'.")
            elif col not in df.columns:
                 logger.error(f"Logique d'erreur: la colonne '{col}' de X_original_columns n'est pas dans le DataFrame créé.")
                 raise HTTPException(status_code=500, detail=f"Internal error: Missing expected model feature: {col}")

        if imputed_cols_log:
            logger.info(f"Imputation (valeurs du training): {', '.join(imputed_cols_log)}")
        if fallback_imputed_cols_log:
            logger.warning(f"Imputation (valeurs de repli): {', '.join(fallback_imputed_cols_log)}")

        for col in cat_cols_for_preprocessor:
            if col in df.columns:
                df[col] = df[col].astype('category')
            else:
                logger.error(f"Missing categorical column for preprocessing: {col}")
                raise HTTPException(status_code=500, detail=f"Internal error: Missing categorical column for preprocessing: {col}")
        
        df_processed = preprocessor.transform(df)
        proba_churn_np = model.predict_proba(df_processed)[0][1]
        prediction_val_np = model.predict(df_processed)[0]

        prediction_label = "Le client risque de churner" if int(prediction_val_np) == 1 else "Le client semble fidèle"
        is_churn_risk = bool(int(prediction_val_np) == 1)
        churn_probability_python_float = float(round(float(proba_churn_np) * 100, 2))

        response_payload = {
            "predictionLabel": prediction_label,
            "churnProbability": churn_probability_python_float,
            "isChurnRisk": is_churn_risk,
            "modelVersion": MODEL_VERSION
        }
        return response_payload

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur interne lors de la prédiction : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur lors de la prédiction: {str(e)}")

# --- API Endpoints ---
@app.post("/predict", response_model=ChurnOutput, summary="Predict Customer Churn")
async def predict_churn_endpoint(data_input: ChurnInput):
    prediction_results = process_prediction_request(data_input)
    return prediction_results

@app.get("/health", response_model=HealthOutput, summary="Health Check")
async def health_check_endpoint():
    return {
        "status": "healthy",
        "modelVersion": MODEL_VERSION,
        "imputationValuesLoaded": bool(imputation_values_map is not None and len(imputation_values_map) > 0)
    }

# --- To Run the Application (using Uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    # Get the name of the current file (module)
    # This assumes you run `python your_file_name.py`
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    port = int(os.environ.get("PORT", 5000))
    # reload=True is for development, remove or set to False for production
    # Use the dynamically obtained module_name
    uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=port, reload=True)