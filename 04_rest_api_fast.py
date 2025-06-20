# api_exploitation_fastapi.py
# (Note: If you saved this file as fast_api.py, the uvicorn.run command at the bottom needs to reflect that)

# main.py

import os
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Configuration and Global Variables ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "churn_xgb_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "churn_preprocessor.pkl")
ORIGINAL_COLUMNS_PATH = os.path.join(MODELS_DIR, "X_original_columns.pkl")
IMPUTATION_VALUES_PATH = os.path.join(MODELS_DIR, "imputation_values.joblib")

# MODIFICATION: Updated version for CustomerID type change
MODEL_VERSION = "1.3.1" 

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
except FileNotFoundError as e:
    logger.error(f"❌ Erreur de chargement de fichier : {e}. Vérifiez les chemins dans {MODELS_DIR}.")
    raise SystemExit(f"Impossible de charger les artefacts du modèle : {e}")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement des artefacts : {e}", exc_info=True)
    raise SystemExit(f"Erreur critique lors du chargement des artefacts : {e}")

cat_fix = {
    'PreferredLoginDevice': {'Phone': 'Mobile', 'Mobile Phone': 'Mobile'},
    'PreferredPaymentMode': {'CC': 'Credit Card', 'COD': 'Cash on Delivery', 'E wallet': 'E-Wallet'},
    'PreferedOrderCat': {'Mobile Phone': 'Mobile'}
}

cat_cols_for_preprocessor = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                             'PreferedOrderCat', 'MaritalStatus', 'CityTier', 'Complain']
num_cols_for_preprocessor = [col for col in X_original_columns if col not in cat_cols_for_preprocessor]


# --- Pydantic Models for Request and Response ---

class ChurnInput(BaseModel):
    # MODIFICATION: Changed CustomerID from str to int
    CustomerID: int = Field(..., example=50021, description="Unique integer identifier for the customer.")
    
    # All other fields remain required
    Tenure: float = Field(..., example=10.0)
    PreferredLoginDevice: str = Field(..., example="Mobile Phone")
    CityTier: str = Field(..., example="1", description="City tier as string '1', '2', or '3'")
    WarehouseToHome: float = Field(..., example=15.0)
    PreferredPaymentMode: str = Field(..., example="Credit Card")
    Gender: str = Field(..., example="Male")
    HourSpendOnApp: float = Field(..., example=3.0)
    NumberOfDeviceRegistered: int = Field(..., example=3)
    PreferedOrderCat: str = Field(..., example="Mobile")
    SatisfactionScore: int = Field(..., example=3)
    MaritalStatus: str = Field(..., example="Married")
    NumberOfAddress: int = Field(..., example=2)
    Complain: str = Field(..., example="0", description="Complaint status, e.g., '0' for No, '1' for Yes")
    OrderAmountHikeFromlastYear: float = Field(..., example=15.0)
    CouponUsed: float = Field(..., example=1.0)
    OrderCount: float = Field(..., example=2.0)
    DaySinceLastOrder: float = Field(..., example=5.0)
    CashbackAmount: float = Field(..., example=150.0)


class ChurnOutput(BaseModel):
    # MODIFICATION: Changed CustomerID from str to int
    CustomerID: int
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
# MODIFICATION: Updated function signature to accept int for customer_id
def process_prediction_request(data_input_model: ChurnInput, customer_id: int) -> Dict[str, Any]:
    """
    Processes the input data, performs preprocessing, prediction, and returns results.
    """
    try:
        data_input_dict = data_input_model.model_dump(exclude={'CustomerID'})
        processed_input_dict = {}

        for col_name in X_original_columns:
            value = data_input_dict.get(col_name)

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
                    raise HTTPException(status_code=400, detail=f"Invalid value for Complain: '{value}'. Expected '0' or '1'.")
            else:
                processed_input_dict[col_name] = str(value)
        
        df = pd.DataFrame([processed_input_dict], columns=X_original_columns)

        for col, mapping in cat_fix.items():
            if col in df.columns:
                df[col] = df[col].replace(mapping)
        
        if df.isnull().values.any():
            logger.warning("NaN values detected in DataFrame post-validation. Proceeding with imputation.")
            for col in df.columns:
                if df[col].isnull().any():
                    if col in imputation_values_map:
                        fill_value = imputation_values_map[col]
                        df[col].fillna(fill_value, inplace=True)
                        logger.info(f"Imputed '{col}' with (training) '{fill_value}'")
                    else:
                        fallback_value = 0 if col in num_cols_for_preprocessor else "Unknown"
                        df[col].fillna(fallback_value, inplace=True)
                        logger.warning(f"Imputed '{col}' with (fallback) '{fallback_value}'")

        for col in cat_cols_for_preprocessor:
            df[col] = df[col].astype('category')
        
        df_processed = preprocessor.transform(df)
        proba_churn_np = model.predict_proba(df_processed)[0][1]
        prediction_val_np = model.predict(df_processed)[0]

        prediction_label = "Le client risque de churner" if int(prediction_val_np) == 1 else "Le client semble fidèle"
        is_churn_risk = bool(int(prediction_val_np) == 1)
        churn_probability_python_float = float(round(float(proba_churn_np) * 100, 2))

        response_payload = {
            "CustomerID": customer_id,
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
    prediction_results = process_prediction_request(data_input, data_input.CustomerID)
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
    module_name = os.path.splitext(os.path.basename(__file__))[0]
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=port, reload=True)