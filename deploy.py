from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI(
    title="Breast Cancer Detection System 🧬",
    description="""
    ## 🤖 AI Diagnostic API (Neural Network)
    هذا النظام يستخدم تقنيات التعلم العميق لتصنيف أورام الثدي بدقة تصل إلى **98%**.
    
    ### 📊 معلومات المودل:
    * **Model Architecture:** Artificial Neural Network (ANN)
    * **Input Features:** 30 Medical Measurements
    * **Output:** Malignant (0) or Benign (1)
    * **Status:** Operational ✅
    
    ---
    **Developer:** Amjad Al-thobaiti
    """,
    version="2.1.0"
)

try:
    model = load_model('model_weights.h5')
    scaler = joblib.load('scaler_weights.pkl')
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading weights: {e}")

class MedicalFeatures(BaseModel):
    features: list 


@app.get("/", tags=["Health Check"])
def home():
    return {
        "status": "Online",
        "accuracy_score": "98%",
        "message": "Welcome to the Breast Cancer Prediction API"
    }

@app.post("/predict", tags=["AI Core Prediction"])
def predict_cancer(data: MedicalFeatures):
    try:
        input_array = np.array(data.features).reshape(1, -1)
        
        if input_array.shape[1] != 30:
            raise HTTPException(status_code=400, detail="Expected 30 features, got " + str(input_array.shape[1]))

        scaled_input = scaler.transform(input_array)
        
        prediction_prob = model.predict(scaled_input)
        prediction_class = (prediction_prob > 0.5).astype("int")
        
        result_label = "Benign (حميد)" if prediction_class[0][0] == 1 else "Malignant (خبيث)"
        confidence = float(prediction_prob[0][0]) if prediction_class[0][0] == 1 else float(1 - prediction_prob[0][0])

        return {
            "prediction_code": int(prediction_class[0][0]),
            "result": result_label,
            "confidence_score": f"{round(confidence * 100, 2)}%",
            "model_accuracy": "98%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)