from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="California Housing API")

@app.on_event("startup")
def load_model():
    global model
    model_name = "california_housing_Random Forest"  # or whatever printed in your flow
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")


class Features(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the California Housing Model API. Use POST /predict to get predictions."
    }


@app.post("/predict")
def predict(features: Features):
    input_df = pd.DataFrame([features.dict()])
    preds = model.predict(input_df)
    return {"predicted_med_house_value": float(preds[0])}
