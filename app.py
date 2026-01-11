from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from us_visa.pipeline.prediction_pipeline import USvisaClassifier
from us_visa.pipeline.training_pipeline import TrainPipeline

app = FastAPI(title="US Visa Approval API")

# -------------------------
# Request schema
# -------------------------
class VisaRequest(BaseModel):
    continent: str
    education_of_employee: str
    has_job_experience: str
    requires_job_training: str
    no_of_employees: int
    yr_of_estab: int
    region_of_employment: str
    prevailing_wage: float
    unit_of_wage: str
    full_time_position: str


# -------------------------
# Health Check
# -------------------------
@app.get("/")
def home():
    return {"status": "US Visa Model API is running"}


# -------------------------
# Train endpoint
# -------------------------
@app.get("/train")
def train():
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        return {"status": "Model trained and pushed to S3"}
    except Exception as e:
        return {"error": str(e)}


# -------------------------
# Predict endpoint
# -------------------------
@app.post("/predict")
def predict(data: VisaRequest):
    try:
        df = pd.DataFrame([data.dict()])

        model = USvisaClassifier()
        prediction = model.predict(df)[0]

        return {
            "prediction": int(prediction),
            "result": "Visa Approved" if prediction == 1 else "Visa Rejected"
        }

    except Exception as e:
        return {"error": str(e)}
