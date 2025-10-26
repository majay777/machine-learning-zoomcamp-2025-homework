import pickle
from typing import Literal
from pydantic import BaseModel, Field


from fastapi import FastAPI
import uvicorn



class Client(BaseModel):
    lead_source: Literal["organic_search"]
    number_of_courses_viewed: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)



class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool


app = FastAPI(title="client-churn-prediction")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(client):
    result = pipeline.predict_proba(client)[0, 1]
    return float(result)


@app.post("/predict")
def predict(client: Client) -> PredictResponse:
    prob = predict_single(client.model_dump())

    return PredictResponse(
        churn_probability=prob,
        churn=prob >= 0.5
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
