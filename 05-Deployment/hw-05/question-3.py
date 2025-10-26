import pickle
from typing import Literal
from pydantic import BaseModel, Field
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# class Customer(BaseModel):
#     lead_source: Literal["paid_ads"]
#     number_of_courses_viewed: int = 2
#     annual_income: float = 79276.0

#
# class PredictResponse(BaseModel):
#     churn_probability: float
#     churn: bool



customer = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

customer1 ={
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
# from sklearn.pipeline import make_pipeline
#
# pipeline = make_pipeline(
#     DictVectorizer(),
#     LogisticRegression(solver='liblinear')
# )




with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

result = float(pipeline.predict_proba(customer1)[0, 1])

print(result)
# def predict_single(customer):
#     result = pipeline.predict_proba(customer)[0, 1]
#     return float(result)
#
#
# def predict(customer: Customer) -> PredictResponse:
#     prob = predict_single(customer.model_dump())
#
#     return PredictResponse(
#         churn_probability=prob,
#         churn=prob >= 0.5
#     )


# print(predict(customer=Customer))
