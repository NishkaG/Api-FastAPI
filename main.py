import uvicorn
from fastapi import FastAPI 
from pydantic import BaseModel

import pickle
import numpy as np
import pandas as pd
app = FastAPI()

li=[]
Dict={}

class arima(BaseModel):
    arrival_date: str
    modal_price: float

@app.get('/')
def index():
    return {"trial":"worked"}
    
@app.post('/predict')
async def predict_price(data:arima):
    data=data.dict()

    loaded_model = pickle.load(open('model.pkl', 'rb'))
    index_future_dates=pd.date_range(start='2021-05-30',end='2021-05-30')
    prediction=loaded_model.predict(start=75,end=75)
    prediction.index=index_future_dates
    output=prediction.to_string()
    li.append(output)
    Dict['price']=output
    print(type(output))
    return {
       'prediction': output, 
    }

@app.get('/predictedValue')
async def displayValue():
    return Dict

