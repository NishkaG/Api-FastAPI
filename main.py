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
    Arrival_Date: str
    modal_price: float

@app.get('/')
def index():
    return {"trial":"worked"}
    
@app.post('/predict')
async def predict_price(data:arima):
    data=data.dict()

    # loaded_model = pickle.load(open('model1.pkl', 'rb'))
    # index_future_dates=pd.date_range(start='2022-01-08',end='2022-01-17')
    # prediction=loaded_model.predict(start=83,end=92)
    # prediction.index=index_future_dates
    # output=prediction.to_string()
    # print(output)
    # li.append(output)
    # Dict['price']=prediction
    # return {
    #    'prediction': output, 
    # }
    # return Dict

    #crop: Potato Mandi:Amravati
    
    loadedModel_potato_AM=pickle.load(open('amrawati_potato.pkl','rb'))
    index_future_dates_PAM=pd.date_range(start='2022-03-06',end='2022-03-11')
    prediction_PAM=loadedModel_potato_AM.predict(start=184,end=189)
    prediction_PAM.index=index_future_dates_PAM
    display1={}
    display1=prediction_PAM.to_dict()
    val1=display1.values()
    maximumAMP=max(val1)
    minimumAMP=min(val1)
    Dict['Amravati_Potato']={'Maximum':maximumAMP,'Minimum':minimumAMP}
    #crop: Potato Mandi:Mumbai
    
    loadedModel_potato_MUM=pickle.load(open('mumbai_potato.pkl','rb'))
    index_future_dates_PMUM=pd.date_range(start='2022-03-08',end='2022-03-13')
    prediction_PMUM=loadedModel_potato_MUM.predict(start=347,end=352)
    prediction_PMUM.index=index_future_dates_PMUM
    output_PMUM=prediction_PMUM.to_string()
    display2={}
    display2=prediction_PMUM.to_dict()
    val2=display2.values()
    maximumMUMP=max(val2)
    minimumMUMP=min(val2)
    Dict['Mumbai_Potato']={'Maximum':maximumMUMP,'Minimum':minimumMUMP}
    
    #crop: Tomato Mandi:Amravati
    
    loadedModel_tomato_AM=pickle.load(open('amrawati_tomato.pkl','rb'))
    index_future_dates_TAM=pd.date_range(start='2022-03-08',end='2022-03-13')
    prediction_TAM=loadedModel_tomato_AM.predict(start=181,end=186)
    prediction_TAM.index=index_future_dates_TAM
    output_TAM=prediction_TAM.to_string()
    display3={}
    display3=prediction_TAM.to_dict()
    val3=display3.values()
    maximumAMT=max(val3)
    minimumAMT=min(val3)
    Dict['Amrawati_Tomato']={'Maximum':maximumAMT,'Minimum':minimumAMT}
   
    #crop: Tomato Mandi:Mumbai
    
    loadedModel_tomato_MUM=pickle.load(open('mumbai_tomato.pkl','rb'))
    index_future_dates_TMUM=pd.date_range(start='2022-03-08',end='2022-03-13')
    prediction_TMUM=loadedModel_tomato_MUM.predict(start=349,end=354)
    prediction_TMUM.index=index_future_dates_TMUM
    output_TMUM=prediction_TMUM.to_string()
    display4={}
    display4=prediction_TMUM.to_dict()
    val4=display4.values()
    maximumMUMT=max(val4)
    minimumMUMT=min(val4)
    Dict['Mumbai_Tomato']={'Maximum':maximumMUMT,'Minimum':minimumMUMT}

    #crop: Carrot Mandi:Amravati
    
    loadedModel_carrot_AM=pickle.load(open('amrawati_carrot.pkl','rb'))
    index_future_dates_CAM=pd.date_range(start='2022-03-08',end='2022-03-13')
    prediction_CAM=loadedModel_carrot_AM.predict(start=159,end=164)
    prediction_CAM.index=index_future_dates_CAM
    output_CAM=prediction_CAM.to_string()
    display5={}
    display5=prediction_CAM.to_dict()
    val5=display5.values()
    maximumAMC=max(val5)
    minimumAMC=min(val5)
    Dict['Amravati_Carrot']={'Maximum':maximumAMC,'Minimum':minimumAMC}

     #crop: Carrot Mandi:Mumbai
    
    loadedModel_carrot_MUM=pickle.load(open('mumbai_carrot.pkl','rb'))
    index_future_dates_CMUM=pd.date_range(start='2022-03-08',end='2022-03-13')
    prediction_CMUM=loadedModel_carrot_MUM.predict(start=352,end=357)
    prediction_CMUM.index=index_future_dates_CMUM
    output_CMUM=prediction_CMUM.to_string()
    display6={}
    display6=prediction_CMUM.to_dict()
    val6=display6.values()
    maximumMUMC=max(val6)
    minimumMUMC=min(val6)
    Dict['Mumbai_Carrot']={'Maximum':maximumMUMC,'Minimum':minimumMUMC}
    
    #crop: Maize Mandi:Amravati
    
    loadedModel_maize_AM=pickle.load(open('amrawati_maize.pkl','rb'))
    index_future_dates_MAM=pd.date_range(start='2022-03-08',end='2022-03-13')
    prediction_MAM=loadedModel_maize_AM.predict(start=182,end=187)
    prediction_MAM.index=index_future_dates_MAM
    output_MAM=prediction_MAM.to_string()
    display7={}
    display7=prediction_MAM.to_dict()
    val7=display7.values()
    maximumAMM=max(val7)
    minimumAMM=min(val7)
    Dict['Amravati_Maize']={'Maximum':maximumAMM,'Minimum':minimumAMM}
    
    #crop: Maize Mandi:Mumbai
    
    loadedModel_maize_MUM=pickle.load(open('mumbai_maize.pkl','rb'))
    index_future_dates_MMUM=pd.date_range(start='2022-03-06',end='2022-03-11')
    prediction_MMUM=loadedModel_maize_MUM.predict(start=321,end=326)
    prediction_MMUM.index=index_future_dates_MMUM
    output_CMUM=prediction_MMUM.to_string()
    display8={}
    display8=prediction_MMUM.to_dict()
    val8=display8.values()
    maximumMUMM=max(val8)
    minimumMUMM=min(val8)
    Dict['Mumbai_Maize']={'Maximum':maximumMUMM,'Minimum':minimumMUMM}

    #crop: Wheat Mandi:Amravati
    
    loadedModel_wheat_AM=pickle.load(open('amrawati_wheat.pkl','rb'))
    index_future_dates_WAM=pd.date_range(start='2022-03-08',end='2022-03-13')
    prediction_WAM=loadedModel_wheat_AM.predict(start=202,end=207)
    prediction_WAM.index=index_future_dates_WAM
    output_WAM=prediction_WAM.to_string()
    display9={}
    display9=prediction_WAM.to_dict()
    val9=display9.values()
    maximumAMW=max(val9)
    minimumAMW=min(val9)
    Dict['Amravati_Wheat']={'Maximum':maximumAMW,'Minimum':minimumAMW}

    #crop: Wheat Mandi:Mumbai
    
    loadedModel_wheat_MUM=pickle.load(open('mumbai_wheat.pkl','rb'))
    index_future_dates_WMUM=pd.date_range(start='2022-03-08',end='2022-03-13')
    prediction_WMUM=loadedModel_wheat_MUM.predict(start=333,end=338)
    prediction_WMUM.index=index_future_dates_WMUM
    output_WMUM=prediction_WMUM.to_string()
    display10={}
    display10=prediction_WMUM.to_dict()
    val10=display10.values()
    maximumMUMW=max(val10)
    minimumMUMW=min(val10)
    Dict['Mumbai_Wheat']={'Maximum':maximumMUMW,'Minimum':minimumMUMW}
    
     #crop: Green chillies Mandi:Amravati
    
    loadedModel_chilli_AM=pickle.load(open('amrawati_chilli.pkl','rb'))
    index_future_dates_GCAM=pd.date_range(start='2022-03-08',end='2022-03-13')
    prediction_GCAM=loadedModel_chilli_AM.predict(start=161,end=166)
    prediction_GCAM.index=index_future_dates_GCAM
    output_GCAM=prediction_GCAM.to_string()
    display11={}
    display11=prediction_GCAM.to_dict()
    val11=display11.values()
    maximumAMGC=max(val11)
    minimumAMGC=min(val11)
    Dict['Amravati_GreenChilli']={'Maximum':maximumAMGC,'Minimum':minimumAMGC}

    # #crop: Green Chillies Mandi:Mumbai
    loadedModel_chilli_MUM=pickle.load(open('mumbai_chilli.pkl','rb'))
    index_future_dates_GCMUM=pd.date_range(start='2022-03-08',end='2022-03-13')
    prediction_GCMUM=loadedModel_chilli_MUM.predict(start=345,end=350)
    prediction_GCMUM.index=index_future_dates_GCMUM
    output_GCMUM=prediction_GCMUM.to_string()
    display12={}
    display12=prediction_GCMUM.to_dict()
    val12=display12.values()
    maximumMUMGC=max(val12)
    minimumMUMGC=min(val12)
    Dict['Mumbai_GreenChilli']={'Maximum':maximumMUMGC,'Minimum':minimumMUMGC}
    
    return Dict

@app.get('/predictedValue')
async def displayValue():
    return Dict

