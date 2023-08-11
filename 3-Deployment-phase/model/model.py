from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, Form, HTTPException
from typing import Annotated
from pydantic import BaseModel
import torch
import numpy as np
import uvicorn
import nest_asyncio
import mlflow
import time
import psutil
import model_back

def logging_run(text):
    log = f"INFO:     {text}"
    print(log)

logging_run("Imports complete")

mlflow.set_tracking_uri("http://mlflow-app.mlsd-sentimovie-test.svc:5000")
mlflow.start_run()
run_name = f"Model-A-{int(time.time())}"
mlflow.set_tag("mlflow.runName", run_name)

logging_run("MLflow complete")

app = FastAPI()
nest_asyncio.apply()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

logging_run("FastAPI complete")

class InputJson(BaseModel):
    texts: list


classifier = model_back.MyClassifier()
logging_run("Classifier defined completely")

def monitor_hardware():
    cpu_percent = psutil.cpu_percent() 
    memory_percent = psutil.virtual_memory().percent
    return cpu_percent, memory_percent


def logging_mlflow(start_time):
    latency = time.time() - start_time
    logging_run(f"Latency: {latency:.5f}s")
    cpu_percent, memory_percent = monitor_hardware()
    try:
        mlflow.log_metric("Latency", latency)
        mlflow.log_metric("CPU Usage", cpu_percent) 
        mlflow.log_metric("Memory Usage", memory_percent)
    except Exception as e:
        logging_run(f"MLFLOW not working")
        logging_run(e)


@app.post("/predict")
async def predict(data: InputJson):
    start_time = time.time()

    result = None
    if len(data.texts) > 1:
        result = await classifier.batch_prediction(data.texts)
    elif len(data.texts) == 1:
        result = await classifier.single_prediction(data.texts[0])
    else:
        raise HTTPException(status_code=400, detail="You must send at least one text.")

    logging_mlflow(start_time)

    return result


@app.post("/explainable")
async def predict_explainable(data: InputJson):
    start_time = time.time()
    print(data)
    result = None
    if len(data.texts) > 1:
        raise HTTPException(status_code=400, detail="Just one text you can send for this job.")
    elif len(data.texts) == 1:
        result = await classifier.explainable_prediction(data.texts[0])
    else:
        raise HTTPException(status_code=400, detail="You must send at least one text.")

    logging_mlflow(start_time)

    return result