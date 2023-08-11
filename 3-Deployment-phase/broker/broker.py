from fastapi import FastAPI
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, Form, HTTPException
from typing import Annotated
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import uvicorn
import nest_asyncio
import mlflow
import asyncio
import time
import psutil
import requests
import dbrepo


def logging_run(text):
    log = f"INFO:     {text}"
    print(log)

logging_run("Imports complete")

mlflow.set_tracking_uri("http://mlflow-app.mlsd-sentimovie-test.svc:5000")
mlflow.start_run()
run_name = f"Broker-{int(time.time())}"
mlflow.set_tag("mlflow.runName", run_name)

logging_run("MLflow complete")

app = FastAPI()
nest_asyncio.apply()

app.mount("/static", StaticFiles(directory="templates"), name="static")
templates = Jinja2Templates(directory='templates')

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


# -- must set with database
model_A_url = 'http://model-app.mlsd-sentimovie-test.svc'
model_B_url = 'http://model-app-b.mlsd-sentimovie-test.svc'
# model_A_url = 'http://0.0.0.0:8001'
# model_B_url = 'https://model-app-b.darkube.app'
# output_other = []


def monitor_hardware():
    cpu_percent = psutil.cpu_percent() 
    memory_percent = psutil.virtual_memory().percent
    return cpu_percent, memory_percent


def logging_mlflow(start_time, item=0):
    latency = time.time() - start_time
    logging_run(f"Latency_{item}: {latency:.5f}s")
    cpu_percent, memory_percent = monitor_hardware()
    try:
        mlflow.log_metric(f"Latency_{item}", latency)
        mlflow.log_metric(f"CPU_Usage_{item}", cpu_percent) 
        mlflow.log_metric(f"Memory_Usage_{item}", memory_percent)
    except Exception as e:
        logging_run(f"MLFLOW not working")
        logging_run(e)


async def model_inference(data, model_url, model_main=True):
    try:
        start_time = time.time()
        logging_run(data)
        response = requests.post(f"{model_url}/predict", json=data)
        logging_run(f"{model_url}: {response.status_code}")
        if model_url is model_A_url:
            logging_mlflow(start_time, "main")
        else:
            logging_mlflow(start_time, "other")
        if model_main:
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code)
        return response.json()
    except Exception as e:
        logging_run(f"The model \"{model_url}\" not working")
        print(e)
        # raise HTTPException(status_code=500, detail="The model not working")
        return None


async def model_explanable(data, model_url, model_main=True):
    try:
        start_time = time.time()
        logging_run(data)
        response = requests.post(f"{model_url}/explainable", json=data)
        logging_run(f"{model_url}: {response.status_code}")
        if model_url is model_A_url:
            logging_mlflow(start_time, "main")
        else:
            logging_mlflow(start_time, "other")
        if model_main:
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code)
        return response.json()
    except Exception as e:
        logging_run(f"The model \"{model_url}\" not working")
        print(e)
        # raise HTTPException(status_code=500, detail="The model not working")
        return None


async def save_to_db(response, model="A"):
    try:
        db = dbrepo.DB()
        if 'detail' in response:
            texts = []
            for res in response['detail']:
                texts.append(dbrepo.Text_record(
                    text = res['text'], 
                    sentiment = 1 if res['sentiment'] == "Positive" else 0,
                    probability = res['probability'], 
                    model = model)
                )
            batch_record = dbrepo.Batch_record(is_done=True, texts=texts)
            db.add_batch_req(db, batch_record)
        else:
            t = dbrepo.Text_record(
                text = response['text'],
                sentiment = 1 if response['sentiment'] == "Positive" else 0,
                probability = response['probability'],
                model = model
            )
            db.add_text(t)
    except Exception as e:
        print(e)


async def call_model_B(input_data, model_B_url):
    response = await model_inference(input_data, model_B_url, False)
    await save_to_db(response, model="B")


@app.post("/predict")
async def predict(data: InputJson):
    start_time0 = time.time()
    logging_run(data.texts)
    input_data = {"texts": data.texts}
    # -- base server inference
    try:
        response = await model_inference(input_data, model_A_url, True)
        await save_to_db(response, model="A")
    except:
        raise HTTPException(status_code=response.status_code)
    
    # -- other servers inference
    try:
        asyncio.create_task(call_model_B(input_data, model_B_url))
    except:
        logging_run("model B not answaring")

    logging_mlflow(start_time0, "total")

    return response


@app.get('/', response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.post("/predict/explainable", response_class=HTMLResponse)
async def explainable(request: Request, text: Annotated[str, Form()]):
    start_time0 = time.time()
    logging_run(text)
    input_data = {"texts": [text]}
    # -- base server inference
    try:
        response = await model_explanable(input_data, model_A_url, True)
        await save_to_db(response, model="A")
    except:
        templates.TemplateResponse("notfound.html", {"request":request})
    
    # -- other servers inference
    try:
        asyncio.create_task(call_model_B(input_data, model_B_url))
    except:
        logging_run("model B not answaring")

    logging_mlflow(start_time0, "total")
    result = {
        "request": request, 
        "text": text,
        "sentiment": f'<p class="text-danger">{response["sentiment"]} ({response["probability"]})</p>' if response['sentiment'] == "Positive" else f'<p class="text-primary">{response["sentiment"]} ({response["probability"]})</p>',
        "diagram":response['diagram'].replace("\"", "\"").replace("\n", ""),
        }

    return templates.TemplateResponse("result.html", result)


# @app.post("/texts", response_class=HTMLResponse)
# async def texts_list(request: Request, 
#                   skip: Annotated[int or None, Form()] = 0, 
#                   limit: Annotated[int or None, Form()] = 10, 
#                   model: Annotated[str or None, Form()] = "A&B"):
#     try:
#         db = dbrepo.DB()
#         texts = await db.get_texts(skip=skip, limit=limit, model=model)
#         return templates.TemplateResponse("text_list.html", {"request": request, "texts": texts})
#     except:
#         raise HTTPException(status_code=400, detail="Item not found")


async def get_list_of_texts(skip: int = 0, limit: int = 10, model: str = "A&B", text=""):
    try:
        db = dbrepo.DB()
        texts = await db.get_texts(skip=skip, limit=limit, model=model, text=text)
        return texts
    except:
        raise Exception("Item not found")


@app.get("/texts", response_class=HTMLResponse)
async def texts_list(request: Request):
    try:
        texts = await get_list_of_texts()
        return templates.TemplateResponse("text_list.html", {"request": request, "texts": texts})
    except:
        templates.TemplateResponse("notfound.html", {"request":request})
    

@app.post("/texts", response_class=HTMLResponse)
async def texts_list(request: Request, 
                     skip: Annotated[int, Form()] = 0, 
                     limit: Annotated[int, Form()] = 10, 
                     model: Annotated[str, Form()] = "A&B",
                     text: Annotated[str, Form()] = ""):
    try:
        texts = None
        if model.upper() != "A" and model.upper() != "B":
            texts = await get_list_of_texts(skip=skip, limit=limit, model="A&B", text=text)
        else:
            texts = await get_list_of_texts(skip=skip, limit=limit, model=model.upper(), text=text)
        return templates.TemplateResponse("text_list.html", {"request": request, "texts": texts})
    except:
        templates.TemplateResponse("notfound.html", {"request":request})
