# Senti-Movie API

## Table of Contents
* [Introduction](#Introduction)
* [Installation](#Installation)
* [Usage](#Usage)
* [Endpoints](#Endpoints)
    * [Predict Sentiment](#Predict)
    * [Explainable Sentiment](#explainable)
    * [Retrieve Texts](#texts-post)

## Introduction <a name="Introduction"></a>
The Sentimovie FastAPI application leverages FastAPI, a modern Python web framework, to provide sentiment analysis and explanation for text inputs. The application utilizes multiple machine learning models and communicates with external services for inference and explanation.

## Installation <a name="Installation"></a>
To run the Sentimovie FastAPI application, follow these steps:

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd 3-Deployment-phase/broker
    ```

2. Install the required dependencies using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application using `uvicorn`:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

The application will be accessible at http://localhost:8000.


## Usage <a name="Usage"></a>
The Sentimovie FastAPI application provides several endpoints for sentiment analysis, explanation, and text retrieval.

1. Run the FastAPI service using the following command:

    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    This will start the FastAPI service on http://localhost:8000.

2. Send `POST` requests to the following endpoints using a tool like curl, httpie, or any API client:

    * `/predict`: This endpoint accepts a list of texts and returns sentiment predictions.

    * `/predict/explainable`: This endpoint accepts a single text and returns an explainable sentiment prediction. (Results are shown as `HTML` page)

    * `/texts`: This endpoint return list of texts that are in DB. (Results are shown as `HTML` page)

3. Send `GET` requests to the following endpoints using a tool like curl, httpie, or any API client:
    * `/texts`: This endpoint return list of texts that are in DB. (Results are shown as `HTML` page)

## Endpoints <a name="#Endpoints"></a>
### Predict Sentiment <a name="#Predict"></a>
* Endpoint: `/predict`
* Method: `POST`
* Request Payload:

```json
{
  "texts": ["Text 1", "Text 2", "..."]
}
```
* Response: The response contains sentiment predictions for the input texts.

### Explainable Sentiment Prediction <a name="#explainable"></a>
* Endpoint: `/predict/explainable`
* Method: `POST`
* Request Payload:

```json
{
  "texts": ["Text"]
}
```
* Response: The response contains an explainable sentiment prediction for the input text.


### Texts <a name="#texts-post"></a>
* Endpoint: `/texts`
* Method: `POST`
* Query Parameters:
    * `skip`: Number of items to skip (default: 0).
    * `limit`: Maximum number of items to retrieve (default: 10).
    * `model`: Model type ("A", "B", "A&B" - default: "A&B").
    * `text`: Text filter (default: empty).


* Response: List of texts as `HTML` page.

### Texts <a name="#texts-get"></a>
* Endpoint: `/texts`
* Method: `GET`

* Response: List of texts as `HTML` page with `skip`=0, `limit`=10, and `model`=\"A&B\".