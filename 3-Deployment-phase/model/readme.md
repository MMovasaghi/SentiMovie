# Senti-Movie Model

### Table of Contents:
* [FastAPI model service](#fastapi)
  * [Installation](#Installation)
  * [Usage](#Usage)
  * [Endpoints](#Endpoints)
    * [Predict](#Predict)
    * [Explainable](#Explainable)
  * [Monitoring](#Monitoring)
* [Back model library (MyClassifier)](#myclassifier)
  * [Features](#Features)
  * [Getting](#Getting)
  * [Methods](#Methods)
  * [Dependencies](#Dependencies)
  * [Note](#Note)

<br><br>

<a name="fastapi"></a>
# Senti-Movie FastAPI Service for Model

This repository contains a FastAPI service for sentiment analysis using a pre-trained model. The service provides two endpoints: `/predict` for sentiment prediction and `/explainable` for explainable sentiment prediction. It utilizes FastAPI for handling HTTP requests, PyTorch for the sentiment analysis model, and MLflow for logging and monitoring.


## Installation <a name="Installation"></a>

1. Clone this repository to your local machine.
2. Install the required dependencies using the 

    ```bash
    pip install -r requirements.txt
    ```


## Usage <a name="Usage"></a>

1. Run the FastAPI service using the following command:

    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    This will start the FastAPI service on http://localhost:8000.

2. Send POST requests to the following endpoints using a tool like curl, httpie, or any API client:

    * `/predict`: This endpoint accepts a list of texts and returns sentiment predictions.

    * `/explainable`: This endpoint accepts a single text and returns an explainable sentiment prediction.


## Endpoints <a name="Endpoints"></a>
### Predict Sentiment <a name="Predict"></a>
* Endpoint: `/predict`
* Method: `POST`
* Request Payload:

```json
{
  "texts": ["Text 1", "Text 2", "..."]
}
```
* Response: The response contains sentiment predictions for the input texts.

### Explainable Sentiment Prediction <a name="Explainable"></a>
* Endpoint: `/explainable`
* Method: `POST`
* Request Payload:

```json
{
  "texts": ["Text"]
}
```
* Response: The response contains an explainable sentiment prediction for the input text.


## Monitoring and Logging <a name="Monitoring"></a>
The service utilizes MLflow for monitoring and logging. Hardware metrics (CPU and memory usage) and latency metrics are logged for each prediction request.


<br>
<br>


# MyClassifier: Sentiment Analysis and Explanation <a name="myclassifier"></a>

The `MyClassifier` class is a Python utility that leverages pre-trained transformer-based models for sentiment analysis and SHAP (SHapley Additive exPlanations) for generating model explanations. This utility provides a simple interface for performing sentiment analysis on individual text inputs, batch prediction, and explainable prediction along with SHAP-based explanations.

## Features <a name="Features"></a>

- Perform sentiment analysis on individual text inputs.
- Conduct batch sentiment analysis prediction on a list of texts.
- Generate explainable predictions along with SHAP-based model explanations.

## Getting Started <a name="Getting"></a>

1. Install the required libraries:

    ```bash
    pip install transformers shap
    ```

2. Import the `MyClassifier` class:

    ```python
    from my_classifier import MyClassifier
    ```

3. Initialize an instance of `MyClassifier`:

    ```python
    classifier = MyClassifier()
    ```

4. Use the available methods for sentiment analysis and explanations:

    ```python
    # Perform single prediction
    single_text = "This is a great product."
    single_result = await classifier.single_prediction(single_text)

    # Perform batch prediction
    batch_texts = ["I love this movie!", "This movie was terrible."]
    batch_results = await classifier.batch_prediction(batch_texts)

    # Perform explainable prediction
    explain_text = "This book was disappointing."
    explain_result = await classifier.explainable_prediction(explain_text)
    ```

## Methods <a name="Methods"></a>

### `single_prediction(text: str) -> dict`

Perform sentiment analysis prediction on a single text.

- Input:
  - `text` (str): The text input for sentiment analysis.

- Output:
  - A dictionary containing sentiment analysis result for the input text.

### `batch_prediction(texts: list) -> dict`

Perform batch sentiment analysis prediction on a list of texts.

- Input:
  - `texts` (list): A list of text inputs for sentiment analysis.

- Output:
  - A dictionary containing sentiment analysis results for the batch.

### `explainable_prediction(text: str) -> dict`

Perform explainable sentiment analysis prediction on a single text.

- Input:
  - `text` (str): The text input for sentiment analysis.

- Output:
  - A dictionary containing sentiment analysis result and SHAP explanation for the input text.

## Dependencies <a name="Dependencies"></a>

- [Hugging Face Transformers](https://github.com/huggingface/transformers): For the sentiment analysis pipeline.
- [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap): For generating model explanations.


## Note <a name="Note"></a>

This utility assumes that you have properly installed the required libraries (`transformers` and `shap`) and have access to the pre-trained model used for sentiment analysis. Make sure to provide the necessary model path or identifier before using the `MyClassifier` class.
