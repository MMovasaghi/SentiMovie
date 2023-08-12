# Senti-Movie API

## Table of Contents
* [Introduction](#Introduction)
* [Installation](#Installation)
* [Usage](#Usage)
* [Endpoints](#Endpoints)
    * [Predict Sentiment](#Predict)
    * [Explainable Sentiment](#explainable)
    * [Retrieve Texts](#texts-post)
* [Database](#db)
    * [Tables](#db-tables)
    * [Database Interaction](#db-interaction)

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


## Database <a name="db"></a>

We utilize the PostgreSQL database as our chosen backend, seamlessly interacting with it through the SQLAlchemy package. This combination offers several advantages. SQLAlchemy facilitates efficient communication and management of database operations, abstracting complexities and promoting robust code development. PostgreSQL, renowned for its reliability, scalability, and powerful features, ensures data integrity and responsiveness, making it an ideal choice for our application's database needs.


## Database Tables Documentation <a name="db-tables"></a>

This document provides an overview of the two database tables: `Text` and `Batch_Request`.

### Text Table

The `Text` table is designed to store individual text entries along with their sentiment analysis results. Each record represents a single text and its corresponding attributes.

#### Table Structure

| Column Name  | Data Type   | Description                                                |
|--------------|-------------|------------------------------------------------------------|
| id           | Integer     | Primary key for the text entry.                            |
| text         | String      | The actual text content.                                   |
| sentiment    | Integer     | Sentiment label for the text (0: Negative, 1: Positive).  |
| probability  | Float       | Probability score of the sentiment prediction.             |
| model        | String      | The identifier of the model used for analysis (default: "A"). |
| batch_id     | Integer     | Foreign key referencing the batch request id (default: None). |
| created_at   | DateTime    | Timestamp of when the text entry was created.              |
| updated_at   | DateTime    | Timestamp of when the text entry was last updated (default: None). |

#### Relationships

The `Text` table has a relationship with the `Batch_Request` table, establishing a one-to-many relationship. A single batch request can contain multiple texts.

### Batch_Request

The `Batch_Request` table is used to manage batch processing of text entries. It represents a collection of texts submitted for analysis as a batch.

#### Table Structure

| Column Name  | Data Type   | Description                                                |
|--------------|-------------|------------------------------------------------------------|
| id           | Integer     | Primary key for the batch request entry.                   |
| is_done      | Boolean     | Indicates whether the batch processing is completed.       |
| created_at   | DateTime    | Timestamp of when the batch request was created.           |
| updated_at   | DateTime    | Timestamp of when the batch request was last updated (default: None). |

#### Relationships

The `Batch_Request` table has a one-to-many relationship with the `Text` table. Each batch request can contain multiple texts.

---

Please note that this documentation provides a high-level overview of the table structures and relationships. The actual implementation may include additional details or modifications based on your specific use case and database design.


# Database Interaction Module Documentation <a name="db-interaction"></a>

This documentation outlines the functionalities of the `DB` class, which is responsible for interacting with the database using the SQLAlchemy ORM and Pydantic models.

## `DB` Class

The `DB` class encapsulates methods to perform various database operations, including adding batch requests and texts, retrieving texts and batch requests, and obtaining pending batch requests.

### Class Initialization

```python
db = DB()
```

Initialize an instance of the `DB` class to begin interacting with the database.

#### `add_batch_req` Method
```python
def add_batch_req(self, batch_record: Batch_record) -> db_models.Batch_Request:
    """
    Add a batch request and its associated texts to the database.

    Parameters:
    - batch_record (Batch_record): A Pydantic model representing the batch request.

    Returns:
    - db_models.Batch_Request: The added batch request instance.
    """
```

This method adds a batch request and its associated texts to the database. If the batch request is marked as done, sentiment analysis results (sentiment, probability, model) for individual texts are also added.


#### `add_text` Method

```python
def add_text(self, text_record: Text_record) -> db_models.Text:
    """
    Add a text record to the database.

    Parameters:
    - text_record (Text_record): A Pydantic model representing the text record.

    Returns:
    - db_models.Text: The added text record instance.
    """
```

This method adds a single text record to the database, including sentiment analysis results if available.

#### `get_texts` Method
```python
async def get_texts(self, skip: int = 0, limit: int = 100, model: str = "A&B", text="") -> List[db_models.Text]:
    """
    Retrieve a list of text records based on specified criteria.

    Parameters:
    - skip (int): Number of records to skip.
    - limit (int): Maximum number of records to retrieve.
    - model (str): Model identifier (default: "A&B").
    - text (str): Text content to search for (default: "" for all texts).

    Returns:
    - List[db_models.Text]: A list of retrieved text record instances.
    """
```
This method retrieves a list of text records based on specified criteria, such as model and text content.

#### `get_batch_req` Method
```python
async def get_batch_req(self, skip: int = 0, limit: int = 100) -> List[db_models.Batch_Request]:
    """
    Retrieve a list of batch requests.

    Parameters:
    - skip (int): Number of records to skip.
    - limit (int): Maximum number of records to retrieve.

    Returns:
    - List[db_models.Batch_Request]: A list of retrieved batch request instances.
    """
```

This method retrieves a list of batch requests.


#### `get_batch_not_done` Method
```python
async def get_batch_not_done(self) -> Optional[db_models.Batch_Request]:
    """
    Retrieve the oldest pending batch request.

    Returns:
    - Optional[db_models.Batch_Request]: The oldest pending batch request instance or None if none are pending.
    """
```

This method retrieves the oldest pending batch request, if any.


Please note that this documentation provides an overview of the functionalities of the `DB` class and its methods. The actual implementation may include additional details or modifications based on your specific use case and database design.