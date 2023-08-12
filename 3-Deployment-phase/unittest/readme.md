# Sentiment Analysis Model Unit Testing in CI/CD Pipeline

This repository contains code for unit testing a sentiment analysis model as part of a Continuous Integration and Continuous Deployment (CI/CD) pipeline. The unit tests ensure that the sentiment analysis model is functioning correctly and producing accurate results. The tests cover various scenarios, including single predictions, batch predictions, and explainable predictions.

## Getting Started

These instructions will guide you through setting up the unit testing environment and running the tests.

### Prerequisites

Before you begin, ensure you have the following installed:

- Python (>= 3.11)
- [Asyncio](https://docs.python.org/3/library/asyncio.html) library

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Tests

To run the unit tests, execute the following command in your terminal:

```bash
python test_sentiment_analysis.py
```

## Description

The unit tests are performed using the `testing` function defined in the `test_sentiment_analysis.py` file. The function utilizes the sentiment analysis model to perform various tests and checks the accuracy of the predictions. The tests cover the following scenarios:

1. Single Prediction:
   - Test sentiment prediction for the input text "I hate this movie".
   - Verify if the predicted sentiment is "Negative" and the probability is above 0.7.

2. Single Prediction (Positive):
   - Test sentiment prediction for the input text "I like this movie".
   - Verify if the predicted sentiment is "Positive" and the probability is above 0.7.

3. Batch Prediction:
   - Test sentiment predictions for a batch of texts containing both negative and positive sentiments.
   - Verify if the predictions match the expected sentiments for each input text and if the probabilities are above 0.7.

4. Explainable Prediction (Positive):
   - Test explainable sentiment prediction for the input text "I like this movie".
   - Verify if the predicted sentiment is "Positive" and the probability is above 0.7.

5. Explainable Prediction (Negative):
   - Test explainable sentiment prediction for the input text "I hate this movie".
   - Verify if the predicted sentiment is "Negative" and the probability is above 0.7.

If any of the tests fail, an exception will be raised, indicating that at least one test has failed.

## CI/CD Pipeline

The unit tests in this repository are designed to be integrated into a CI/CD pipeline. The pipeline can be configured to automatically execute the tests whenever changes are made to the sentiment analysis model or its dependencies. This helps ensure that any changes to the model or codebase do not introduce regressions or break existing functionality.