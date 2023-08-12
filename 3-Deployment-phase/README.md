# Sentimovie FastAPI Application Documentation

This documentation provides an overview of the `SentiMovie` FastAPI application codebase. This application is designed to provide sentiment analysis and explanation for given text inputs using multiple models.

## Table of Contents
* [API](broker/readme.md)
    * [Introduction](broker/readme.md#Introduction)
    * [Installation](broker/readme.md#Installation)
    * [Usage](broker/readme.md#Usage)
    * [Endpoints](broker/readme.md#Endpoints)
        * [Predict Sentiment](broker/readme.md#Predict)
        * [Explainable Sentiment](broker/readme.md#explainable)
        * [Retrieve Texts](broker/readme.md#texts-post)
* [Database](broker/readme.md#db)
    * [Tables](broker/readme.md#db-tables)
    * [Database Interaction](broker/readme.md#db-interaction)
* [Model](model/readme.md)
    * [FastAPI model service](model/readme.md#fastapi)
        * [Installation](model/readme.md#Installation)
        * [Usage](model/readme.md#Usage)
        * [Endpoints](model/readme.md#endpoints)
            * [Predict](model/readme.md#predict)
            * [Explainable](model/readme.md#Explainable)
        * [Monitoring](model/readme.md#Monitoring)
    * [Back model library (MyClassifier)](model/readme.md#myclassifier)
        * [Features](model/readme.md#Features)
        * [Getting](model/readme.md#Getting)
        * [Methods](model/readme.md#Methods)
        * [Dependencies](model/readme.md#Dependencies)
        * [Note](model/readme.md#Note)
    * [Unit-testing](unittest/readme.md)



# Test, Build, and Deploy Images CI/CD Pipeline

This repository contains a CI/CD pipeline for testing, building, and deploying images using GitHub Actions. The pipeline includes the following stages: test, build, and deployment of multiple components.

## Workflow Overview

This workflow is triggered on `push` events to the repository. It consists of three main stages: `test`, `build`, and `darkube_deploy`.

### Test Stage

The `test` stage is responsible for running unit tests for the application. It sets up the required environment and dependencies, installs dependencies from the specified requirements files, and then runs the unit tests.

### Build Stage

The `build` stage is dependent on the successful completion of the `test` stage. It performs the following actions:
- Checks out the repository code.
- Builds and pushes Docker images for various components:
  - Model A
  - MLflow
  - Model B
  - Broker

The images are built using the code from the appropriate directories and tagged with the short SHA of the commit. The built images are pushed to the specified Docker registry.

### Deploy Stage (darkube_deploy)

The `darkube_deploy` stage is dependent on the successful completion of the `build` stage. It deploys the built Docker images using the Darkube CLI. The deployment is performed for the following components:
- MLflow
- Model
- Model B
- Broker

The deployment is triggered using the Darkube CLI, with the specified `ref`, `token`, `app-id`, `image-tag`, and other parameters. The deployment process is executed for each component separately.

## Environment Variables

The following environment variables are used in the workflow:

- `REGISTRY`: The Docker registry URL.
- `REGISTRY_PASSWORD`: The password for authenticating with the Docker registry.
- `REGISTRY_USER`: The username for authenticating with the Docker registry.
- `APP_NAME_MODEL`, `APP_NAME_MLFLOW`, `APP_NAME_MODEL_B`, `APP_NAME_API`: Names of the different application components.
- `DARKUBE_DEPLOY_TOKEN_MLFLOW`, `DARKUBE_DEPLOY_TOKEN_MODEL`, `DARKUBE_DEPLOY_TOKEN_MODEL_B`, `DARKUBE_DEPLOY_TOKEN_API`: Darkube deployment tokens for different components.
- `DARKUBE_APP_ID_MLFLOW`, `DARKUBE_APP_ID_MODEL`, `DARKUBE_APP_ID_MODEL_B`, `DARKUBE_APP_ID_API`: Darkube application IDs for different components.

## How to Use

To use this CI/CD pipeline for your own project, follow these steps:

1. Update the workflow file `.github/workflows/cicd.yml` as needed.
2. Configure the environment variables according to your project requirements.
3. Set up Darkube CLI on your deployment environment if you haven't already.
4. Push changes to your repository.

The workflow will automatically trigger on each `push` event to the repository. It will test, build, and deploy the specified Docker images based on the configured components.

Please ensure that you review and adjust the pipeline configuration according to your project's specific needs before using it in a production environment.

**Note:** This README provides an overview of the CI/CD pipeline. For detailed information on each stage and the deployment process, refer to the workflow definition file (`ci-cd-pipeline.yml`) and the respective deployment scripts in your repository.