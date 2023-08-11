FROM python:3.11.4
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
COPY meta.yaml ./mlruns/0/

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]