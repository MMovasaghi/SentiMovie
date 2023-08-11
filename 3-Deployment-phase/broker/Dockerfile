FROM python:3.11.4
WORKDIR /app
COPY requirements.txt .
COPY . .
RUN pip install -r requirements.txt

CMD ["uvicorn", "broker:app", "--host", "0.0.0.0", "--port", "80"]