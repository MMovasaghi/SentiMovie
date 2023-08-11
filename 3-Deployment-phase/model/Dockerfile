FROM python:3.11.4
WORKDIR /app
COPY requirements.txt .
COPY . .
RUN pip install -r requirements.txt
RUN pip install -r requirements-2.txt

CMD ["uvicorn", "model:app", "--host", "0.0.0.0", "--port", "80"]