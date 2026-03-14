FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python download_model.py || echo "Model will be downloaded on first request"

CMD gunicorn emotion_api_server:app --bind 0.0.0.0:${PORT:-5050} --workers 1 --timeout 120
