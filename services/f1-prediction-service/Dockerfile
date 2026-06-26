FROM python:3.12-slim

WORKDIR /app
COPY services/f1-prediction-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY packages ./packages
COPY services/f1-prediction-service/f1_prediction_service ./f1_prediction_service
ENV PYTHONPATH=/app

EXPOSE 8002
CMD ["uvicorn", "f1_prediction_service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8002"]
