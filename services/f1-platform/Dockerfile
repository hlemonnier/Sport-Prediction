FROM python:3.12-slim

WORKDIR /app
COPY services/f1-platform/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY services/f1-platform/f1_platform ./f1_platform

EXPOSE 8001
CMD ["uvicorn", "f1_platform.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8001"]
