FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.prod.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.prod.txt

COPY packages ./packages
COPY services ./services
COPY artifacts ./artifacts

RUN pip install --no-cache-dir ./packages/dermassist && \
    pip install --no-cache-dir ./packages/skin_gatekeeper

EXPOSE 8000

CMD ["uvicorn", "services.inference_api.app:app", "--host", "0.0.0.0", "--port", "8000"]