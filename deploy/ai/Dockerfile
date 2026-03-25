FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY packages ./packages
COPY services ./services
COPY artifacts ./artifacts

RUN pip install --no-cache-dir ./packages/dermassist && \
    pip install --no-cache-dir ./packages/skin_gatekeeper

EXPOSE 8000

CMD ["uvicorn", "services.inference_api.app:app", "--host", "0.0.0.0", "--port", "8000"]