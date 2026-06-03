# ----- base image -----
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    MODEL_DIR=/app/model

WORKDIR /app

# system deps needed by xgboost / lightgbm at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ----- python deps (cached separately from code) -----
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ----- application code -----
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# ----- bake in one trained model -----
# swap the run id below if you promote a different model
ARG RUN_ID=f3f610061e3e4b78bcdd34ee427d0806
ARG EXPERIMENT_ID=850886219068847109
COPY mlruns/${EXPERIMENT_ID}/${RUN_ID}/artifacts/model/ /app/model/

EXPOSE 7860

CMD ["python", "src/app/app.py"]
