FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/input /app/output /app/models && \
    test -f /app/models/model.joblib || \
    (echo "ERROR: /app/models/model.joblib was not found. Run 'python train_model.py' before building the Docker image." >&2; exit 1)

VOLUME ["/app/input", "/app/output"]

CMD ["python", "-m", "src.pipeline"]
