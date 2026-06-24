# Single image used by both the API and the dashboard services (compose picks
# the command). Keeps the build cache shared and the footprint small.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY dashboard ./dashboard

EXPOSE 8000 8501

# Default: serve the API. The dashboard service overrides this in compose.
CMD ["uvicorn", "delivery_delay.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
