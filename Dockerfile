FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full package into /app/supportdesk_env/
# This makes it importable as `supportdesk_env.*`
COPY . /app/supportdesk_env/

# Add /app to PYTHONPATH so `import supportdesk_env` works
ENV PYTHONPATH=/app

# Hugging Face Spaces use port 7860
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "supportdesk_env.server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
