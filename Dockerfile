# Use official Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Make sure the app directory is in PYTHONPATH
ENV PYTHONPATH=/app

# Expose port (Railway uses PORT env var)
EXPOSE 5000

# Run Gunicorn with:
# - 1 worker
# - 2 threads per worker
# - debug logging
# - explicit working directory
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--log-level", "debug", "--chdir", "/app", "wsgi:app"]