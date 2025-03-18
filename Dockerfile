FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entrypoint script
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p mab_web_app/logs mab_web_app/data mab_web_app/static/images && \
    chmod -R 755 mab_web_app/static && \
    chmod -R 777 mab_web_app/logs mab_web_app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Expose port
EXPOSE 5000

# Use entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run app when the container launches
CMD ["python", "app.py"] 