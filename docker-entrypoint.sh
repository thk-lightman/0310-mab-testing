#!/bin/bash
set -e

# Create required directories if they don't exist
mkdir -p /app/mab_web_app/logs
mkdir -p /app/mab_web_app/data
mkdir -p /app/mab_web_app/static/images

# Set correct permissions
chmod -R 755 /app/mab_web_app/static
chmod -R 777 /app/mab_web_app/logs
chmod -R 777 /app/mab_web_app/data

# If .env doesn't exist, copy from example
if [ ! -f .env ] && [ -f .env.example ]; then
  echo "Creating .env from .env.example"
  cp .env.example .env
fi

# Run command passed to entrypoint
exec "$@" 