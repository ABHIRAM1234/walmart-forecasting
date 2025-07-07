# Dockerfile for the prediction function
FROM python:3.9-slim
WORKDIR /app

# Copy all local files (main.py, requirements.txt) into the container.
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the command that Google Cloud Run uses to start the server.
CMD ["functions-framework", "--target=run_batch_forecast", "--port=8080"]