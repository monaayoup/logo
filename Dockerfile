FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY Requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r Requirements.txt

# Copy the rest of the project
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "8000"]