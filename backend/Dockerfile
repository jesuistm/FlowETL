FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend code into the container
COPY . .

# Create logs directory that will be mounted as a volume
RUN mkdir -p /app/logs

# Expose port 8000 so FastAPI can receive requests
EXPOSE 8000

# Command to run the FastAPI backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]