FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit frontend code
COPY . .

EXPOSE 8501

# Command to run Streamlit frontend
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]