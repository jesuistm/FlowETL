#!/bin/bash

echo "[INFO] - Running FlowETL in 3rd party mode"

source venv/Scripts/activate

echo "[INFO] - Activated virtual environment" 

cd backend

pip install -r requirements.txt

echo "[INFO] - Installed all required backend Python packages" 

uvicorn main:app --reload --host 0.0.0.0 --port 8000 & BACKEND_PID=$!

echo "[INFO] - Backend API started" 

cd ../frontend

pip install -r requirements.txt

echo "[INFO] - Installed all required frontend Python packages" 

streamlit run dashboard.py & FRONTEND_PID=$!

echo "[INFO] - Frontend dashboard started" 

trap "kill $BACKEND_PID $FRONTEND_PID; deactivate; exit" INT

wait
