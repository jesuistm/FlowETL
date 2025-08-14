from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json

# imports for type hinting
from typing import Dict, Any

app = FastAPI()

# configure API middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Request(BaseModel):
    abstraction: str  # json representation of the dataframe
    task_description: str


@app.post("/")
async def process_abstraction(request: Request) -> Dict[str, Any]:
    try:
        # reconstruct DataFrame from JSON
        abstraction = pd.DataFrame(json.loads(request.abstraction))
        
        size = len(abstraction) 

        if size < 1: 
            raise HTTPException(status_code=400, detail="No instances found in the dataset uploaded")
        
        return {
            "abstraction_length": size
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process data: {str(e)}")