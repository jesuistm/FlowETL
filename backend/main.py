from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json

# imports for type hinting
from typing import Dict, Any

app = FastAPI()

# configure API middleware
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class Request(BaseModel):
    abstraction: str  # json representation of the dataframe
    task_description: str # task requirements in natural language

@app.post("/")
async def process_abstraction(request: Request) -> Dict[str, Any]:
    try:
        # reconstruct DataFrame from JSON
        abstraction = pd.DataFrame(json.loads(request.abstraction))
        
        size = len(abstraction) 

        if size < 1: 
            raise HTTPException(status_code=400, detail="No instances found in the dataset uploaded")
        

        """
        TODO
        - given the task description, return a validated json plan to be parsed and applied


        - the architecture for this task will involve 2 agents:
            1. the distiller agent takes in a natural language prompt and an optional feedback. This agent will
            then return a parseable json object to instruct the data preparation process

            2. a validator agent which takes in a sample of the abstraction/multiple samples (same idea as cross-validation in ML), a 
            json plan created by agent 1. This agent then returns feedback to inform agent 1. If the plan is deemed correct, then the 
            feedback will be none.
        """


        return { "abstraction_length": size }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process data: {str(e)}")