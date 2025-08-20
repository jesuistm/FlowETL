from typing import List, Any, Dict
from pydantic import BaseModel, Field

class FrontEndRequest(BaseModel):
    abstraction: str = Field(description="JSON representation of the abstracted dataset sample") 
    task_description: str = Field(description="User task description in natural language")

class Plan(BaseModel):
    """The transformation plan to be parsed and applied onto the abstraction"""
    id : str = Field(description="Plan identifier")
    task_summary : str = Field(description="User supplied data engineering task description summary")
    #source_dataset : str = Field(description="Name of the dataset targeted by the plan")
    #source_schema : Dict[str, Any] = Field(description="Inferred schema for the input abstraction, using native Pandas data types")
    pipeline : List[Any] = Field(description="FlowETL functions to be applied onto the dataset")