from typing import List, Any, Dict, Optional
from pydantic import BaseModel, Field

class DataEngineerRequest(BaseModel):
    abstraction: str = Field(description="JSON representation of the abstracted dataset sample") 
    source_dataset: str = Field(description="Name of the source dataset")
    task_description: str = Field(description="User task description in natural language")

class DataAnalystRequest(BaseModel):
    abstraction: str = Field(description="JSON representation of the abstracted dataset sample") 
    query: str = Field(description="User data analysis query in natural language")

class Plan(BaseModel):
    """The transformation plan to be parsed and applied onto the abstraction"""
    id : str = Field(description="Plan identifier")
    task_summary : str = Field(description="User supplied data engineering task description summary")
    source_dataset : str = Field(description="Name of the dataset targeted by the plan")
    source_schema : Dict[str, Any] = Field(description="Inferred schema for the input abstraction, using native Pandas data types")
    pipeline : List[Any] = Field(description="FlowETL functions to be applied onto the dataset")

class Analysis(BaseModel):
    """The data analyis response model from the data analysis agent"""
    analysis_code : str = Field(description="Pandas code to carry out the analysis task")
    plot_code : Optional[str] = Field(None, description="Matplotlib code with labels and formatting") 