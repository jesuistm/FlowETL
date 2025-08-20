from typing import List, Any
from pydantic import BaseModel, Field
import uuid

class Plan(BaseModel):
    """The transformation plan to be parsed and applied onto the abstraction"""
    id : uuid.UUID = Field(default_factory=uuid.uuid4, description="Plan identifier")
    task_description : str = Field(description="User supplied data engineering task description")
    source_dataset : str = Field(description="Name of the dataset targeted by the plan")
    steps : List[Any] = Field(description="FlowETL functions to be applied onto the dataset")