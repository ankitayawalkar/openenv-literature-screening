from pydantic import BaseModel
from typing import Optional

class Action(BaseModel):
    action: str

class State(BaseModel):
    title: str
    abstract: str

class Prediction(BaseModel):
    label: str
    reason: Optional[str] = None
