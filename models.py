from pydantic import BaseModel
from typing import Optional, Dict, Any


# ---------- ACTION ----------
class Action(BaseModel):
    action: str


# ---------- STATE ----------
class State(BaseModel):
    title: str
    abstract: str


# ---------- OBSERVATION (same as state for this env) ----------
class Observation(BaseModel):
    title: str
    abstract: str


# ---------- STEP RESPONSE ----------
class StepResponse(BaseModel):
    observation: Optional[Observation]
    reward: float
    done: bool
    info: Dict[str, Any] = {}


# ---------- PREDICTION ----------
class Prediction(BaseModel):
    label: str
    reason: Optional[str] = None
