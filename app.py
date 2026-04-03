from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import json

from models import State, Action, Observation, StepResponse, Prediction
from typing import List

app = FastAPI()

# ---------- ROOT ----------
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>OpenEnv: Clinical Literature Screening</h2>
    <p>API is running successfully.</p>

    <h3>Available Endpoints:</h3>
    <ul>
        <li><a href="/docs">API Documentation (Swagger UI)</a></li>
    </ul>

    <p>Use /reset, /state, /step, /grader to interact programmatically.</p>
    """

# ---------- LOAD DATA ----------
with open("data.json") as f:
    ORIGINAL_DATA = json.load(f)

data = []
index = 0

# ---------- RESET ----------
@app.post("/reset")
def reset():
    global data, index
    data = ORIGINAL_DATA.copy()
    index = 0
    return {"status": "reset successful"}

# Optional fallback
@app.get("/reset")
def reset_get():
    return {"message": "Use POST /reset"}

# ---------- STATE ----------
@app.get("/state", response_model=State)
def state():
    global index, data

    if index >= len(data):
        return None

    paper = data[index]

    return State(
        title=paper["title"],
        abstract=paper["abstract"]
    )

# ---------- STEP (FIXED FOR OPENENV) ----------
@app.post("/step", response_model=StepResponse)
def step(action: Action):
    global index, data

    if index >= len(data):
        return StepResponse(
            observation=None,
            reward=0.0,
            done=True,
            info={}
        )

    paper = data[index]

    observation = Observation(
        title=paper["title"],
        abstract=paper["abstract"]
    )

    # reward logic
    reward = 1.0 if action.action == paper["label"] else 0.0

    index += 1
    done = index >= len(data)

    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info={}
    )

# ---------- TASKS ----------
@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"name": "easy", "description": "Binary classification"},
            {"name": "medium", "description": "Classification + reason"},
            {"name": "hard", "description": "High precision with penalties"},
        ]
    }

# ---------- BASELINE ----------
@app.get("/baseline")
def baseline():
    return {"message": "Baseline not implemented"}

# ---------- GRADER FUNCTION ----------
def grade(predictions, data, task="medium"):
    score = 0

    for pred, paper in zip(predictions, data):

        if task == "easy":
            if pred.get("label") == paper.get("label"):
                score += 1

        elif task == "medium":
            if pred.get("label") == paper.get("label"):
                score += 0.5
                if pred.get("label") == "exclude":
                    if pred.get("reason") == paper.get("exclusion_code"):
                        score += 0.5

        elif task == "hard":
            if pred.get("label") == paper.get("label"):
                score += 0.5
                if pred.get("label") == "exclude":
                    if pred.get("reason") == paper.get("exclusion_code"):
                        score += 0.5
            else:
                score -= 1

    final_score = max(0, score / len(data))
    return final_score

# ---------- GRADER ENDPOINT ----------
@app.post("/grader")
def grader(predictions: List[Prediction]):
    try:
        preds = [p.dict() for p in predictions]
        score = grade(preds, ORIGINAL_DATA)
        return {"score": score}
    except Exception as e:
        return {"error": str(e)}
