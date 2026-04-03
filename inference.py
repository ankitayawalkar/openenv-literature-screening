import requests
import os

# ---------- ENV VARIABLES ----------
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline-model")
HF_TOKEN = os.getenv("HF_TOKEN")  # no default

TASK_NAME = "literature-screening"
BENCHMARK = "openenv"

# ---------- LOGGING ----------
def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True
    )

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True
    )

# ---------- SIMPLE BASELINE LOGIC ----------
def decide_action(abstract):
    if "review" in abstract.lower():
        return "exclude", "E1"
    return "include", None

# ---------- MAIN ----------
def run_inference():
    step_count = 0
    rewards = []
    predictions = []
    success = False

    log_start()

    try:
        # Reset environment
        requests.post(f"{API_BASE_URL}/reset")

        while True:
            # Get state
            res = requests.get(f"{API_BASE_URL}/state")
            state = res.json()

            if state is None:
                break

            abstract = state.get("abstract", "")

            # Decide action
            label, reason = decide_action(abstract)

            action_payload = {"action": label}

            # Call step
            step_res = requests.post(f"{API_BASE_URL}/step", json=action_payload)
            result = step_res.json()

            observation = result.get("observation")
            reward = result.get("reward", 0.0)
            done = result.get("done", False)

            rewards.append(reward)
            step_count += 1

            # Store prediction
            pred = {"label": label}
            if label == "exclude":
                pred["reason"] = reason

            predictions.append(pred)

            # Log step
            log_step(
                step=step_count,
                action=label,
                reward=reward,
                done=done,
                error=None
            )

            if done:
                break

        # Call grader
        grade_res = requests.post(f"{API_BASE_URL}/grader", json=predictions)
        score = grade_res.json().get("score", 0)

        success = score > 0

    except Exception as e:
        print(f"[DEBUG] Error: {e}", flush=True)

    finally:
        log_end(success=success, steps=step_count, rewards=rewards)


if __name__ == "__main__":
    run_inference()
