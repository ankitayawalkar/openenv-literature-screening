import requests

BASE_URL = "http://127.0.0.1:8000"

def run_inference():
    print("Starting inference...")

    predictions = []

    # Reset environment
    requests.get(f"{BASE_URL}/reset")

    while True:
        # Get current paper
        res = requests.get(f"{BASE_URL}/state")
        state = res.json()

        if state is None:
            break

        title = state.get("title", "")
        abstract = state.get("abstract", "")

        # 🔍 Improved baseline logic
        text = (title + " " + abstract).lower()

        if "review" in text or "meta-analysis" in text:
            action = "exclude"
            reason = "E1"
        elif "animal" in text:
            action = "exclude"
            reason = "E2"
        else:
            action = "include"
            reason = ""

        # Store prediction
        pred = {"label": action}
        if action == "exclude":
            pred["reason"] = reason

        predictions.append(pred)

        # Debug print
        print("Prediction:", pred)

        # Step environment
        requests.post(f"{BASE_URL}/step", params={"action": action})

    # Show all predictions
    print("All predictions:", predictions)

    # Send to grader (JSON body)
    grade_res = requests.post(
    f"{BASE_URL}/grader",
    json=predictions
)

    # Safe print
    try:
        print("Final Score:", grade_res.json())
    except:
        print("Raw response:", grade_res.text)


if __name__ == "__main__":
    run_inference()
