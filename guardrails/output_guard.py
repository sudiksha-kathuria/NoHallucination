from detoxify import Detoxify

model = Detoxify("original")
TOXICITY_THRESHOLD = 0.5

def check_toxicity(text: str) -> dict:
    scores = model.predict(text)
    is_toxic = scores["toxicity"] > TOXICITY_THRESHOLD
    return {
        "is_toxic": is_toxic,
        "scores": {k: round(float(v), 4) for k, v in scores.items()}
    }

def output_guard_agent(state, **kwargs):
    answer = state.get("draft_answer", "")
    result = check_toxicity(answer)
    print(f"[OutputGuard] Toxicity score: {result['scores']['toxicity']}")
    print(f"[OutputGuard] Is toxic: {result['is_toxic']}")
    if result["is_toxic"]:
        return {
            **state,
            "is_toxic": True,
            "toxicity_scores": result["scores"],
            "should_block": True,
            "block_reason": f"Output flagged as toxic (score: {result['scores']['toxicity']})"
        }
    return {**state, "is_toxic": False, "toxicity_scores": result["scores"]}