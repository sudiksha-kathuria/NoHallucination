from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()

def detect_pii(text: str) -> dict:
    results = analyzer.analyze(text=text, language="en")
    if results:
        entities = list(set([r.entity_type for r in results]))
        return {"contains_pii": True, "pii_entities": entities}
    return {"contains_pii": False, "pii_entities": []}

def input_guard_agent(state, **kwargs):
    query = state.get("rewritten_query") or state.get("original_query", "")
    result = detect_pii(query)
    print(f"[InputGuard] PII detected: {result['contains_pii']}")
    if result["contains_pii"]:
        print(f"[InputGuard] Entities: {result['pii_entities']}")
        return {
            **state,
            "pii_detected": True,
            "pii_entities": result["pii_entities"],
            "should_block": True,
            "block_reason": f"Input contains PII: {', '.join(result['pii_entities'])}"
        }
    return {**state, "pii_detected": False, "pii_entities": []}