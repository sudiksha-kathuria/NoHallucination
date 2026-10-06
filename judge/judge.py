import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_token_overlap(text1: str, text2: str) -> float:
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    return len(intersection) / len(tokens1)

def compute_faithfulness(answer: str, chunks: list) -> float:
    if not chunks or not answer:
        return 0.0
    context = " ".join(chunks)
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())
    overlap = answer_tokens & context_tokens
    return round(len(overlap) / len(answer_tokens), 4)

def compute_relevancy(query: str, answer: str, embedding_model) -> float:
    q_emb = embedding_model.get_text_embedding(query)
    a_emb = embedding_model.get_text_embedding(answer)
    q = np.array(q_emb).reshape(1, -1)
    a = np.array(a_emb).reshape(1, -1)
    return float(cosine_similarity(q, a)[0][0])

def compute_context_utilisation(answer: str, chunks: list) -> float:
    if not chunks:
        return 0.0
    answer_tokens = set(answer.lower().split())
    used = 0
    for chunk in chunks:
        chunk_tokens = set(chunk.lower().split())
        if len(answer_tokens & chunk_tokens) > 2:
            used += 1
    return round(used / len(chunks), 4)

def judge_agent(state, embedding_model):
    answer = state.get("draft_answer", " ")
    query = state.get("rewritten_query") or state.get("original_query", " ")
    chunks = state.get("retrieved_chunks", [])

    scores = {
        "faithfulness": compute_faithfulness(answer, chunks),
        "relevancy": round(compute_relevancy(query, answer, embedding_model), 4),
        "context_utilisation": compute_context_utilisation(answer, chunks)
    }
    overall = round(np.mean(list(scores.values())), 4)

    print(f"[Judge] Faithfulness:        {scores['faithfulness']}")
    print(f"[Judge] Relevancy:           {scores['relevancy']}")
    print(f"[Judge] Context Utilisation: {scores['context_utilisation']}")
    print(f"[Judge] Overall Score:       {overall}")

    return {
        **state,
        "judge_scores": scores,
        "judge_overall": overall
    }