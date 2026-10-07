import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_faithfulness(answer: str, chunks: list) -> float:
    if not chunks or not answer.strip():
        return 0.0
    context = " ".join(chunks)
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())
    if not answer_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return round(len(overlap) / len(answer_tokens), 4)


def compute_relevancy(query: str, answer: str, embedding_model) -> float:
    if not query.strip() or not answer.strip():
        return 0.0
    q_emb = np.array(embedding_model.get_text_embedding(query)).reshape(1, -1)
    a_emb = np.array(embedding_model.get_text_embedding(answer)).reshape(1, -1)
    return round(float(cosine_similarity(q_emb, a_emb)[0][0]), 4)


def compute_context_utilisation(answer: str, chunks: list) -> float:
    if not chunks:
        return 0.0
    answer_tokens = set(answer.lower().split())
    used = sum(1 for chunk in chunks if len(answer_tokens & set(chunk.lower().split())) > 2)
    return round(used / len(chunks), 4)


def judge_agent(state, embedding_model):
    answer = state.get("draft_answer") or ""
    query = state.get("rewritten_query") or state.get("original_query") or ""
    chunks = state.get("retrieved_chunks") or []

    scores = {
        "faithfulness": compute_faithfulness(answer, chunks),
        "relevancy": compute_relevancy(query, answer, embedding_model),
        "context_utilisation": compute_context_utilisation(answer, chunks),
    }
    overall = round(float(np.mean(list(scores.values()))), 4)

    print(f"[Judge] Faithfulness:        {scores['faithfulness']}")
    print(f"[Judge] Relevancy:           {scores['relevancy']}")
    print(f"[Judge] Context Utilisation: {scores['context_utilisation']}")
    print(f"[Judge] Overall Score:       {overall}")

    return {**state, "judge_scores": scores, "judge_overall": overall}