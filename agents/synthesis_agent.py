def synthesis_agent(state, llm):
    context = "\n\n--\n\n".join(state.get("retrieved_chunks", []))
    prompt = f"""You are a precise answer generator.
Your job is to answer the user's question using ONLY the information provided in the context below.
Respond directly with the answer only. No thinking, no steps, no analysis.
Rules:
    - Only use information explicitly present in the context
    - If the context does not contain enough information to answer, say "I could not find sufficient information in the provided documents to answer this question"
    - Do not add information from your general knowledge
    - Be concise and direct
    - Do not mention that you are using a context or chunks
    - Do not send conversational texts, just the answer directly
Context: {context}
Question: {state["original_query"]}
Answer:"""

    response = llm.complete(prompt)
    response_text = response.text.strip()
    if "</think>" in response_text:
        response_text = response_text.split("</think>")[-1].strip()
    elif "<think>" in response_text:
        if "Draft Response:" in response_text:
            response_text = response_text.split("Draft Response:")[-1].strip()
        elif "**Draft Response:**" in response_text:
            response_text = response_text.split("**Draft Response:**")[-1].strip()
        else:
            response_text = "Could not generate a clean response. Please try again."

    return {**state, "draft_answer": response_text}