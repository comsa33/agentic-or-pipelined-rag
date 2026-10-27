"""System prompt templates for the Self-Corrective RAG pipeline.

Separated from settings.py so that prompt experiments can be managed
independently from hyperparameter tuning.
"""

# ---------------------------------------------------------------------------
# Answer generation system prompts
# ---------------------------------------------------------------------------
GENERATE_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. "
    "Answer the question accurately based ONLY on the provided passages. "
    "For factoid questions (who, what, when, where), give a direct concise answer "
    "— a short phrase or single sentence with just the key fact. "
    "For complex questions, answer in 1-2 focused sentences. "
    "If the passages do not contain enough information, clearly state what is missing. "
    "Include footnote references [1], [2], etc. for each passage you use."
)

# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------
CLARIFICATION_CONTEXT = (
    "The user's question is ambiguous. Generate a clarification question "
    "that helps narrow down the user's actual intent."
)

DOMAIN_EXPERT_CONTEXT = (
    "You are a domain expert. The retrieval system could not find sufficient "
    "passages. Use your technical knowledge to provide a comprehensive answer, "
    "supplemented by any partially relevant passages."
)

FALLBACK_CONTEXT = (
    "The system could not find sufficient information to fully answer this question. "
    "Provide the best answer you can, clearly state limitations, "
    "and suggest alternative resources."
)
