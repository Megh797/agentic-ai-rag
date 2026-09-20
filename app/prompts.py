SYSTEM_PROMPT = """
You are an AI assistant answering questions about the
Agentic AI ebook.

STRICT RULES:

1. Answer ONLY using the retrieved context.
2. Do not use outside knowledge.
3. Do not invent or assume information.
4. If the answer cannot be found in the retrieved context,
   say exactly:

I could not find this information in the provided knowledge base.

5. Keep the answer clear and concise.
6. When possible, mention the relevant page number.
"""


def build_prompt(
    question: str,
    contexts: list
):
    """
    Build the prompt using only retrieved
    ebook context.
    """

    context_text = ""

    for i, item in enumerate(
        contexts,
        start=1
    ):

        page = item.get(
            "page",
            "unknown"
        )

        text = item.get(
            "text",
            ""
        )

        context_text += (
            f"\n--- Context {i} "
            f"(Page {page}) ---\n"
            f"{text}\n"
        )

    prompt = f"""
{SYSTEM_PROMPT}

RETRIEVED CONTEXT:
{context_text}

USER QUESTION:
{question}

ANSWER:
"""

    return prompt