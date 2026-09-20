from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from .retrieval import retrieve
from .prompts import build_prompt


class RAGState(TypedDict):
    question: str
    contexts: list
    confidence: float
    answer: str


def retrieve_node(state: RAGState):
    """
    Retrieve the most relevant chunks from Pinecone.
    """

    contexts = retrieve(state["question"])

    confidence = 0.0

    if contexts:
        confidence = max(
            item.get("score", 0.0)
            for item in contexts
        )

    return {
        "contexts": contexts,
        "confidence": confidence,
    }


def generate_node(state: RAGState):
    """
    Generate an answer using the retrieved context.

    This function will be connected to our free LLM
    after the retrieval pipeline is working.
    """

    prompt = build_prompt(
        state["question"],
        state["contexts"]
    )

    # Temporary response while we test retrieval.
    answer = (
        "Retrieved context successfully. "
        "LLM generation will be connected next.\n\n"
        + prompt
    )

    return {
        "answer": answer
    }


def fallback_node(state: RAGState):
    """
    Return a grounded fallback when the retrieved
    context is not relevant enough.
    """

    return {
        "answer": (
            "I could not find this information "
            "in the provided knowledge base."
        )
    }


def check_relevance(state: RAGState):
    """
    Decide whether the retrieved context is relevant
    enough to generate an answer.
    """

    if not state["contexts"]:
        return "fallback"

    if state["confidence"] < 0.45:
        return "fallback"

    return "generate"


# ============================================================
# Build LangGraph
# ============================================================

workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("fallback", fallback_node)

workflow.add_edge(START, "retrieve")

workflow.add_conditional_edges(
    "retrieve",
    check_relevance,
    {
        "generate": "generate",
        "fallback": "fallback",
    },
)

workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

graph = workflow.compile()


# ============================================================
# Public function used by FastAPI
# ============================================================

def ask(question: str):
    """
    Run the RAG workflow.
    """

    result = graph.invoke(
        {
            "question": question,
            "contexts": [],
            "confidence": 0.0,
            "answer": "",
        }
    )

    return {
        "question": question,
        "answer": result["answer"],
        "confidence": round(
            result["confidence"],
            4
        ),
        "retrieved_context": result["contexts"],
    }