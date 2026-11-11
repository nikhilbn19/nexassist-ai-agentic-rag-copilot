import logging
from pathlib import Path
from typing import Literal
from langchain_core.tools import ToolException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None

def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content)

def llm():

    global _llm

    if _llm is None:

        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing")

        _llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0,
            google_api_key=settings.gemini_api_key,
            max_retries=1,
            timeout=10,
        )

    return _llm

def web_search_tool():

    global _web_search

    if _web_search is None:

        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")

        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )

    return _web_search

def add_trace(state: AgentState, message: str):

    return [*state.get("trace", []), message]

def route_question(state: AgentState):
    route = "kb"
    try:
        router = llm().with_structured_output(
            RouteDecision,
            method="json_schema"
        )

        decision = router.invoke(f"""
You route messages for an enterprise IT support assistant.

Use kb for questions about:
- Company IT policies
- VPN
- Password reset
- MFA
- Laptop setup
- Software access
- Security
- Email
- Devices
- Troubleshooting
- Technology support

Use direct only for greetings, thanks, or casual chat
that needs no company knowledge.

Question: {state['question']}

Return the routing decision using the required schema.
""")
        route = decision.route
    except Exception as exc:
        logger.error("Router LLM invocation failed: %s", exc)
        route = "kb"

    return {
        "source_used": route,
        "trace": add_trace(
            state,
            f"Router → {route.upper()}"
        )
    }


def route_after_router(
    state: AgentState
) -> Literal["retrieve_kb", "direct_answer"]:

    return (
        "retrieve_kb"
        if state["source_used"] == "kb"
        else "direct_answer"
    )

def retrieve_kb(state: AgentState):
    try:
        docs = get_retriever().invoke(state["current_query"])
    except Exception as exc:
        logger.error("Private KB retrieval failed: %s", exc)
        return {
            "kb_docs": [],
            "trace": add_trace(
                state,
                "Private KB retrieval failed → falling back to web search"
            )
        }

    return {
        "kb_docs": docs,
        "trace": add_trace(
            state,
            f"Private KB retrieval → {len(docs)} chunks"
        )
    }

def grade_kb(state: AgentState):
    if not state.get("kb_docs"):
        return {
            "kb_grade": "weak",
            "trace": add_trace(
                state,
                "KB evidence grade → WEAK (no documents retrieved)"
            )
        }

    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_schema"
    )

    context = "\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n"
        f"{d.page_content}"
        for d in state["kb_docs"]
    )

    try:
        grade = grader.invoke(f"""
You grade evidence for an enterprise IT support assistant.

Question:
{state['question']}

Private company KB evidence:
{context}

Return good only if the evidence is sufficient to answer
confidently and specifically.

Otherwise return weak.

Return the result using the required schema.
""")
        grade_val = grade.grade
    except Exception as exc:
        logger.error("Evidence grading LLM invocation failed: %s", exc)
        grade_val = "weak"

    return {
        "kb_grade": grade_val,
        "trace": add_trace(
            state,
            f"KB evidence grade → {grade_val.upper()}"
        )
    }


def after_kb(
    state: AgentState
) -> Literal["generate_from_kb", "search_web"]:

    return (
        "generate_from_kb"
        if state["kb_grade"] == "good"
        else "search_web"
    )

def search_web(state: AgentState):
    lines = []
    citations = []

    try:
        result = web_search_tool().invoke({
            "query": state["current_query"]
        })
    except ToolException as te:
        logger.warning("Tavily search returned no results for query '%s': %s", state["current_query"], te)
        return {
            "web_results": "",
            "citations": [],
            "source_used": "web",
            "trace": add_trace(
                state,
                "Web fallback → Tavily search (no results found)"
            )
        }
    except Exception as exc:
        logger.error("Tavily search failed for query '%s': %s", state["current_query"], exc)
        return {
            "web_results": "",
            "citations": [],
            "source_used": "web",
            "trace": add_trace(
                state,
                "Web fallback → Tavily search failed"
            )
        }

    if isinstance(result, dict):
        if "error" in result:
            logger.error("Tavily search returned error: %s", result["error"])
            return {
                "web_results": "",
                "citations": [],
                "source_used": "web",
                "trace": add_trace(
                    state,
                    "Web fallback → Tavily search unavailable"
                )
            }

        if result.get("answer"):
            lines.append(
                "Search answer: " + result["answer"]
            )

        for item in result.get("results", []):

            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")

            lines.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )

            citations.append({
                "title": title or url,
                "url": url,
                "type": "web"
            })

    else:
        lines.append(str(result))

    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(
            state,
            "Web fallback → Tavily search"
        )
    }

def grade_web(state: AgentState):
    if not state.get("web_results", "").strip():
        return {
            "web_grade": "weak",
            "trace": add_trace(
                state,
                "Web evidence grade → WEAK (no web results)"
            )
        }

    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_schema"
    )

    try:
        grade = grader.invoke(f"""
You grade web evidence for an enterprise IT support assistant.

Question:
{state['question']}

Web evidence:
{state['web_results']}

Return good if the evidence is sufficient and directly relevant.
Otherwise return weak.

Return the result using the required schema.
""")
        grade_val = grade.grade
    except Exception as exc:
        logger.error("Web evidence grading LLM invocation failed: %s", exc)
        grade_val = "weak"

    return {
        "web_grade": grade_val,
        "trace": add_trace(
            state,
            f"Web evidence grade → {grade_val.upper()}"
        )
    }


def after_web(
    state: AgentState
) -> Literal[
    "generate_from_web",
    "rewrite_query",
    "insufficient"
]:

    if state["web_grade"] == "good":
        return "generate_from_web"

    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"

    return "insufficient"

def rewrite_query(state: AgentState):
    try:
        raw = llm().invoke(f"""
Rewrite this IT support question for better private
knowledge retrieval and vendor web search.

Preserve intent.
Add useful technical keywords.
Do not answer.
Return only the rewritten query.

Question: {state['question']}
""").content
        rewritten = extract_text(raw).strip()
    except Exception as exc:
        logger.error("Query rewrite LLM invocation failed: %s", exc)
        rewritten = state["current_query"]

    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(
            state,
            f"Query rewrite → {rewritten}"
        )
    }

def generate_from_kb(state: AgentState):

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n"
        f"{d.page_content}"
        for d in state["kb_docs"]
    )

    try:
        raw = llm().invoke(f"""
You are an enterprise IT support copilot.

Answer ONLY from the private company KB below.

Instructions:
- Be concise and actionable.
- If steps are present, present them clearly.
- Do not invent policy details.
- Mention that the answer is based on the company's
  private knowledge base.

Question:
{state['question']}

Private KB:
{context}
""").content
        answer = extract_text(raw)
    except Exception as exc:
        logger.error("KB generation LLM invocation failed: %s", exc)
        answer = "An error occurred while generating the answer from the knowledge base. Please try again or contact IT support."

    citations = []
    seen = set()

    for d in state["kb_docs"]:

        src = d.metadata.get("source", "Private KB")

        if src not in seen:

            seen.add(src)

            filename = Path(src).name if src else "Private KB"
            citations.append({
                "title": filename,
                "url": "",
                "type": "private_kb"
            })

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(
            state,
            "Answer generation → PRIVATE KB"
        )
    }

def generate_from_web(state: AgentState):

    try:
        raw = llm().invoke(f"""
You are an enterprise IT support copilot.

The private company KB was insufficient.

Answer ONLY from the web evidence below.

Clearly say this is external web information and may
need IT validation before changing company-managed systems.

Question:
{state['question']}

Web evidence:
{state['web_results']}
""").content
        answer = extract_text(raw)
    except Exception as exc:
        logger.error("Web generation LLM invocation failed: %s", exc)
        answer = "An error occurred while synthesizing external web search results. Please try again or contact IT support."

    return {
        "answer": answer,
        "source_used": "web_search",
        "trace": add_trace(
            state,
            "Answer generation → WEB SEARCH"
        )
    }

def direct_answer(state: AgentState):

    try:
        raw = llm().invoke(
            f"Respond briefly and naturally to: {state['question']}"
        ).content
        answer = extract_text(raw)
    except Exception as exc:
        logger.error("Direct answer LLM invocation failed: %s", exc)
        answer = "Hello! How can I assist you with your IT questions today?"

    return {
        "answer": answer,
        "source_used": "direct",
        "citations": [],
        "trace": add_trace(
            state,
            "Direct response → no retrieval"
        )
    }

def insufficient(state: AgentState):

    return {
        "answer": (
            "I couldn't find enough reliable evidence in the "
            "company knowledge base or external search to "
            "answer confidently. Please contact the IT help "
            "desk or provide more details."
        ),
        "source_used": "insufficient_evidence",
        "citations": [],
        "trace": add_trace(
            state,
            "Stopped → insufficient reliable evidence"
        )
    }

def build_graph():

    graph = StateGraph(AgentState)

    nodes = {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    
    graph.add_edge(START, "route_question")

    graph.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "direct_answer": "direct_answer"
        }
    )

    graph.add_edge("retrieve_kb", "grade_kb")

    graph.add_conditional_edges(
        "grade_kb",
        after_kb,
        {
            "generate_from_kb": "generate_from_kb",
            "search_web": "search_web"
        }
    )

    graph.add_edge("search_web", "grade_web")

    graph.add_conditional_edges(
        "grade_web",
        after_web,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient"
        }
    )

    graph.add_edge("rewrite_query", "retrieve_kb")

    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)

    return graph.compile()


agent_graph = build_graph()

def ask(question: str):

    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }

    return agent_graph.invoke(initial)
