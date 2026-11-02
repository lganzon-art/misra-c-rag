from __future__ import annotations

import logging
import uuid
from pathlib import Path

import streamlit as st

from rag import DOCS_DIR, MisraRag, corpus_fingerprint
from input_guardrails import screen_question, warm_up_guardrails

st.set_page_config(
    page_title="MISRA C | Reference assistant",
    page_icon=":material/menu_book:",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.title("MISRA C assistant", icon=":material/menu_book:")
st.caption("Clear answers grounded in your reference documents, with citations you can inspect.")


def log_safe_error(stage: str, exc: Exception) -> str:
    """Log only fixed metadata; exception text and traceback may contain secrets."""
    event_id = uuid.uuid4().hex[:12]
    logging.getLogger(__name__).error(
        "event_id=%s stage=%s exception_type=%s",
        event_id, stage, type(exc).__name__,
    )
    return event_id


st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("conversations", {})


def save_conversation() -> None:
    """Keep the sanitized chat in Streamlit session state for sidebar switching."""
    history = st.session_state.chat_history
    if history:
        st.session_state.conversations[st.session_state.thread_id] = list(history)


def conversation_title(history: list[dict]) -> str:
    first_question = next(
        (turn["content"] for turn in history if turn["role"] == "user"),
        "Conversation",
    )
    return first_question[:48] + ("…" if len(first_question) > 48 else "")


with st.sidebar:
    st.header("Workspace")
    st.caption("MISRA C reference assistant")
    if st.button("New conversation", icon=":material/add_comment:", width="stretch"):
        save_conversation()
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.conversations:
        st.subheader("Recent conversations")
        for thread_id, history in reversed(list(st.session_state.conversations.items())):
            if st.button(
                conversation_title(history),
                key=f"open_{thread_id}",
                icon=":material/chat_bubble_outline:",
                width="stretch",
                disabled=thread_id == st.session_state.thread_id,
            ):
                save_conversation()
                st.session_state.thread_id = thread_id
                st.session_state.chat_history = list(history)
                st.rerun()

    st.subheader("Reference documents", icon=":material/folder_open:")
    st.caption("Add a PDF or text reference that you are authorized to use.")
    uploads = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if uploads:
        changed = False
        for uploaded in uploads:
            target = DOCS_DIR / Path(uploaded.name).name
            payload = uploaded.getvalue()
            if not target.exists() or target.read_bytes() != payload:
                target.write_bytes(payload)
                changed = True
        if changed:
            st.toast("Documents saved. The index will refresh automatically.", icon=":material/check_circle:")
    st.caption("Only uploaded or locally provided references are searched.")


@st.cache_resource
def get_rag(fingerprint: str) -> MisraRag:
    return MisraRag(corpus_hash=fingerprint)


@st.cache_resource
def prepare_guardrails() -> None:
    warm_up_guardrails()


try:
    fingerprint = corpus_fingerprint()
    if not fingerprint:
        st.info("Add authorized MISRA C PDF or TXT references from the sidebar to get started.")
        st.stop()
    rag = get_rag(fingerprint)
except Exception as exc:
    event_id = log_safe_error("index_setup", exc)
    st.error(f"Could not prepare the RAG index. Reference: {event_id}.")
    st.stop()

try:
    with st.spinner("Preparing input guardrails…"):
        prepare_guardrails()
except Exception as exc:
    event_id = log_safe_error("guardrail_warmup", exc)
    st.error(f"Could not prepare input guardrails. Reference: {event_id}.")
    st.stop()

with st.sidebar:
    st.badge("Documents ready", icon=":material/check_circle:", color="green")


def citation_label(citation: dict) -> str:
    page = citation.get("page") or "n/a"
    return (
        f"{citation['id']}: {citation['source']} · page {page} · "
        f"{citation['section']}"
    )


if not st.session_state.chat_history:
    st.space("large")
    st.subheader("Find a rule. Check the source.")
    st.markdown(
        "Ask about a requirement, exception, or rule category. "
        "Each answer links back to the retrieved evidence."
    )
    suggestions = st.empty()
    with suggestions.container():
        selected = st.pills(
            "Try a question",
            [
                "What does Rule 6.2 prohibit?",
                "Is Rule 5.7 Required or Advisory?",
                "What do Rules 2.3 through 2.7 have in common?",
            ],
        )
else:
    selected = None

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("insufficient_evidence"):
            st.caption("The indexed excerpts did not provide enough evidence.")
        if turn.get("citations"):
            st.caption("Sources: " + "; ".join(citation_label(c) for c in turn["citations"]))
        evidence = turn.get("contexts", turn.get("sources", []))
        if evidence:
            with st.expander("Retrieved evidence"):
                for i, item in enumerate(evidence, start=1):
                    st.markdown(f"**{i}. {item['source']} · page {item['page'] or 'n/a'} · {item['section']}**")
                    st.caption(item["text"][:1800])

question = st.chat_input("Ask about a MISRA C rule or directive", submit_mode="disable")
if selected:
    question = selected
if question:
    if selected:
        suggestions.empty()
    screened = screen_question(question)
    safe_question = screened.text
    st.session_state.chat_history.append({"role": "user", "content": safe_question})
    save_conversation()
    with st.chat_message("user"):
        st.markdown(safe_question)
    with st.chat_message("assistant"):
        with st.spinner("Searching sections and preparing an evidence-based answer…"):
            try:
                result = rag.ask(safe_question, st.session_state.thread_id,
                                 screened_question=screened)
                answer = result["answer"]
                st.markdown(answer)
                if screened.pii_types:
                    st.caption("Personal information was masked before processing.")
                if result["insufficient_evidence"] and not result["guardrail_blocked"]:
                    st.caption("The indexed excerpts did not provide enough evidence.")
                if result["citations"]:
                    st.caption("Sources: " + "; ".join(citation_label(c) for c in result["citations"]))
                if result["contexts"]:
                    with st.expander("Retrieved evidence"):
                        for i, item in enumerate(result["contexts"], start=1):
                            st.markdown(f"**{i}. {item['source']} · page {item['page'] or 'n/a'} · {item['section']}**")
                            st.caption(item["text"][:1800])
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": result["citations"],
                    "contexts": result["contexts"],
                    "insufficient_evidence": result["insufficient_evidence"] and not result["guardrail_blocked"],
                })
                save_conversation()
            except Exception as exc:
                event_id = log_safe_error("answer_generation", exc)
                st.error(f"Answer generation failed. Reference: {event_id}.")


