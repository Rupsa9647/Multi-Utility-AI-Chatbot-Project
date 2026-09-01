import uuid
import queue
import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend_rag import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    submit_async_task,
    thread_document_metadata,
)

# =========================== Utilities ===========================

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def get_ui_messages_from_thread(thread_id):
    """🔥 ONLY return Human + AI messages (NO ToolMessage)"""
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    ui_messages = []

    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            ui_messages.append(
                {"role": "user", "content": msg.content}
            )

        elif isinstance(msg, AIMessage) and msg.content.strip():
            ui_messages.append(
                {"role": "assistant", "content": msg.content}
            )

        # ❌ ToolMessage ignored completely

    return ui_messages


# ======================= Session Initialization ===================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_id = st.session_state["thread_id"]
thread_docs = st.session_state["ingested_docs"].setdefault(thread_id, {})

# ============================ Sidebar ============================

st.sidebar.title("🧠 LangGraph MCP + RAG Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_id}`")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# ---- PDF Upload ----
uploaded_pdf = st.sidebar.file_uploader(
    "📄 Upload PDF for this chat",
    type=["pdf"],
)

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already indexed.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_id,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status.update(
                label="✅ PDF indexed successfully",
                state="complete",
                expanded=False,
            )

# ---- Show active document ----
doc_meta = thread_document_metadata(thread_id)
if doc_meta:
    st.sidebar.success(
        f"Using **{doc_meta.get('filename')}**  \n"
        f"Pages: {doc_meta.get('documents')} | "
        f"Chunks: {doc_meta.get('chunks')}"
    )
else:
    st.sidebar.info("No document indexed for this chat.")

# ---- Past Conversations ----
st.sidebar.subheader("💬 Past Conversations")

for tid in reversed(st.session_state["chat_threads"]):
    if st.sidebar.button(tid, key=f"thread-{tid}"):
        st.session_state["thread_id"] = tid
        st.session_state["message_history"] = get_ui_messages_from_thread(tid)
        st.session_state["ingested_docs"].setdefault(tid, {})
        st.rerun()

# ============================ Main UI ============================

st.title("🤖 Multi-Utility AI Chatbot")

# ---- Render chat history ----
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools…")

# ============================ Chat Logic ============================

if user_input:
    # ---- Show user message ----
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    # ================= Assistant Streaming =================

    status_holder = {"box": None}

    with st.chat_message("assistant"):
        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for msg, _ in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put(msg)
                except Exception as e:
                    event_queue.put(e)
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break

                if isinstance(item, Exception):
                    raise item

                # 🔧 Tool status (NO chat bubble)
                if isinstance(item, ToolMessage):
                    tool_name = getattr(item, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …",
                            expanded=True,
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                        )
                    continue

                # 🤖 Stream ONLY AI text
                if isinstance(item, AIMessage) and item.content.strip():
                    yield item.content

        ai_response = st.write_stream(ai_only_stream())

    if status_holder["box"] is not None:
        status_holder["box"].update(
            label="✅ Tool finished",
            state="complete",
            expanded=False,
        )

    # ---- Save assistant message ----
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_response}
    )
