import streamlit as st
from langgraph_tool_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# ===================Thread_id=========================
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []  

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_thread(thread_id):
    # Placeholder for loading thread-specific history if needed
    return chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values["messages"]

# =====================Session Setups=======================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# =======================Siderbar UI=====================
st.sidebar.title("LangGraph Chatbot")
if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Conversation History")

for thread_id in st.session_state["chat_threads"]:
    if st.sidebar.button(str(thread_id), key=str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages  = load_thread(thread_id)

        temp_history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role="user"
            else:
                role="assistant"
            temp_history.append({"role": role, "content": msg.content})
        
        st.session_state["message_history"] = temp_history

# =====================Main UI======================
# Display chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Add user message to history and display
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "recursion_limit": 50,  # increased limit
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "Chat_turn",
    }

    # ================== STATUS CONTAINER ==================
    status_container = st.empty()  # placeholder for status updates
    status_container.info("🔎 Processing... (Normal Search)")  

    # Stream AI response - ONLY show AI messages
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                # ⭐ FIXED TOOL DETECTION ⭐
                tool_name = None
                if metadata:
                    if "tool" in metadata:
                        tool_name = metadata["tool"]
                    elif "tool_name" in metadata:
                        tool_name = metadata["tool_name"]
                    elif "node" in metadata:
                        tool_name = metadata["node"]

                if tool_name:
                    status_container.warning(f"⚙️ Tool Search: Using **{tool_name}**")
                else:
                    status_container.info("💬 Normal Search...")

                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        # Use st.write_stream to display the AI-only response
        ai_response = st.write_stream(ai_only_stream())
    
    # Save the AI response to history
    st.session_state["message_history"].append({"role": "assistant", "content": ai_response})
    
    # Final status update
    status_container.success("✅ Response Complete")
