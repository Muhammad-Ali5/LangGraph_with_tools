import streamlit as st
from langgraph_tool_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import json

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
    return chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values["messages"]

# =====================Session Setups=======================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# =======================Sidebar UI=====================
st.sidebar.title("LangGraph Chatbot")
if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Conversation History")

for thread_id in st.session_state["chat_threads"]:
    if st.sidebar.button(str(thread_id), key=str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_thread(thread_id)
        temp_history = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_history.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_history

# =====================Main UI======================
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "recursion_limit": 10,
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "Chat_turn",
    }

    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    if message_chunk.content and message_chunk.content.strip():
                        # Yield direct AI message content (e.g., for "how are you")
                        yield message_chunk.content + "\n"
                    elif message_chunk.tool_calls:
                        # Handle tool call results (e.g., jokes)
                        for tool_call in message_chunk.tool_calls:
                            if tool_call["name"] == "get_joke":
                                # Get the tool result by invoking the chatbot
                                result = chatbot.invoke(
                                    {"messages": [message_chunk]},
                                    config=CONFIG
                                )["messages"][-1]
                                if isinstance(result, AIMessage) and result.content:
                                    try:
                                        # Parse the tool output (e.g., {"joke": "Why did..."})
                                        tool_output = json.loads(result.content)
                                        if "joke" in tool_output and tool_output["joke"]:
                                            yield tool_output["joke"] + "\n"
                                        elif "error" in tool_output:
                                            yield f"Error: {tool_output['error']}\n"
                                        else:
                                            yield "No joke found.\n"
                                    except json.JSONDecodeError:
                                        # If result isn't JSON, use it directly (fallback)
                                        yield result.content + "\n"
                                else:
                                    yield "No joke found.\n"

        # Use st.write_stream to display the streamed response
        ai_response = st.write_stream(ai_only_stream)
    
    # Save the AI response to history (join streamed parts)
    st.session_state["message_history"].append({"role": "assistant", "content": ai_response})