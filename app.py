import streamlit as st
from langgraph_tool_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
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
    st.session_state["current_status"] = "Ready"
    st.session_state["current_tool"] = "None"

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

if "current_status" not in st.session_state:
    st.session_state["current_status"] = "Ready"

if "current_tool" not in st.session_state:
    st.session_state["current_tool"] = "None"

add_thread(st.session_state["thread_id"])

# =====================Status Container======================
# Status container at the top - ALWAYS VISIBLE
st.subheader("🤖 Agent Status")

status_col1, status_col2, status_col3 = st.columns([1, 1, 1])

with status_col1:
    # Current Status
    status_emoji = {
        "Ready": "✅",
        "Thinking": "💭", 
        "Searching": "🔍",
        "Calculating": "🧮",
        "Fetching Stock": "📈",
        "Processing": "⚡"
    }
    current_status = st.session_state["current_status"]
    emoji = status_emoji.get(current_status, "✅")
    st.metric("Status", f"{emoji} {current_status}")

with status_col2:
    # Current Tool
    tool_emoji = {
        "None": "🛠️",
        "Web Search": "🔍",
        "Calculator": "🧮", 
        "Stock Data": "📈"
    }
    current_tool = st.session_state["current_tool"]
    emoji = tool_emoji.get(current_tool, "🛠️")
    st.metric("Tool", f"{emoji} {current_tool}")

with status_col3:
    # Activity Indicator
    if st.session_state["current_status"] != "Ready":
        st.metric("Activity", "🔄 Working")
    else:
        st.metric("Activity", "✅ Idle")

# Separator
st.divider()

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
    
    # Update status to Thinking
    st.session_state["current_status"] = "Thinking"
    st.session_state["current_tool"] = "None"
    
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "recursion_limit": 10,
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "Chat_turn",
    }

    # Stream AI response - ONLY show AI messages with status updates
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                # Update status based on what the agent is doing
                if isinstance(message_chunk, AIMessage):
                    if hasattr(message_chunk, 'tool_calls') and message_chunk.tool_calls:
                        # Agent is using a tool
                        tool_call = message_chunk.tool_calls[0]
                        tool_name = tool_call['name']
                        
                        if tool_name == "DuckDuckGoSearchRun":
                            st.session_state["current_status"] = "Searching"
                            st.session_state["current_tool"] = "Web Search"
                        elif tool_name == "calculator_tool":
                            st.session_state["current_status"] = "Calculating" 
                            st.session_state["current_tool"] = "Calculator"
                        elif tool_name == "get_stock_price":
                            st.session_state["current_status"] = "Fetching Stock"
                            st.session_state["current_tool"] = "Stock Data"
                    
                    else:
                        # Agent is thinking/processing
                        st.session_state["current_status"] = "Processing"
                        st.session_state["current_tool"] = "None"
                
                elif isinstance(message_chunk, ToolMessage):
                    # Tool execution completed
                    st.session_state["current_status"] = "Processing"
                    st.session_state["current_tool"] = "None"
                
                # Yield only AI message content for display
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

            # Final status update when done
            st.session_state["current_status"] = "Ready"
            st.session_state["current_tool"] = "None"

        # Use st.write_stream to display the AI-only response
        ai_response = st.write_stream(ai_only_stream())
    
    # Save the AI response to history
    st.session_state["message_history"].append({"role": "assistant", "content": ai_response})