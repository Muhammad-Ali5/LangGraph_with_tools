from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import os
import requests
import json
import random

load_dotenv()

# =========================LLM Setup======================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
)

# =========================Tools Setup======================
search_tool = DuckDuckGoSearchRun()

@tool
def calculator_tool(first_num: float, second_num: float, operation: str) -> dict:
    """
    A simple calculator tool for basic arithmetic operations.
    Supported operations: add, subtract, multiply, divide.
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "subtract":
            result = first_num - second_num
        elif operation == "multiply":
            result = first_num * second_num
        elif operation == "divide":
            if second_num == 0:
                return {"error": "Division by zero is not allowed."}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation: {operation}"}
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the current stock price for a given symbol using Alpha Vantage API.
    symbol (str): The stock symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
    """
    try:
        url = "https://www.alphavantage.co/query"
        params = {'function': 'GLOBAL_QUOTE', 'symbol': symbol.upper(), 'apikey': os.getenv("ALPHA_VANTAGE_API_KEY")}
        response = requests.get(url, params=params)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@tool
def fetch_weather(city: str) -> dict:
    """
    Fetch the current weather for a given city using the WeatherAPI.
    city (str): The name of the city (e.g., 'London', 'New York')
    """
    try:
        API_KEY = os.getenv("WEATHER_API_KEY")
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
        r = requests.get(url)
        data = json.loads(r.text)
        if "error" in data:
            return {"error": data["error"]["message"]}
        return {
            "city": city,
            "temperature_c": data["current"]["temp_c"],
            "condition": data["current"]["condition"]["text"],
            "humidity": data["current"]["humidity"]
        }
    except Exception as e:
        return {"error": f"Failed to fetch weather: {str(e)}"}

@tool
def fetch_news(topic: str) -> dict:
    """
    Fetch latest news headlines on a given topic.
    topic (str): The news topic (e.g., 'technology', 'sports').
    """
    try:
        API_KEY = os.getenv("NEWS_API_KEY")
        url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={API_KEY}&pageSize=5"
        r = requests.get(url)
        data = json.loads(r.text)
        if "articles" in data:
            return {"headlines": [article["title"] for article in data["articles"]]}
        return {"error": "No news found."}
    except Exception as e:
        return {"error": str(e)}

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """
    Convert an amount from one currency to another.
    amount (float): The amount to convert.
    from_currency (str): Source currency (e.g., 'USD').
    to_currency (str): Target currency (e.g., 'EUR').
    """
    try:
        API_KEY = os.getenv("EXCHANGE_API_KEY")
        url = f"https://openexchangerates.org/api/latest.json?app_id={API_KEY}"
        r = requests.get(url)
        data = json.loads(r.text)
        if "rates" in data:
            rate = data["rates"][to_currency] / data["rates"][from_currency]
            result = amount * rate
            return {"result": result}
        return {"error": "Conversion failed."}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_joke(category: str = "Any") -> dict:
    """
    Fetch a random joke from a category.
    category (str): Joke category (e.g., 'Programming', 'Pun', 'Misc', 'Any'). Default: 'Any'.
    """
    try:
        url = f"https://v2.jokeapi.dev/joke/{category}?type=single"
        r = requests.get(url)
        data = json.loads(r.text)
        if "joke" in data and data["joke"]:
            return {"joke": data["joke"]}
        return {"error": "No joke found."}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_nasa_apod() -> dict:
    """
    Fetch NASA's Astronomy Picture of the Day (APOD).
    """
    try:
        API_KEY = os.getenv("NASA_API_KEY")
        url = f"https://api.nasa.gov/planetary/apod?api_key={API_KEY}"
        r = requests.get(url)
        data = json.loads(r.text)
        return {
            "title": data.get("title"),
            "explanation": data.get("explanation")[:200],
            "image_url": data.get("url")
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def get_ip_location(ip: str) -> dict:
    """
    Fetch location info for a given IP address.
    ip (str): The IP address (e.g., '8.8.8.8').
    """
    try:
        url = f"https://ipapi.co/{ip}/json/"
        r = requests.get(url)
        data = json.loads(r.text)
        return {
            "city": data.get("city"),
            "country": data.get("country_name"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude")
        }
    except Exception as e:
        return {"error": str(e)}

tools = [search_tool, calculator_tool, get_stock_price, fetch_weather, fetch_news, convert_currency, get_joke, get_nasa_apod, get_ip_location]
llm_with_tools = llm.bind_tools(tools=tools)

# =========================State===========================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# =========================Graph Node Definition======================
def chat_node(state: ChatState) -> dict:
    """LLM node that handles conversation or requests a tool call."""
    try:
        last_message = state["messages"][-1].content.lower()
        # Handle casual conversation
        if "how are you" in last_message or "hey" in last_message:
            return {"messages": [AIMessage(content="Hey there! I'm ready to help. What's on your mind?")]}
        # Handle single joke request
        elif "tell me a joke" in last_message or "another joke" in last_message:
            return {"messages": [AIMessage(content="", tool_calls=[{"name": "get_joke", "args": {"category": "Any"}, "id": "joke_call"}])]}
        # Handle multiple joke requests (e.g., "tell me 4 jokes")
        elif "joke" in last_message and any(num in last_message for num in ["1", "2", "3", "4", "5"]):
            try:
                num_jokes = int(next(num for num in ["1", "2", "3", "4", "5"] if num in last_message))
                tool_calls = [
                    {"name": "get_joke", "args": {"category": random.choice(["Any", "Programming", "Pun", "Misc"])}, "id": f"joke_call_{i}"}
                    for i in range(num_jokes)
                ]
                return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}
            except:
                return {"messages": [AIMessage(content="", tool_calls=[{"name": "get_joke", "args": {"category": "Any"}, "id": "joke_call"}])]}
        else:
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
    except Exception as e:
        print(f"Error in chat_node: {str(e)}")
        return {"messages": [SystemMessage(content="Sorry, I hit an error. Please try again.")]}

def custom_tools_node(state: ChatState) -> dict:
    """Custom tools node to handle tool call results cleanly."""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_results = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    result = tool.invoke(tool_args)
                    # Convert result to string if it's a dict
                    if isinstance(result, dict):
                        if "joke" in result:
                            result = result["joke"]
                        elif "error" in result:
                            result = f"Error: {result['error']}"
                        else:
                            result = json.dumps(result)
                    tool_results.append(AIMessage(content=result, tool_call_id=tool_id))
        return {"messages": tool_results}
    return {"messages": []}

# =========================Database Setup======================
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# =========================Graph Definition======================
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools_node", custom_tools_node)

graph.add_edge(START, "chat_node")

def route_tools(state: ChatState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools_node"
    return "__end__"

graph.add_conditional_edges(
    "chat_node",
    route_tools,
    {
        "tools_node": "tools_node",
        "__end__": END,
    }
)

graph.add_edge("tools_node", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# =========================Database Operations======================
def retrieve_all_threads():
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint["config"]["configurable"]["thread_id"])
    except Exception as e:
        print(f"Error retrieving threads: {e}")
    return list(all_threads)