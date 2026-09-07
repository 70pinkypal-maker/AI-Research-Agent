import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain import agents

# Load environment variables
load_dotenv("research/.env", override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# Weather tool
@tool
def get_weather_data(city: str) -> str:
    """Fetch current weather information for a city."""
    import requests

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    current = data["current"]

    return (
        f"Weather in {city}: "
        f"{current['temperature']} C, "
        f"{current['weather_descriptions'][0]}, "
        f"Humidity: {current['humidity']}%, "
        f"Wind Speed: {current['wind_speed']} km/h"
    )


# Tavily search tool
search_tool = TavilySearchResults(max_results=5,
tavily_api_key=st.secrets[TAVILY_API_KEY])

tools = [search_tool, get_weather_data]


# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    max_retries=0,
    convert_system_message_to_human=True
)


# Agent
agent_executor = agents.initialize_agent(
    tools,
    llm,
    agent=agents.AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)


# Streamlit UI
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Research Agent")
st.write("Ask a question and let the AI agent research it for you.")

question = st.text_input(
    "🔍 Ask your question",
    placeholder="e.g. What is the current weather in Delhi?"
)

if st.button("Ask Agent 🚀"):
    if question:
        with st.spinner("🤖 Agent is thinking..."):
            response = agent_executor.invoke({
                "input": question
            })

        st.subheader("🤖 AI Answer")
        st.write(response["output"])

    else:
        st.warning("Please enter a question.")
