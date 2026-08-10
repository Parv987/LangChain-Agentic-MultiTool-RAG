from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from src.config import OPENROUTER_API_KEY, SERPAPI_API_KEY, WEATHER_API_KEY 


@tool
def search_tool(query: str) -> str:
    """
    Search Google for current general information using SerpAPI.
    Do not use this tool for weather questions.
    """
    print("SearchAPI - Search Toll Called")
    return SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY).run(query)

@tool
def weather_tool(location: str) -> str:
    """
    Use this tool only for current weather questions. Input should be a city name.
    """    