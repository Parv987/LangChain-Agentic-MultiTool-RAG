import requests
from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from src.config import SERPAPI_API_KEY, WEATHER_API_KEY


retriever = None


def get_tools(retriever_instance):
    global retriever
    retriever = retriever_instance
    return [search_tool, weather_tool, resume_tool]


@tool
def search_tool(query: str) -> str:
    """
    Search Google for current general information using SerpAPI.
    Do not use this tool for weather questions.
    """
    print("SearchAPI - Search Tool Called")

    if not SERPAPI_API_KEY:
        return "Search_TOOL_ERROR: SERPAPI_API_KEY is missing."

    try:
        return SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY).run(query)
    except Exception as exc:
        return f"Search_TOOL_ERROR: {exc}"


@tool
def weather_tool(location: str) -> str:
    """Use this tool only for current weather questions. Input should be a city name."""
    print("WeatherAPI - Weather Tool Called For location:", location)
    api_key = WEATHER_API_KEY

    if not api_key:
        return (
            "WEATHER_TOOL_ERROR: OPENWEATHER_API_KEY is missing. "
            "Do not call any other tool."
        )

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric",
    }

    response = requests.get(url, params=params, timeout=10)
    print("Status code:", response.status_code)
    print("Raw response:", response.text)

    if response.status_code != 200:
        return (
            f"WEATHER_TOOL_ERROR: OpenWeather API failed with status code "
            f"{response.status_code}. Response: {response.text}. "
            "Do not call search_tool. Tell the user the weather API failed."
        )

    data = response.json()
    city = data["name"]
    country = data["sys"]["country"]
    condition = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    wind_speed = data["wind"]["speed"]

    return (
        f"FINAL_WEATHER_RESULT: Current weather in {city}, {country}: "
        f"{condition}, temperature {temp}°C, feels like {feels_like}°C, "
        f"humidity {humidity}%, wind speed {wind_speed} m/s. "
        "Use this result directly. Do not call search_tool."
    )


@tool
def resume_tool(question: str) -> str:
    """Search the uploaded PDF and return relevant context."""
    print("RESUME TOOL CALLED")

    if retriever is None:
        return "RESUME_TOOL_ERROR: No document retriever has been configured yet."

    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(question)
    elif hasattr(retriever, "retrieve"):
        docs = retriever.retrieve(question)
    elif hasattr(retriever, "invoke"):
        # Try common invocation signatures for different retriever implementations
        try:
            docs = retriever.invoke(question)
        except Exception:
            try:
                docs = retriever.invoke({"query": question})
            except Exception:
                try:
                    docs = retriever.invoke({"input": question})
                except Exception:
                    return "RESUME_TOOL_ERROR: Document retriever invoke failed." 
    else:
        return "RESUME_TOOL_ERROR: Document retriever does not support retrieval."

    if not docs:
        return "No relevant information found in the uploaded resume PDFs."

    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "uploaded_resume_pdf")
        formatted_chunks.append(
            f"Chunk {i} | Source: {source} | Page: {page}\n{doc.page_content}"
        )

    return "\n\n".join(formatted_chunks)