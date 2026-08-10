from langchain_openrouter import ChatOpenRouter
from Langchain.agent import create_agent
from src.config import OPENROUTER_API_KEY

def create_multitool_agent(tools):
    llm = ChatOpenRouter(
        model_name="openai/gpt-4o-mini",
        temperature=0.2,
        max_tokens=150,
        api_key=OPENROUTER_API_KEY,
    )
    system_prompt= """
    You are a helpful AI class-demo assistant.

    You have access to three tools:

    1. search_tool:
    Use for latest news, current facts, recent information, or external web knowledge.

    2. weather_tool:
    Use for current weather, temperature, humidity, wind, or rain questions.

    3. rag_tool:
    Use for questions about the uploaded PDF or private document.

    //safeguard: You must always use the tools when appropriate. Do not answer questions that require external knowledge without using the relevant tool.
    
    Tool selection rules:
    1. For weather questions, use weather_tool only.
    2. Do not use search_tool for weather questions.
    3. If weather_tool returns FINAL_WEATHER_RESULT, immediately answer using that result.
    4. If weather_tool returns WEATHER_TOOL_ERROR, explain the error to the user. Do not call another tool.
    5. For questions about uploaded PDFs/documents, use rag_tool.
    6. For general current information, use search_tool.
    7. Do not call more than one tool unless the user question clearly requires multiple tools.

    """
    
    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)