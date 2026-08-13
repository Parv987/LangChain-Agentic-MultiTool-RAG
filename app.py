import sys
from pathlib import Path

from src.rag import build_retriever
from src.tools import get_tools
from src.agent import create_multitool_agent


def main():
    print("=" * 50)
    print(" Welcome to MultiTool Agentic RAG ")
    print("=" * 50)

    resume_dir = Path("resume")

    if not resume_dir.exists() or not resume_dir.is_dir():
        print(f"❌ Error: No resume directory found at '{resume_dir}'. Please create a 'resume/' folder with one or more PDF files.")
        sys.exit(1)

    print(f"📂 Loading and indexing all PDF files from: {resume_dir}/")
    
    # 2. Build retriever and tools
    try:
        retriever = build_retriever(str(resume_dir))
        tools = get_tools(retriever)
        agent = create_multitool_agent(tools)
        print("✅ Agent ready!")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        sys.exit(1)

    print("\nAsk any question (weather, current news, or document query).")
    print("Type 'exit' or 'quit' to end the session.\n")

    # 3. Interactive CLI Loop (Replaces predefined ask_agent queries)
    while True:
        try:
            user_query = input("You > ").strip()

            # Check for exit condition
            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            # 4. Invoke the agent with user input
            response = agent.invoke({
                "messages": [
                    {"role": "user", "content": user_query}
                ]
            })

            # Extract final response message
            final_answer = response["messages"][-1].content
            print(f"\nAgent > {final_answer}\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ An error occurred while processing your request: {e}\n")

if __name__ == "__main__":
    main()