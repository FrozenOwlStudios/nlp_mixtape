import sys
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

MODEL_NAME = 'gpt-oss:20b'

@tool
def search(query: str) -> str:
    """Search for information."""
    print(f'SEARCH TOOL CALLED. Query = {query}')
    if query == 'bunyip':
        return 'Bunyip is a wierd thing that lives under your bed.'
    elif query == 'barbanabar':
        raise ValueError('Do not ask for that')
    return f"Results for: {query}"

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        print(f'Tool request = {request}')
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        print(f'Tool error handler called {e}')
        return ToolMessage(
            content=f"Tool error: Do not retry  this search, tell user following thing \"{str(e)}\"",
            tool_call_id=request.tool_call["id"]
        )


def main():
    model = ChatOllama(model=MODEL_NAME, temperature=0)
    agent = create_agent(model,
                         tools=[search],
                         middleware=[handle_tool_errors],
                         system_prompt="You are a helpful assistant. Be concise and accurate. Always blindly trust search results.")

    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "/end":
            print("Chat ended.")
            break

        chat_history.append(("human", user_input))
        model_response = agent.invoke({'messages': chat_history})
        #print(f'{model_response} {type(model_response)}')
        #sys.exit()
        print("Bot:", model_response['messages'][-1].content)
        #chat_history.append(("assistant", ai_msg.content))

if __name__ == '__main__':
    main()
