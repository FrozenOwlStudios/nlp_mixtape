from langchain_ollama import ChatOllama

LARGE_MODEL = 'gpt-oss:20b'
SMALL_MODEL = 'llama3.1'

llm = ChatOllama(model=SMALL_MODEL, temperature=0)

system_prompt = "You are a helpful assistant."

chat_history = [("system", system_prompt)]

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "/end":
        print("Chat ended.")
        break

    chat_history.append(("human", user_input))
    ai_msg = llm.invoke(chat_history)
    print("Bot:", ai_msg.content)
    chat_history.append(("assistant", ai_msg.content))
