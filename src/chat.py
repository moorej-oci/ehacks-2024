import dotenv

from langchain.memory import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


class ChatClient:
    def __init__(self):
        dotenv.load_dotenv(".env")  # load OPENAI_API_KEY
        self.history = ChatMessageHistory()
        chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant named eHacks. Answer all questions to the best of your ability.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        self.chain = prompt | chat

    def chat(self):
        self.history = ChatMessageHistory()  # reset history
        username = input("> [AI] What is your name? ")
        print(f"> [AI] Hello {username}! I am a helpful chat assistant. Ask me anything! Type '/exit' to leave.")
        while True:
            user_input = input(f"> [{username}] ")
            if user_input == "/exit":
                break

            self.history.add_message(HumanMessage(content=user_input))
            response = self.chain.invoke({"messages": self.history.messages})
            print("> [AI]", response.content)
            self.history.add_message(response)


if __name__ == "__main__":
    chat_client = ChatClient()
    chat_client.chat()
