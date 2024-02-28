import pathlib
from operator import itemgetter
from typing import Any

import dotenv
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool


class LLMClient:
    def __init__(self):
        dotenv.load_dotenv(".env")  # load environment variable OPENAI_API_KEY
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.db = SQLDatabase.from_uri(self.get_db_uri())
        self.generate_query_chain = create_sql_query_chain(self.llm, self.db)
        self.write_and_execute_query_chain = self.create_chain()
        self.advanced_chain = self.create_advanced_chain()
        self.agent = create_sql_agent(self.llm, db=self.db, agent_type="openai-tools", verbose=True)

    @staticmethod
    def get_resources_path() -> pathlib.Path:
        return pathlib.Path(__file__).parent.parent / "resources"

    def get_db_uri(self) -> str:
        resources_path = self.get_resources_path()
        return f"sqlite:///{resources_path.as_posix()}/Chinook_Sqlite.sqlite"

    def create_chain(self) -> RunnableSerializable:
        execute_query = QuerySQLDataBaseTool(db=self.db)
        write_query = create_sql_query_chain(self.llm, self.db)
        chain = write_query | execute_query
        return chain

    def create_advanced_chain(self):
        answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer: """
        )
        execute_query = QuerySQLDataBaseTool(db=self.db)
        write_query = create_sql_query_chain(self.llm, self.db)
        answer = answer_prompt | self.llm | StrOutputParser()
        chain = (
                RunnablePassthrough.assign(query=write_query).assign(
                    result=itemgetter("query") | execute_query
                )
                | answer
        )
        return chain

    def generate_query(self, question: str) -> str:
        return self.generate_query_chain.invoke({"question": question})

    def answer_question(self, question: str) -> str:
        return self.write_and_execute_query_chain.invoke({"question": question})

    def answer_question_advanced(self, question: str) -> str:
        return self.advanced_chain.invoke({"question": question})

    def answer_question_agent(self, question: str) -> dict[str, Any]:
        return self.agent.invoke({"input": question})


if __name__ == "__main__":
    client = LLMClient()
    question = "How many employees are there?"

    query = client.generate_query(question)
    print(f"{question=} produces {query=}")

    num_employees = client.answer_question(question)
    print(f"{question} There are {num_employees=}")

    answer = client.answer_question_advanced(question)
    print(f"{question} {answer}")

    results = client.answer_question_agent(question)
    print(results)
