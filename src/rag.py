# Lots of content pulled from langchain quickstart
# https://python.langchain.com/docs/use_cases/question_answering/quickstart

import dotenv
from langchain import hub
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class RAGClient:
    def __init__(self):
        dotenv.load_dotenv(".env")  # load OPENAI_API_KEY
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.embedding = OpenAIEmbeddings()
        self.prompt = hub.pull("rlm/rag-prompt")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    @staticmethod
    def load_docs(web_paths: tuple[str]) -> list[Document]:
        # Load, chunk and index the contents of the blog.
        loader = WebBaseLoader(
            web_paths=web_paths,
        )
        return loader.load()

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def load_vectorstore_from_web(self, web_paths: tuple[str]) -> Chroma:
        docs = self.load_docs(web_paths=web_paths)
        splits = self.text_splitter.split_documents(docs)
        return Chroma.from_documents(documents=splits, embedding=self.embedding)

    def create_rag_chain(self, vectorstore: Chroma):
        # Retrieve and generate using the relevant snippets of the blog.
        retriever = vectorstore.as_retriever()

        return (
                {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )


def ask_question(question: str, rag_chain):
    print(question)
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n\n")


def main():
    web_urls = ("https://ai.google/responsibility/principles/",)

    client = RAGClient()
    vectorstore = client.load_vectorstore_from_web(web_urls)
    rag_chain = client.create_rag_chain(vectorstore)

    ask_question("What is responsible AI use?", rag_chain)

    ask_question("Explain responsible AI use like I am five years old", rag_chain)

    ask_question("Please list the principles for responsible AI use in bullet points", rag_chain)

    ask_question("What should AI applications not pursue?", rag_chain)

    # cleanup
    vectorstore.delete_collection()


if __name__ == "__main__":
    main()
