from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings

load_dotenv()

DB_PATH = ".chromadb"
EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text")

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory=DB_PATH,
    embedding_function=EMBEDDINGS,
).as_retriever()


def _ingesting():
    """
    This function is responsible for ingesting data into the Chroma database.

    It uses a WebBaseLoader to load documents from a specified URL, then splits the text using a RecursiveCharacterTextSplitter.
    The embeddings are generated using the OllamaEmbeddings model and stored in a Chroma collection with the name "rag-chroma".

    Returns:
        None
    """

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=EMBEDDINGS,
        persist_directory=DB_PATH,
    )

    print("Vectorstore created successfully.")


if __name__ == "__main__":
    print("\nIngesting...")

    _ingesting()
