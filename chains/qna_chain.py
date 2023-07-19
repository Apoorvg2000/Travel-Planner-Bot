import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import tiktoken


class QNAChain:
    def token_count(text):
        tokenizer = tiktoken.get_encoding("p50k_base")

        tokens = tokenizer.encode(text, disallowed_special=())

        return len(tokens)

    def run(query):
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        doc_path = os.path.join(os.path.dirname(__file__), "../kb/")
        loader = DirectoryLoader(doc_path, glob="*.md")
        documents = loader.load()

        text_splitter = CharacterTextSplitter(
            chunk_size=200, chunk_overlap=20, length_function=QNAChain.token_count, separator="\n\n"
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(chunks, embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=300),
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
        )

        return qa.run(query)
