import os
import textwrap
from pathlib import Path  
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from llama_parse import LlamaParse  

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")


def print_response(response) :
    response_text = response["result"]
    for chunk in response_text.split("\n") :
        if not chunk :
            print()
            continue
        print("\n".join(
            textwrap.wrap(chunk, 100, break_long_words=False)
            )
        )


def llama3RAG(markdown_document, user_question, chunk_size=2048, chunk_overlap_size=128, retriever_k_value=5, llm_model_name=None, temperature=0, max_new_tokens=0, **kwargs) :
    if kwargs["parse_function"] :
        print("[SYS INFO] ----> Running the RAG Pipeline, please wait.....{}".format("\n"))
        loader = UnstructuredMarkdownLoader(
            markdown_document
        )

        loaded_documents = loader.load()
        print("[SYS INFO] ----> Splitting the text into chunks for the VectorDB....{}".format("\n"))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size, 
            chunk_overlap = chunk_overlap_size,
        )

        docs = text_splitter.split_documents(loaded_documents)
        print("[INFO] ----> Total Length of the documents is :- {}".format(len(docs)))

        print("{}[SYS INFO] ----> Loading the embedding model for the LLM...{}".format("\n", "\n"))
        embeddings = FastEmbedEmbeddings(model_name = "BAAI/bge-base-en-v1.5")

        qdrant = Qdrant.from_documents(
            docs, 
            embeddings, 
            path="./db",
            collection_name="document_embeddings",
        )

        user_querry = user_question
        similar_docs = qdrant.similarity_search_with_score(
            user_querry
        )

        print("[SYS INFO] ----> Top Results With Cosine Similarity")
        for doc, score in similar_docs :
            print(f"Text :- {doc.page_content[:256]} \n")
            print(f"Score :- {score}")
            print("=" * 100)
            print()

        print("\n")
        print("[SYS INFO] ----> Top Results With Retriever....")
        retriever = qdrant.as_retriever(
            search_kwargs={"k" : retriever_k_value}
        )
        retrieved_docs = retriever.invoke(user_querry)

        for doc in retrieved_docs :
            print(f"ID :- {doc.metadata["_id"]} \n")
            print(f"Text :- {doc.page_content[:256]} \n")
            print("=" * 100)
            print()

        print("\n")
        print("[SYS INFO] ----> Loading the Reranker Model...{}".format("\n"))
        compressor = FlashrankRerank(
            model="ms-marco-MiniLM-L-12-v2"
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=retriever,
        )
        reranked_documents = compression_retriever.invoke(user_querry)
        print("[SYS INFO] ----> Length of the Reranked Documents :- {}".format(len(reranked_documents)))
        print("==" * 100)
        print("{}Top Results :- {}".format("\n", "\n"))

        for doc in reranked_documents :
            print(f"ID :- {doc.metadata["_id"]} \n")
            print(f"Text :- {doc.page_content[:256]} \n")
            print(f"Relevance Score :- {doc.metadata['relevance_score']}")
            print("=" * 100)
            print()

        print("\n")
        print("[SYS INFO] ----> Invoking the LLMs using Groq Model...{}".format("\n"))
        llm = ChatGroq(
            model_name = llm_model_name, 
            temperature = temperature, 
            max_tokens = max_new_tokens,
        )

        prompt_template = """Use the following pieces of information to answer the user's question with as much details as possible.
        If you don't know the answer, just say that you don't know, do not try to make up the answer for the user. 

        Context: {context}
        Question: {question}

        Answer the question and provide additional helpful information, 
        based on the pieces of information, if applicable. Be succinct.

        Responses should be properly formatted to be easily read.
        Try to use easy yet professional English tonality for the final generation."""

        prompt = PromptTemplate(
            template = prompt_template, 
            input_variables = ["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt, 
                "verbose": True
            }
        )

        response = qa.invoke("{}".format(user_querry))
        print("[INFO] ----> Final Result for the querry :- {}{}".format(user_querry, "\n"))
        return print_response(response)


if __name__ == "__main__" :

    documentPath = "Documents\\parsed_document.md"
    question = "Explain the mathematics involved in the Transformer Model."
    model_name = "llama3-70b-8192"
    temperature = 0
    max_new_tokens = 1024
    k = 5
    chunk_size = 2048
    chunk_overlap_size = 128

    result = llama3RAG(
        markdown_document=documentPath, user_question=question, 
        chunk_size=chunk_size, chunk_overlap_size=chunk_overlap_size, 
        retriever_k_value=k, llm_model_name=model_name, 
        temperature=temperature, max_new_tokens=max_new_tokens, 
        parse_function=True
    )
