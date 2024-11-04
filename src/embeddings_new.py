import os
import getpass
import sys
import time
import uuid
import logging
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter 
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

from typing import Any, Dict, Iterable
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_core.embeddings import Embeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def load_documents():
    documents_dir = "../documents/" # Catarina
    # documents_dir = "documents/"  # Marta
    target_folders = ["Condotril", "Duobiotic", "Neurofil"]  # nomes dos produtos
    documents = {folder: "" for folder in target_folders}  # Inicializa um dicionário para cada produto

    for folder in target_folders:
        folder_path = os.path.join(documents_dir, folder)
        
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for doc_name in os.listdir(folder_path):
                if doc_name.endswith(".txt"):
                    print(f"txt encontrado ({doc_name})!")
                    file_path = os.path.join(folder_path, doc_name)
                    with open(file_path, "r", encoding="utf-8") as doc:
                        content = doc.read()
                        documents[folder] += content + "\n"
    return documents

def split_documents():
    
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=4000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    documents = load_documents()
    print("Número de documentos: ", len(documents))
    docs = []
    for content in documents.values():
        docs.extend(text_splitter.create_documents([content]))  # 

    for _, (product, content) in enumerate(documents.items()):
        num_characters = len(content)
        print(f"Número de caracteres no documento '{product}': {num_characters}")



    print("Número de chunks: ", len(docs))
    print("\n\n")
    print("Documento 1\n", docs[0].page_content, "\n", "-"*80)
    print("Documento 2\n", docs[1].page_content, "\n", "-"*80)
    print("Documento 3\n", docs[2].page_content, "\n", "-"*80)

    return docs

def chatbot_first_message(vectorstore):
    question = "Tell me something about Duobiotic"

    docs = vectorstore.similarity_search(question, k=2)

    # docs = [doc.page_content if doc.page_content is not None else "" for doc in documents]

    context = "\n".join([doc.page_content for doc in docs])

    llm = OllamaEmbeddings(model="llama3.2:1b")


    template = """
    Answer the question based on the context below. If you can't 
            answer the question, reply "I don't know". If the 
            question has nothing to do with the context, 
            answer the question normally.

            Context: {context}

            Question: {question}
    """

    template = ChatPromptTemplate.from_template(template)
    template.format(context=context, question=question)
    embedding_dict = prompt(text=template)
    chain = parser.run(embedding_dict)
    parser = StrOutputParser()

    chain = prompt | llm | parser
    response = str(chain.invoke({
        "context": context,
        "question": question
    }).get('text', ''))

    print("Contexto:", context) 
    print("Resposta:", response)

    return question, response

def setup_pinecone_environment():
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
    
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    
    return pinecone_api_key


def create_index(pc, index_name):

    logger = logging.getLogger(__name__)

    existing_indexes = [
        index_info["name"] for index_info in pc.list_indexes()
    ]


    spec = ServerlessSpec(
        cloud="aws",
        # region="eu-central-1"
        region='us-east-1'
    )


    if index_name not in existing_indexes:
        try:
            pc.create_index(
                name=index_name,
                dimension=2048,
                metric='cosine',
                spec=spec
            )
            
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
        
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            return None
    
    
    print(pc.Index(index_name).describe_index_stats())

    index = pc.Index(index_name)
    
    return index



def main():
    
    documents = split_documents()
    model = OllamaEmbeddings(model="llama3.2:1b")

    pinecone_api_key = setup_pinecone_environment()
    if not pinecone_api_key:
        raise ValueError("API key not found")
    
    pc = Pinecone(api_key=pinecone_api_key)
    
    index_name = "lala"
    index = create_index(pc, index_name)

    docs = [doc.page_content if doc.page_content is not None else "" for doc in documents]
    # para os ids
    uuids = [str(uuid.uuid4()) for _ in range(len(documents))]

    if index:
        vector_store = PineconeVectorStore(index=index, embedding=model, text_key="text")
        print("Índice criado com sucesso!")
    else:
        print("Falha na criação do índice.")

    # DESCOMENTAR A FUNÇAO QUE É PRETENDIDA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    vector_store.add_documents(documents=documents, ids=uuids) # comentar depois de adicionar a primeira vez para nao duplicar informação
    # vector_store._index.delete(delete_all=True) # apagar todos os documentos da base de dados
    print("Documentos armazenados com sucesso!")
    print("chegou aqui\n\n")

    question, response = chatbot_first_message(vector_store)
    print("Pergunta:", question)
    print("Resposta:", response)
    



if __name__ == "__main__":
    main()
