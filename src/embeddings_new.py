import os
import getpass
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
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_elasticsearch import ElasticsearchRetriever

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


def load_documents():
    # documents_dir = "../documents/" # Catarina
    documents_dir = "documents/"  # Marta
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

    # documents = "\n\n".join(load_documents())
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

# def get_ids(docs):
#     documents = [doc.page_content if doc.page_content is not None else "" for doc in docs]
#     document_ids = [f"id{index}" for index, _ in enumerate(documents)]
    
#     # for doc_id in document_ids:
#     #     print(doc_id)

#     return document_ids, docs

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


# results = vector_store.similarity_search(
#     "Tell me something about Duobiotic",
#     k=2,
# )
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")



def initialize_retriever():

    index_name = "test-langchain-retriever"
    text_field = "text"
    dense_vector_field = "fake_embedding"
    num_characters_field = "num_characters"
    texts = [
        "foo",
        "bar",
        "world",
        "hello world",
        "hello",
        "foo bar",
        "bla bla foo",
    ]


def vector_query(search_query: str) -> Dict:
    dense_vector_field = "fake_embedding"

    embeddings = OllamaEmbeddings(model="llama3.2:1b")

    vector = embeddings.embed_query(search_query)  # same embeddings as for indexing
    return {
        "knn": {
            "field": dense_vector_field,
            "query_vector": vector,
            "k": 5,
            "num_candidates": 10,
        }
    }



def es_create_index(
    es_client: Elasticsearch,
    index_name: str,
    text_field: str,
    dense_vector_field: str,
    num_characters_field: str,
):
    es_client.indices.create(
        index=index_name,
        mappings={
            "properties": {
                text_field: {"type": "text"},
                dense_vector_field: {"type": "dense_vector"},
                num_characters_field: {"type": "integer"},
            }
        },
    )


def es_index_data(
    es_client: Elasticsearch,
    index_name: str,
    text_field: str,
    dense_vector_field: str,
    embeddings: Embeddings,
    texts: Iterable[str],
    refresh: bool = True,
) -> None:
    es_create_index(
        es_client, index_name, text_field, dense_vector_field, num_characters_field # nao sei como resolver isto
    )

    vectors = embeddings.embed_documents(list(texts))
    requests = [
        {
            "_op_type": "index",
            "_index": index_name,
            "_id": i,
            text_field: text,
            dense_vector_field: vector,
            num_characters_field: len(text),
        }
        for i, (text, vector) in enumerate(zip(texts, vectors))
    ]

    bulk(es_client, requests)

    if refresh:
        es_client.indices.refresh(index=index_name)

    return len(requests)




def main():
    documents = split_documents()
    embed = OllamaEmbeddings(model="llama3.2:1b")

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
        vector_store = PineconeVectorStore(index=index, embedding=embed)
        print("Índice criado com sucesso!")
    else:
        print("Falha na criação do índice.")

    # DESCOMENTAR A FUNÇAO QUE É PRETENDIDA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # vector_store.add_documents(documents=documents, ids=uuids) # comentar depois de adicionar a primeira vez para nao duplicar informação
    # vector_store._index.delete(delete_all=True) # apagar todos os documentos da base de dados
    print("Documentos armazenados com sucesso!")
    print("chegou aqui\n\n")
    
    # # Testes de retriever do pinecone
    # results = vector_store.similarity_search(
    #     "Tell me something about Duobiotic",
    #     k=2
    # )
    # for res in results:
    #     print(f"* {res.page_content} [{res.metadata}]")


    # results2 = vector_store.similarity_search_with_score(
    #     "Quais são os ingredientes do Neurofil?", 
    #     k=1
    # )
    # for res, score in results2:
    #     print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
    

    # retriever = vector_store.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={"k": 2, "score_threshold": 0.0001},
    # )
    # retriever.invoke("Quais são os ingredientes do Neurofil?")



    
    
    # RETRIEVER

    # os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
    # os.environ["LANGSMITH_TRACING"] = "true"

    es_endpoint = "https://324ecfa0ea11408f9c1d397bb89dad15.us-east-1.aws.found.io:443"
    es_api_key = "MXExTDJwSUJpQ0tFUXJESU5HZTY6TWxWSXpsSHFReXFSS0ExczFZWm56Zw=="
    # es_url = "http://localhost:9200"
    # es_client = Elasticsearch(hosts=[es_url]) # local
    es_client = Elasticsearch(
        es_endpoint,
        api_key=es_api_key
    )
    print(es_client.info())




    es_index_name = "test-langchain-retriever"
    text_field = "text"
    dense_vector_field = "fake_embedding"
    num_characters_field = "num_characters"
    texts = [
        "foo",
        "bar",
        "world",
        "hello world",
        "hello",
        "foo bar",
        "bla bla foo",
    ]

    es_index_data(es_client, es_index_name, text_field, dense_vector_field, embed, texts, refresh=True)


    # print("conexao com elasticsearch com sucesso!")

    # initialize_retriever()

    text_field = "text"

    vector_retriever = ElasticsearchRetriever.from_es_params(
        index_name=es_index_name,
        body_func=vector_query,
        content_field=text_field,
        # url=es_endpoint,
        api_key=es_api_key
    )

    vector_retriever.invoke("Condotril")


    # usar o retriever em chain
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context provided.

        Context: {context}

        Question: {question}"""
    )

    llm = ChatOpenAI(model="gpt-4o-mini") # podemos alterar este modelo

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": vector_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    chain.invoke("Quais são os benefícios do Condotril?")



if __name__ == "__main__":
    main()
