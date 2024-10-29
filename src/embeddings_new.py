import os
import getpass
import time
import uuid
import logging
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter 
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

def load_documents():
    documents_dir = "../documents/"
    target_folders = ["Condotril", "Duobiotic", "Neurofil"] # se calhar arranjar forma de generalizar esta parte
    documents = []
    for folder in target_folders:
        folder_path = os.path.join(documents_dir, folder)
        
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for doc_name in os.listdir(folder_path):
                if doc_name.endswith(".txt"):
                    # print(f"txt encontrado ({doc_name})!")
                    file_path = os.path.join(folder_path, doc_name)
                    with open(file_path, "r", encoding="utf-8") as doc:
                        content = doc.read()
                        documents.append(content)
    return documents

def split_documents():
    
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    documents = "\n\n".join(load_documents())
    print("Número de caracteres: ", len(documents))
    docs = text_splitter.create_documents([documents])

    print("Número de chunks: ", len(docs))
    
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
        region="eu-central-1"
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

    vector_store.add_documents(documents=documents, ids=uuids)
    print("Documentos armazenados com sucesso!")
    print("chegou aqui")


if __name__ == "__main__":
    main()
