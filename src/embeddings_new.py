import os
import getpass
import time
import uuid
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import CharacterTextSplitter 
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.retrievers import PineconeHybridSearchRetriever

from pinecone_text.sparse import BM25Encoder

import nltk
nltk.download('punkt_tab')


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
        chunk_size=4000, # o chunk_size tem que ser grande para termos o maximo de informaçao sobre um produto num vetor apenas
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

def chatbot_first_message(vectorstore, retriever):
    question = "Diz-me algo sobre o Condotril"

    # k is the number of chunks to retrieve
    #retriever = vectorstore.as_retriever(k=4)

    # docs = retriever.invoke(question)

    # db = FAISS.from_documents(texts, embeddings)

    docs = retriever.vectorstore.similarity_search(question, k=2)
    # docs = retriever.get_relevant_documents(question, k=2)
    

    # retriever = db.as_retriever(search_type="mmr")
    # docs = [doc.page_content if doc.page_content is not None else "" for doc in documents]

    context = "\n".join([doc.page_content for doc in docs])
    # print("context: ", context)

    # llm = OllamaEmbeddings(model="llama3.2:1b")
    llm = OllamaLLM(model="llama3.2:1b")


    template = """ Responde à questão com base no contexto indicado. Se não conseguires
        responder à pergunta, responde "Não sei". Se a questão não tem nada
        a ver com o contexto, responde à pergunta normalmente, com base em 
        informações que saibas.

        Context: {context}

        Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    prompt.format(context=context, question=question)
    parser = StrOutputParser()
    
    chain = prompt | llm | parser

    response = str(chain.invoke({
        "context": context,
        "question": question
    }).strip())


    # print("\nContexto:", context) 
    # print("\nResposta:", response)

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
    # vector_store.add_documents(documents=documents, ids=uuids) # comentar depois de adicionar a primeira vez para nao duplicar informação
    # vector_store._index.delete(delete_all=True) # apagar todos os documentos da base de dados
    print("Documentos armazenados com sucesso!")
    print("chegou aqui\n\n")

    # Configurar o encoder esparso (BM25)
    bm25_encoder = BM25Encoder().default()
    corpus = ["Condotril", "Duobiotic", "Neurofil"]

    bm25_encoder.fit(["Condotril", "Duobiotic", "Neurofil"])
    print("\nfit feito com sucesso")

    bm25_encoder.dump("bm25_values.json")

    bm25_encoder = BM25Encoder().load("bm25_values.json")


    # Configurar o retriever híbrido
    retriever = PineconeHybridSearchRetriever(
        embeddings=model, 
        sparse_encoder=bm25_encoder, 
        index=index
    )

    
    retriever.add_texts(corpus)
    # Realizar a pergunta inicial
    # question, response = chatbot_first_message(vector_store)
    result = retriever.invoke("Condotril")
    print(result[0])
    

    #question, response = chatbot_first_message(vector_store)
    # print("\nPergunta:", question)
    # print("\nResposta:", response)
    



if __name__ == "__main__":
    main()
