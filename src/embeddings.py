from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import os
import chromadb


def load_documents():
    documents_dir = "documents/"
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


# Esta função tem que retornar os vetores para depois se fazer os embeddings
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


def get_ids(docs):
    documents = [doc.page_content if doc.page_content is not None else "" for doc in docs]
    document_ids = [f"id{index}" for index, _ in enumerate(documents)]
    
    # for doc_id in document_ids:
    #     print(doc_id)

    return document_ids, docs



def vector_storage(embed, document_ids, documents):
    
    persistent_client = chromadb.PersistentClient()
    collection = persistent_client.get_or_create_collection("myPharma")

    # docs = [str(doc) if doc is not None else "" for doc in docs]

# collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])
    # AQUI
    docs = [doc.page_content if doc.page_content is not None else "" for doc in documents]
    collection.add(ids=document_ids, documents=docs)

    vector_store = Chroma(
        client=persistent_client, 
        collection_name="myPharma",
        embedding_function=embed,
        persist_directory="./chroma_langchain_db" # vamos ter que alterar esta diretoria
    )

    vector_store.add_documents(documents=documents, ids=document_ids)


    

    # return vector_store

    # results = collection.query(
    #     query_texts=["This is a query document about duobiotic"], # Chroma will embed this for you
    #     n_results=1 # how many results to return
    # )
    # print(results)

    results = vector_store.similarity_search(
        "This is a query document about duobiotic",
        k=1,
        # filter={"source": "tweet"},
    )
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")



def main():
    
    documents = split_documents()
    embed = OllamaEmbeddings(model="llama3.2:1b")
    document_ids, docs = get_ids(documents)
    vector_storage(embed, document_ids, docs)



    # result = collection.get()
    # embeddings = result['embeddings']
    # ids = result['ids']
    # documents = result['documents']

    # vector_storage(embed, collection)

    # ZEEEEEEEEEEEEEEEEEE

#     from langchain.embeddings import TruncateEmbedding

#     truncate_embed = TruncateEmbedding(dim=384)

# # Antes de adicionar aos documentos
#     docs_truncated = [truncate_embed(doc) for doc in documents]

#     vector_store.add_documents(documents=docs_truncated, ids=document_ids)

#     results = vector_store.similarity_search(
#         "This is a query document about duobiotic",
#         k=1,
#         embedding_dim=384  # Use a dimensão correspondente
#     )


    

    

if __name__ == "__main__":
    main()