from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os
import re
from tqdm import tqdm


def load_documents():
    documents_dir = "documents/"
    target_folders = ["Condotril", "Duobiotic", "Neurofil"]
    documents = []
    for folder in target_folders:
        folder_path = os.path.join(documents_dir, folder)
        
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for doc_name in os.listdir(folder_path):
                if doc_name.endswith(".txt"):
                    print(f"txt encontrado ({doc_name})!")
                    file_path = os.path.join(folder_path, doc_name)
                    
                    with open(file_path, "r", encoding="ISO-8859-1") as doc:
                        content = doc.read()
                        documents.append(content)
    return documents


# Esta função tem que retornar os vetores para depois se fazer os embeddings
def split_documents():
    
    
    # index_dimension = 4096
    # vectors_dictionary = {}
    # vector = []
    # vector_count = 0
    # line_count = 0

    embed = OllamaEmbeddings(
        model="llama3.2:1b"
    )
    
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    all_documents_content = "\n\n".join(load_documents())
    print(len(all_documents_content)) # caracteres
    docs = text_splitter.create_documents([all_documents_content])

    print(len(docs)) # chunks
    
    print("Documentos 1\n", docs[0].page_content)
    print("Documentos 2", docs[1].page_content)
    print("Documentos 3", docs[2].page_content)

    #content_Condotril = load_documents("documents/Condotril", "Condotril")
    # content_Duobiotic = load_documents("documents/Condotril", "Condotril")
    # content_Neurofil = load_documents("documents/Condotril", "Condotril")

    #texts = text_splitter.create_documents([content_Condotril])

    #print(texts[0].page_content)


    # if medicamento=="Condotril":
    #     for doc in Condotril_docs:
    #         for line in doc.split('\n'):
    #             line_count += 1
    #             line = (str(line)).lower()
    #             line = (re.sub(r'[^\w\s]', '', line)).strip()
    #             vector.append(line)
    #             if len(vector) >= index_dimension:
    #                 vector_count += 1
    #                 vectors_dictionary[vector_count] = vector
    #                 vector = []
    #         if vector:
    #             vectors_dictionary[vector_count] = vector

    # if medicamento=="Duobiotic":
    #     for doc in Duobiotic_docs:
    #         for line in doc.split('\n'):
    #             line_count += 1
    #             line = (str(line)).lower()
    #             line = (re.sub(r'[^\w\s]', '', line)).strip()
    #             vector.append(line)
    #             if len(vector) >= index_dimension:
    #                 vector_count += 1
    #                 vectors_dictionary[vector_count] = vector
    #                 vector = []
    #         if vector:
    #             vectors_dictionary[vector_count] = vector

    # if medicamento=="Neurofil":
    #     for doc in Neurofil_docs:
    #         for line in doc.split('\n'):
    #             line_count += 1
    #             line = (str(line)).lower()
    #             line = (re.sub(r'[^\w\s]', '', line)).strip()
    #             vector.append(line)
    #             if len(vector) >= index_dimension:
    #                 vector_count += 1
    #                 vectors_dictionary[vector_count] = vector
    #                 vector = []
    #         if vector:
    #             vectors_dictionary[vector_count] = vector

    # print(f"{medicamento} splited!")


def main():
    
    split_documents()
    # documents_Condotril = []
    # documents_Duobiotic = []
    # documents_Neurofil = []

    # documents_dir = "documents/"

    # target_folders = ["Condotril", "Duobiotic", "Neurofil"]

    # for folder in target_folders:
    #     folder_path = os.path.join(documents_dir, folder)
        
    #     if os.path.exists(folder_path) and os.path.isdir(folder_path):
    #         for doc_name in os.listdir(folder_path):
    #             if doc_name.endswith(".txt"):
    #                 file_path = os.path.join(folder_path, doc_name)
                    
    #                 with open(file_path, "r", encoding="ISO-8859-1") as doc:
    #                     content = doc.read()
                        
    #                     if folder == "Condotril":
    #                         documents_Condotril.append(content)
    #                     elif folder == "Duobiotic":
    #                         documents_Duobiotic.append(content)
    #                     elif folder == "Neurofil":
    #                         documents_Neurofil.append(content)

    # print(f"Total de documentos em Condotril: {len(documents_Condotril)}")
    # print(f"Total de documentos em Duobiotic: {len(documents_Duobiotic)}")
    # print(f"Total de documentos em Neurofil: {len(documents_Neurofil)}")

    # split_documents("Condotril", documents_Duobiotic, documents_Neurofil, documents_Neurofil)
    # split_documents("Duobiotic", documents_Duobiotic, documents_Neurofil, documents_Neurofil)
    # split_documents("Neurofil", documents_Duobiotic, documents_Neurofil, documents_Neurofil)


    # for doc in documents_Condotril:
    #     print(doc)


    # embed = OllamaEmbeddings(
    #     model="llama3.2:1b"
    #)
    # ...

    # embedded_vectors = []
    # for _, vector in tqdm(vectors_dictionary.items(), desc="Embedding Progress"):
    #     # Embed the vector
    #     embedded_vector = embed.embed_documents(vector)
    #     print(f'embedded_vector: \n{embedded_vector}')
    #     print(f'len(embedded_vector): \n{len(embedded_vector)}')

    #     embedded_vectors.append(embedded_vector)

    # print(f'\nlen(embedded_vectors): {len(embedded_vectors)}')

    # print(f'len(embedded_vectors) : {len(embedded_vectors)}')
    # for vec in embedded_vectors:
    #     print(f'len(vec) : {len(vec)}\n')

    # embedded_vectors = [item for sublist in embedded_vectors for item in sublist]
    

if __name__ == "__main__":
    main()