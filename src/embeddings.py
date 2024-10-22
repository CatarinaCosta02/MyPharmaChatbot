from langchain_ollama import OllamaEmbeddings
import os
import re



# Esta função tem que retornar os vetores para depois se fazer os embeddings
def split_documents(medicamento, Condotril_docs, Duobiotic_docs, Neurofil_docs):
    index_dimension = 4096
    vectors_dictionary = {}
    vector = []
    vector_count = 0
    line_count = 0

    if medicamento=="Condotril":
        for doc in Condotril_docs:
            for line in doc.split('\n'):
                line_count += 1
                line = (str(line)).lower()
                line = (re.sub(r'[^\w\s]', '', line)).strip()
                vector.append(line)
                if len(vector) >= index_dimension:
                    vector_count += 1
                    vectors_dictionary[vector_count] = vector
                    vector = []
            if vector:
                vectors_dictionary[vector_count] = vector

    if medicamento=="Duobiotic":
        for doc in Duobiotic_docs:
            for line in doc.split('\n'):
                line_count += 1
                line = (str(line)).lower()
                line = (re.sub(r'[^\w\s]', '', line)).strip()
                vector.append(line)
                if len(vector) >= index_dimension:
                    vector_count += 1
                    vectors_dictionary[vector_count] = vector
                    vector = []
            if vector:
                vectors_dictionary[vector_count] = vector

    if medicamento=="Neurofil":
        for doc in Neurofil_docs:
            for line in doc.split('\n'):
                line_count += 1
                line = (str(line)).lower()
                line = (re.sub(r'[^\w\s]', '', line)).strip()
                vector.append(line)
                if len(vector) >= index_dimension:
                    vector_count += 1
                    vectors_dictionary[vector_count] = vector
                    vector = []
            if vector:
                vectors_dictionary[vector_count] = vector

    # print(f"{medicamento} splited!")


def main():

    documents_Condotril = []
    documents_Duobiotic = []
    documents_Neurofil = []

    documents_dir = "documents/"

    target_folders = ["Condotril", "Duobiotic", "Neurofil"]

    for folder in target_folders:
        folder_path = os.path.join(documents_dir, folder)
        
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            for doc_name in os.listdir(folder_path):
                if doc_name.endswith(".txt"):
                    file_path = os.path.join(folder_path, doc_name)
                    
                    with open(file_path, "r", encoding="ISO-8859-1") as doc:
                        content = doc.read()
                        
                        if folder == "Condotril":
                            documents_Condotril.append(content)
                        elif folder == "Duobiotic":
                            documents_Duobiotic.append(content)
                        elif folder == "Neurofil":
                            documents_Neurofil.append(content)

    print(f"Total de documentos em Condotril: {len(documents_Condotril)}")
    print(f"Total de documentos em Duobiotic: {len(documents_Duobiotic)}")
    print(f"Total de documentos em Neurofil: {len(documents_Neurofil)}")

    split_documents("Condotril", documents_Duobiotic, documents_Neurofil, documents_Neurofil)
    split_documents("Duobiotic", documents_Duobiotic, documents_Neurofil, documents_Neurofil)
    split_documents("Neurofil", documents_Duobiotic, documents_Neurofil, documents_Neurofil)


    embed = OllamaEmbeddings(
        model="llama3.2:1b"
    )
    # ...


if __name__ == "__main__":
    main()