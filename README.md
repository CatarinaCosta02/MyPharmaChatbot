# MyPharmaChatbot

**MyPharmaChatbot** is an AI virtual assistant developed as part of the Master's Informatics Project course (2024/2025). This project is a collaboration between University of Minho students, **OmniumAI**, and **MyPharma S.A.**, a Portuguese pharmaceutical company.

The primary **goal** is to provide detailed information about the company’s food supplements (Condotril, Neurofil, Duobiotic) to all types of users, ensuring accurate responses based on official technical documentation.

**Course:** PEI

## Key Features
- RAG-based system;
- Language normalization pipeline that removes special characters and accents to improve Portuguese spell checking.
- Spell Checking (```SpellChecker``` tool)
- Similarity matching (```fuzzywuzzy```)

## Architecture 

<p align="center">
  <img src="https://github.com/CatarinaCosta02/MyPharmaChatbot/blob/main/Documents/Project%20Architecture.png" width="600">
</p>

## Chatbot Workflow

<p align="center">
  <img src="https://github.com/CatarinaCosta02/MyPharmaChatbot/blob/main/Documents/ChatbotWorkflow.png" width="300">
</p>

## Tech Stack

- **Language:** Python
- **Vector Database:** FAISS
- **LLM Models:** Llama3.1:8b (Ollama) & GPT-4o mini (OpenAI)
- **Framework:** LangChain
- **Environment:** Jupyter Notebook.

## Structure

| Folder | Description |
| :--- | :--- |
| `Documents` | Final Report|
| `chatbot` | Chatbot Jupyter notebooks and food supplements documentation  |
| `chroma` | Experimental environment files for ChromaDB integration|
| `other` | Technical scripts metadata extraction and Node.js API layer for front-end system integration.|

## Chatbot Team Members
Catarina Costa, pg52676

Inês D. Ferreira, pg53870

Marta Aguiar, pg52694
