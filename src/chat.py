from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.1,
    num_predict = 256
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
messages = chain.invoke(
    {
        "input_language": "English",
        "output_language": "Portuguese",
        "input": "I love programming.",
    }
)

# for chunk in llm.stream(messages):
#     print(chunk)

print(messages.content)