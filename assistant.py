import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # Added AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma  # Updated import

from langchain_openai import ChatOpenAI

from vector_store import create_vector_store
from prompts import contextualize_q_system_prompt, qa_system_prompt

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")
document_directory = os.path.join(current_dir, "books", "dotcom-secrets-the-underground-playbook-for-growing-your-company-online-9781630474782-2014919068.pdf")

if not os.path.exists(persistent_directory):
    create_vector_store(persistent_directory, document_directory)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Added model_name
    model_kwargs={"device": "cuda"}
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o")

contextualize_q_system_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        ("human", "{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_system_prompt,
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{input}")
    ]
)

qa_chain = create_stuff_documents_chain(
    llm, qa_prompt,
)

rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

def continual_chat():
    
    print("Start chatting with the assistant. Type 'exit' to stop.")
    chat_history = []
    
    while True:
        query = input("You: ")
 
        if query.lower() == "exit":
            print("Exiting chat.")
            break
        
        result = rag_chain.invoke(
            {
                "input": query,
                "chat_history": chat_history,
            }
        )
        print(f"Assistant: {result['answer']}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"]))  # Changed SystemMessage to AIMessage

if __name__ == "__main__":
    continual_chat()