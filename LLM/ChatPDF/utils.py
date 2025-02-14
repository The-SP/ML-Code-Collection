import os

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def load_pdfs(pdf_paths):
    """
    Load and process multiple PDF documents.
    """
    documents = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            print(f"Loaded {pdf_path} successfully")
        except Exception as e:
            print(f"Error loading {pdf_path}: {str(e)}")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Each chunk will be ~1000 characters
        chunk_overlap=200,  # Overlap between chunks to maintain context
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} document chunks")
    return chunks


def create_vector_store(document_chunks):
    """
    Create a vector store from document chunks.

    Args:
        document_chunks (list): List of processed document chunks
    Returns:
        FAISS: Vector store object
    """
    print("Creating FAISS vector store...")
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create a vector store
    vector_store = FAISS.from_documents(document_chunks, embeddings)

    # Optionally save the vector store locally
    vector_store.save_local("faiss_index")

    print("Vector store created successfullly")
    return vector_store


def setup_llm_and_prompt():
    """
    Set up the language model and prompt template

    Returns:
        tuple: (ChatGroq instance, ChatPromptTemplate instance)
    """
    # Initialize LLM
    llm = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0.3,
        max_tokens=512,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # Create prompt template
    system_template = """
    You are a helpful assistant for answering questions about the provided document. 
    Use the following pieces of retrieved context to answer the user's question.
    If you don't know the answer, just say "I cannot find this information in the provided documents." Don't try to make up an answer. Always base your answer on the provided context and be as specific as possible.
    
    Context: {context}
    """

    user_template = """Question: {question}"""

    # Create messages for the prompt
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template),
    ]

    # Create chat prompt template
    prompt = ChatPromptTemplate.from_messages(messages)

    return llm, prompt


def create_qa_chain(llm, vector_store, prompt):
    """
    Create the question-answering chain.

    Args:
        llm: Language model instance
        vector_store: Vector store object
        prompt: Prompt template
    Returns:
        ConversationalRetrievalChain: QA chain instance
    """
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return qa_chain


def chat_with_pdf(qa_chain):
    chat_history = []
    print("\nWelcome to PDF Chat! Type 'exit' to exit.")
    while True:
        question = input("\nQuestion: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break

        try:
            # Get response from QA chain
            result = qa_chain.invoke(
                {"question": question, "chat_history": chat_history}
            )

            # Print answer and sources
            print("\nAnswer:", result["answer"])
            print("\nSources:")
            for doc in result["source_documents"]:
                print(
                    f"- {doc.metadata['source']} [Page {doc.metadata.get('page', 'Unknown')}]"
                )

            # Update chat history
            chat_history.append((question, result["answer"]))
        except Exception as e:
            print(f"Error: {str(e)}")
