import google.genai as genai
import os
from dotenv import load_dotenv

# --- LangChain & PDF Processing Imports ---
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --- RAG Functions ---
def load_and_index_documents(docs_directory="documents"):
    """
    Loads documents from a directory, splits them into chunks, and indexes them
    into a ChromaDB vector store. This version saves the embeddings to disk.
    """
    # Define the directory where the vector store will be saved
    persist_directory = "./chroma_db"
    
    # Create an embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Check if the vector store already exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Vector store already exists. Loading from disk...")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        print("Vector store loaded successfully.")
    else:
        print("Vector store not found. Indexing documents...")
        
        # Load documents from the 'docs' directory
        documents = []
        for file_name in os.listdir(docs_directory):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(docs_directory, file_name)
                loader = PyMuPDFLoader(file_path)
                documents.extend(loader.load())

        if not documents:
            print("No PDF files found in the 'docs' directory.")
            return None

        print(f"Loaded {len(documents)} pages from PDFs.")

        # Split documents into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  #---------1000 so that It fits safely inside most embedding + LLM limit
            chunk_overlap=200,  #------------overlap is 200 tokens so that context from previous chunk isnot lost
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")

        # Index the chunks into ChromaDB and persist to disk
        print("Indexing documents into ChromaDB...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )

        
        print("Indexing complete and saved to disk.")
    
    return vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 3} ) # Semantic retrieval (top-k)


def call_gemini_api(prompt: str) -> str:
    """
    Gemini SDK.
    """

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "API key not found. Set GEMINI_API_KEY."

    # Configure once per call (safe for hackathon)
    client = genai.Client()
    try:
        response = client.models.generate_content(
            model= "gemini-2.5-flash",
            contents=prompt
        )
        return response.text

    except Exception as e:
        print("Gemini Error:", e)
        return "Error generating response."

def generate_rag_response(query: str, retriever) -> str:
    """
    Generates a response using the RAG approach with the vector store retriever.
    """
    # Step 1: Retrieval
    relevant_docs = retriever.invoke(query)  #Pass retrieved context to the LLM
    
    if not relevant_docs:
        return "I'm sorry, I couldn't find any relevant information in the documents."

    # Combine retrieved documents into a single string
    context = "\n---\n".join([doc.page_content for doc in relevant_docs])

    # Step 2: Prompt Construction
    prompt_template = PromptTemplate.from_template("""
You are a context-aware AI assistant designed to answer user queries by intelligently combining general knowledge with retrieved knowledge.

Use the retrieved context when answering the question.

================ CONTEXT ================
{context}
========================================

User Question:
{query}

### Knowledge Sources
You have access to two knowledge bases:

1. NEC Code Guidelines
2. Wattmonk Company Information

### Rules
- If the answer exists in the provided context, use it.
- If context is insufficient, respond using general knowledge.
- Do NOT fabricate NEC or company policies if they are not in context.
- Mention the source if possible.
- If the answer doesn't exist in context, answer it but don't mention that it is not there in context.

Provide a clear and helpful answer.
""")
    


    prompt = prompt_template.format(context=context, query=query)

    # Step 3: Call the LLM
    return call_gemini_api(prompt)

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Initialize the vector store and retriever once at the start of your application
    retriever_instance = load_and_index_documents()
    flag = True

    if retriever_instance:
        while flag:
            test_query = input("Q:")
            print(f"\nUser Query: {test_query}")
            if test_query == 'q':
                flag = False
                exit()
            response = generate_rag_response(test_query, retriever_instance)
            print(f"Assistant Response: {response}")

       