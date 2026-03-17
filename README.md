# RAG-Based Context-Aware Chatbot

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) based chatbot** capable of answering questions from multiple knowledge sources while maintaining general conversational abilities.

The chatbot intelligently determines the user's intent and retrieves relevant information from domain-specific knowledge bases before generating responses using a Large Language Model (LLM).

The system supports queries related to:

* **NEC Electrical Code Guidelines**
* **Wattmonk Company Information**
* **General conversational questions**

The chatbot seamlessly switches between these contexts to provide accurate and context-aware responses.

---

# Objective

The objective of this project is to build a **context-aware AI assistant** that can:

1. Answer **general conversational queries** using the base LLM knowledge.
2. Provide **technical answers from NEC Code Guidelines** when electrical standards or regulations are requested.
3. Respond to **company-specific queries about Wattmonk** using internal documentation.
4. Automatically **identify user intent and switch knowledge sources dynamically**.

---

# Key Features

### Multi-Context Query Handling

The chatbot identifies the user's query type and routes it to the appropriate knowledge source.

### Retrieval-Augmented Generation (RAG)

Instead of relying solely on LLM knowledge, the system retrieves relevant document chunks and injects them into the prompt before generating a response.

### Semantic Search

User queries are converted into embeddings and matched with relevant document embeddings using similarity search.

### Conversation Interface

Users interact with the chatbot through a simple web interface.

### Source-Aware Responses

The assistant prioritizes answers based on retrieved context from NEC or Wattmonk documents.

### Fallback Mechanism

If no relevant context is found, the assistant responds using its general knowledge.

---

# System Architecture

The system follows a **Retrieval-Augmented Generation (RAG) architecture**.

```
User Query
     │
     ▼
Intent Detection
     │
     ▼
Embedding Generation
     │
     ▼
Vector Similarity Search (ChromaDB)
     │
     ▼
Retrieve Relevant Document Chunks
     │
     ▼
Context Injection into Prompt
     │
     ▼
LLM Response Generation (Gemini API)
     │
     ▼
Response Displayed to User
```

---

# Technology Stack

### Backend

* FastAPI

### LLM Provider

* Google Gemini API

### Vector Database

* ChromaDB

### Embedding Model

* sentence-transformers/all-MiniLM-L6-v2

### Document Processing

* PyMuPDF (PDF parsing)
* LangChain text splitters

### Frontend

* HTML
* Jinja Templates
* JavaScript
  
FOR DEPLOYMENT TO HF SPACES, I UPDATED THE FRONTEND STACK TO:
* Gradio
---

# RAG Pipeline Implementation

The RAG pipeline consists of the following components.

## 1. Document Processing

Documents are loaded from the **docs/** directory and parsed using PyMuPDF.

Steps:

* Extract text from PDFs
* Split text into manageable chunks
* Prepare metadata

Chunk configuration:

* Chunk size: **1000 characters**
* Overlap: **200 characters**

This ensures contextual continuity between document segments.

---

## 2. Embedding Generation

Each document chunk is converted into a numerical vector using:

```
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings enable semantic similarity search.

---

## 3. Vector Store

All embeddings are stored in **ChromaDB**.

The vector database allows fast retrieval of the most relevant document chunks for a user query.

Search configuration:

```
Top-K Retrieval = 3
Search Type = Similarity
```

---

## 4. Query Processing

When a user sends a query:

1. The query is converted into an embedding.
2. The vector store retrieves the most relevant document chunks.
3. The retrieved context is combined into a prompt.

---

## 5. Response Generation

The system constructs a structured prompt containing:

* Retrieved document context
* User query
* Instruction prompt

This prompt is sent to **Gemini (gemini-2.5-flash)** to generate the final response.

---

# Project Structure

```
project/
│
├── main.py
├── rag_backend.py
│
├── docs/
│   ├── nec_guidelines.pdf
│   └── wattmonk_docs.pdf
│
├── chroma_db/
│
├── templates/
│   └── index.html
│
├── .env
├── requirements.txt
└── README.md
```

---

# Setup Instructions

## 1. Clone the Repository

```
git clone https://github.com/yourusername/rag-chatbot
cd rag-chatbot
```

---

## 2. Create Virtual Environment

```
python -m venv .venv
```

Activate environment:

Windows

```
.venv\Scripts\activate
```

Mac/Linux

```
source .venv/bin/activate
```

---

## 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

---

## 5. Add Knowledge Base Documents

Place NEC and Wattmonk PDFs inside the **docs/** folder.

Example:

```
docs/
├── nec_guidelines.pdf
└── wattmonk_docs.pdf
```

---

## 6. Run the Application

Start the FastAPI server:

```
uvicorn main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000
```

---

# API Endpoints

### GET /

Loads the chatbot interface.

---

### POST /ask

Processes user queries.

Example request:

```
{
  "query": "What does NEC say about grounding?"
}
```

Example response:

```
{
  "user": "What does NEC say about grounding?",
  "bot": "According to NEC Article..."
}
```

---

# Deployment

The application is deployed on:

* Hugging Face Spaces
* Link : https://huggingface.co/spaces/v01trix/wattMonkBot
   

Ensure:

* Environment variables are configured
* API keys are secured
* The vector store persists correctly

---

# Performance Considerations

* Document embeddings are cached in **ChromaDB**
* Vector store loads directly from disk on startup
* Only top relevant chunks are sent to the LLM
* Retrieval reduces token usage and improves accuracy

---

# Limitations

* Accuracy depends on document quality
* Large documents may increase indexing time
* Responses depend on retrieval quality

---

# Future Improvements

* Hybrid search (semantic + keyword)
* Query expansion
* Reranking models
* Multi-language support
* Response confidence scoring
* Document citation links

---


