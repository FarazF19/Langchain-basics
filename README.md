# 🧠 LangChain Basics – From Data to GenAI App

This repository demonstrates my journey into the **LangChain ecosystem**, where I explored the core building blocks needed to create **GenAI applications**. The project walks through the essential phases of:

- 🔄 Data ingestion & transformation
- 📚 Chunking and vectorizing documents
- 🧠 Embedding generation with **HuggingFace** & **OpenAI**
- 📦 Using **Vectorstores** like **FAISS** and **ChromaDB**
- 🔍 Building and using **retrievers**
- 🤖 Creating a **basic RAG (Retrieval-Augmented Generation)** GenAI app

> 📌 This project uses **LangChain**, **OpenAI GPT-4o**, **Chroma** and **FAISS** vectorstore for storing document embeddings.

---

## 🚀 What I Learned

✅ Basics of **LangChain**: From ingestion to retrieval  
✅ How to use **Ollama** for local LLMs like Llama 2 & Gemma  
✅ Creating **document loaders** from websites  
✅ Splitting text into chunks for embedding  
✅ Using **OpenAI Embeddings**  
✅ Storing & retrieving vectors with **FAISS** and **ChromaDB**  
✅ Using retrievers in chains  
✅ Building a **simple GenAI RAG app** using LangChain

---

## 🛠️ Technologies Used

| Tool / Library         | Purpose                         |
| ---------------------- | ------------------------------- |
| `LangChain`            | Framework for GenAI apps        |
| `OpenAI GPT-4o`        | LLM used in the final chain     |
| `FAISS`, `ChromaDB`    | Vector databases for embeddings |
| `dotenv`               | Securely load API keys          |
| `Ollama` (optional)    | Run open-source LLMs locally    |
| `LangSmith` (optional) | For observability & tracing     |

---

## 📦 Project Workflow

Here's the code breakdown in logical steps:

### 1. 🌐 Load Environment Variables

```python
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
2. 📰 Data Ingestion from a Website
python
Copy
Edit
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.smith.langchain.com/evaluation/tutorials/evaluation")
docs = loader.load()
3. ✂️ Text Splitting
python
Copy
Edit
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
4. 🧬 Embedding Creation using OpenAI
python
Copy
Edit
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
5. 📚 Storing Embeddings in FAISS Vector Store
python
Copy
Edit
from langchain_community.vectorstores import FAISS

vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)
6. 🔍 Search Similar Chunks
python
Copy
Edit
query = "Each datapoint should consist of, at the very least, the inputs to the application"
result = vectordb.similarity_search(query)
print(result[0].page_content)
7. 🤖 LLM Setup for Response Generation
python
Copy
Edit
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
8. 🔗 Create Document Chain with a Prompt
python
Copy
Edit
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the content:
<context>
{context}
</context>
""")

document_chain = create_stuff_documents_chain(llm, prompt)
9. 🧠 Basic Document Invocation Example
python
Copy
Edit
from langchain_core.documents import Document

document_chain.invoke({
    "input": query,
    "context": [Document(page_content=result[0].page_content)]
})
10. 🔁 Full Retrieval Chain Pipeline
python
Copy
Edit
retriever = vectordb.as_retriever()

from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": query})
print(response)
📁 Folder Structure
bash
Copy
Edit
├── .env                     # API keys and project secrets
├── main.py                  # Full code for ingestion to RAG app
├── requirements.txt         # Dependencies
└── README.md                # This file
📸 Architecture

🧑‍💻 Author
Muhammad Faraz
💼 AI Full Stack Developer
🔗 Connect on LinkedIn

I’m currently diving deep into AI Agents, LLMOps, and GenAI frameworks. Open to collaboration and freelance/remote roles.

📬 Contact
Want to collaborate or hire? Feel free to connect!
```
