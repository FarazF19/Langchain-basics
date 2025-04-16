import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Streamlit UI
st.set_page_config(page_title="LangChain GenAI Demo", layout="centered")
st.title("🔗 GenAI App with LangChain and OPENAI")
st.markdown("Ask a question based on scraped LangChain documentation.")

# Load and process data once
@st.cache_resource
def load_vectorstore():
    loader = WebBaseLoader("https://docs.smith.langchain.com/evaluation/tutorials/evaluation")
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create FAISS vectorstore
    vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)
    return vectordb

vectordb = load_vectorstore()

# Set up retriever
retriever = vectordb.as_retriever()

# LLM & prompt setup
llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the content:
<context>
{context}
</context>
""")
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User query input
user_input = st.text_input("🔎 Enter your question here:", placeholder="e.g., What is a datapoint in LangChain eval?")
if user_input:
    with st.spinner("💡 Generating answer..."):
        response = retrieval_chain.invoke({"input": user_input})

    st.subheader("📌 Answer")
    st.write(response['answer'])

    # st.subheader("📄 Retrieved Context")
    # for i, doc in enumerate(response['context']):
    #     with st.expander(f"Document {i+1}"):
    #         st.write(doc.page_content)
