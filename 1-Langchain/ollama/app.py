import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import Ollama
from langchain_core.output_parsers import StrOutputParser




os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")  ## Fpr Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

load_dotenv()

## Prompt Template
prompt= ChatPromptTemplate([
    ("system","You are an assistant,Please respond to the question asked."),
    ("user","Question:{question}")
 
])

## Streamlit framework
st.title("Langchain App with Gemma")

input_text= st.text_input("What question have u in mind?")

## Run the Ollama llm
llm= Ollama(model="gemma3:1b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))