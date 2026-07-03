import os
import json
import torch
import pandas as pd
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load Vector Store
@st.cache_resource
def load_vectorstore():
    data_path = "universities_dataset.json"
    if not os.path.exists(data_path):
        data_path = "data/universities_dataset.json"
        
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    docs = []
    for item in data:
        combined_content = f"Context: {item['instruction']}\nDetails: {item['output']}"
        docs.append(Document(page_content=combined_content))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# 2. Load LLM Function
@st.cache_resource
def load_llm():
    model_id = "google/flan-t5-small" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=256,
        device=-1 
    )
    return HuggingFacePipeline(pipeline=pipe)

# 3. Prompt Template
template = """You are a strict academic assistant. Use ONLY the provided details text to answer the question.

If the details contain a list of universities, copy the list exactly.
If the details match a specific university, state its location and contact number exactly as written.
If the text does not contain the answer to the question, reply with exactly: "Sorry, I don't know information about this question"

Provided Text:
{context}

Question: {question}
Answer:"""

prompt_template = PromptTemplate.from_template(template)

# 4. Initialize Chain
def get_qa_chain():
    try:
        vs = load_vectorstore()
        llm_model = load_llm()
        retriever = vs.as_retriever(search_kwargs={"k": 1})
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm_model
            | StrOutputParser()
        )
        return chain
    except Exception as e:
        st.error(f"Critical Initialization Error Details: {e}")
        return None

# Global chain instance
qa_chain = get_qa_chain()

def ask_rag(query: str) -> str:
    if qa_chain is None:
        return "The system failed to initialize. Please look at the error details printed above."
    try:
        response = qa_chain.invoke(query)
        return response
    except Exception as e:
        return f"Error during search: {str(e)}"
