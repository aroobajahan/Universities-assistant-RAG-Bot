import os
import json
import torch
import pandas as pd
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_core.prompts import PromptTemplate

# 1. Load Vector Store
@st.cache_resource
def load_vectorstore():
    data_path = "data/universities_dataset.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    documents = df["output"].tolist()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# 2. Load LLM Function
@st.cache_resource
def load_llm():
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=256,
        device=-1 
    )
    return HuggingFacePipeline(pipeline=pipe)

# 3. Define the Prompt Template
template = """[SYSTEM ROLE]
You are a strict academic assistant. You must answer questions using ONLY the provided text under [CONTEXT]. Do not use outside knowledge.

[INSTRUCTION 1 - RANKING LISTS]
If the user asks for a list, ranking, top 10, or category of universities:
- Look at the [CONTEXT] text block.
- Extract the specific list of ranked universities provided there.
- Display them clearly as a numbered list.

[INSTRUCTION 2 - SPECIFIC UNIVERSITIES]
If the user asks about a specific university or its contact details:
- Find that exact university name in the [CONTEXT].
- Output the location and contact details exactly as written.

[INSTRUCTION 3 - ABSENCE OF INFORMATION]
If the [CONTEXT] does not contain the exact list requested, or does not contain details about the requested university, you must reply with exactly:
"Sorry, I don't know information about this question"

[Context]
{context}

[Question] 
{question}
[Answer]"""

prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

# 4. Initialize Chain func
def get_qa_chain():
    try:
        vs = load_vectorstore()
        llm_model = load_llm()
        
        return RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="stuff",
            retriever=vs.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=False
        )
    except Exception as e:
        st.error(f"Critical Initialization Error: {e}")
        return None

# global instance
qa_chain = get_qa_chain()

def ask_rag(query: str) -> str:
    if qa_chain is None:
        return "The system failed to initialize. Please check the error message above."
    try:
        # .invoke() 
        response = qa_chain.invoke({"query": query})
        return response["result"]
    except Exception as e:
        return f"Error during search: {str(e)}"
