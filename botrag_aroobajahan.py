import os
import json
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load Dataset + Build FAISS

@st.cache_resource
def load_vectorstore():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "universities_dataset.json")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_path}"
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    for item in data:

        docs.append(
            Document(
                page_content=f"""
Instruction:
{item["instruction"]}

Answer:
{item["output"]}
"""
            )
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings,
    )

    return vectorstore

# Load LLM

@st.cache_resource
def load_llm():

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0,
        device=-1,
    )

    llm = HuggingFacePipeline(
        pipeline=pipe,
    )

    return llm

# Prompt

template = """
You are an assistant for answering university-related questions.

Use ONLY the retrieved context.

If the answer is not present in the context, reply exactly:

Sorry, I don't know information about this question.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate.from_template(template)

# QA Chain

@st.cache_resource
def get_chain():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = load_llm()

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# Chat Function

def ask_rag(question: str):

    try:

        chain = get_chain()

        answer = chain.invoke(question.strip())

        if answer.strip() == "":
            return "Sorry, I don't know information about this question."

        return answer.strip()

    except Exception as e:

        return f"Error: {str(e)}"
