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


RELEVANCE_THRESHOLD = 0.45

FALLBACK_ANSWER = "Sorry, I don't know information about this question."

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "universities_dataset.json")

# Load Dataset + Build FAISS


@st.cache_resource
def load_vectorstore():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}\n"
            "Make sure 'universities_dataset.json' is inside a 'data/' "
            "folder next to this script."
        )

    with open(DATASET_PATH, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    docs = []
    for item in data:
        instruction = item.get("instruction", "").strip()
        answer = item.get("output", "").strip()

        if not instruction or not answer:
            continue

        docs.append(
            Document(
                page_content=f"Instruction:\n{instruction}\n\nAnswer:\n{answer}",
                metadata={"instruction": instruction, "answer": answer},
            )
        )

    if not docs:
        raise ValueError("Dataset loaded but contains no usable entries.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

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
        do_sample=False, 
        device=-1, 
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm

# Prompt

template = """You are an assistant for answering university-related questions.

Use ONLY the retrieved context below to answer. Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chat Function


def ask_rag(question: str):

    question = (question or "").strip()
    if not question:
        return FALLBACK_ANSWER

    try:
        vectorstore = load_vectorstore()
        results = vectorstore.similarity_search_with_relevance_scores(
            question, k=3
        )

        relevant = [
            (doc, score)
            for doc, score in results
            if score >= RELEVANCE_THRESHOLD
        ]

        if not relevant:
            return FALLBACK_ANSWER

        docs = [doc for doc, _ in relevant]
        context = format_docs(docs)

        llm = load_llm()
        chain = prompt | llm

        answer = chain.invoke({"context": context, "question": question})

        answer = answer.strip() if isinstance(answer, str) else str(answer).strip()

        if not answer:
            return FALLBACK_ANSWER

        return answer

    except FileNotFoundError as e:
        return f"Setup error: {str(e)}"

    except Exception as e:
        return f"Error: {str(e)}"
