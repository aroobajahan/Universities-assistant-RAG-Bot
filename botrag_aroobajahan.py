import os
import re
import json
from collections import Counter

import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
# Config

RELEVANCE_THRESHOLD = 0.45
DENSE_WEIGHT = 0.4
KEYWORD_WEIGHT = 0.6
CANDIDATE_POOL_SIZE = 20
FALLBACK_ANSWER = "Sorry, I don't know information about this question."
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "universities_dataset.json")

# Distinguishing-keyword overlap

_STOPWORDS = {
    "what", "are", "the", "top", "in", "of", "and", "or", "for", "a", "an",
    "is", "list", "give", "tell", "me", "about", "which", "universities",
    "university", "pakistan", "details", "colleges", "college", "to",
    "how", "many", "does", "do", "please", "can", "you",
}

_DISTINCTIVE_DOC_FREQ_RATIO = 0.25


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z]+", text.lower())) - _STOPWORDS


@st.cache_resource
def load_keyword_stats():
    """
    Builds a word -> document-frequency count across all dataset
    questions, used to figure out which words in a user's query are
    actually distinguishing vs generic filler.
    """
    with open(DATASET_PATH, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    doc_freq = Counter()
    total = 0
    for item in data:
        instruction = item.get("instruction", "").strip()
        if not instruction:
            continue
        total += 1
        doc_freq.update(_tokenize(instruction))

    return doc_freq, total


def _distinctive_tokens(text: str, doc_freq: Counter, total: int) -> set:
    tokens = _tokenize(text)
    max_count = max(1, int(_DISTINCTIVE_DOC_FREQ_RATIO * total))
    return {t for t in tokens if doc_freq.get(t, 0) <= max_count}


def keyword_overlap_score(question: str, candidate_instruction: str) -> float:
    doc_freq, total = load_keyword_stats()

    q_tokens = _distinctive_tokens(question, doc_freq, total)
    c_tokens = _distinctive_tokens(candidate_instruction, doc_freq, total)

    if not q_tokens or not c_tokens:
        return 0.0

    overlap = q_tokens & c_tokens
    return len(overlap) / len(c_tokens)
    
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
                page_content=instruction,
                metadata={"instruction": instruction, "answer": answer},
            )
        )

    if not docs:
        raise ValueError("Dataset loaded but contains no usable entries.")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )

    return vectorstore
    
# Hybrid retrieval
def hybrid_search(question: str):
    """
    Returns a list of (doc, dense_score, keyword_score, combined_score),
    sorted best-first by combined_score.
    """
    vectorstore = load_vectorstore()
    dense_results = vectorstore.similarity_search_with_relevance_scores(
        question, k=CANDIDATE_POOL_SIZE
    )
    scored = []
    for doc, dense_score in dense_results:
        kw_score = keyword_overlap_score(question, doc.metadata["instruction"])
        combined = DENSE_WEIGHT * dense_score + KEYWORD_WEIGHT * kw_score
        scored.append((doc, dense_score, kw_score, combined))

    scored.sort(key=lambda x: x[3], reverse=True)
    return scored

# Chat Function

def ask_rag(question: str, with_debug: bool = False):
    """
    Returns the answer string.
    If with_debug=True, returns (answer, debug_info) instead, where
    debug_info holds the top candidates and their scores - useful for
    tuning RELEVANCE_THRESHOLD / DENSE_WEIGHT / KEYWORD_WEIGHT.
    """

    question = (question or "").strip()

    def _return(answer, debug_info=None):
        if with_debug:
            return answer, (debug_info or {})
        return answer

    if not question:
        return _return(FALLBACK_ANSWER)

    try:
        scored = hybrid_search(question)

        debug_info = {
            "candidates": [
                {
                    "question": doc.metadata["instruction"],
                    "dense": round(dense, 4),
                    "keyword": round(kw, 4),
                    "combined": round(combined, 4),
                }
                for doc, dense, kw, combined in scored[:5]
            ],
            "threshold": RELEVANCE_THRESHOLD,
        }

        if not scored or scored[0][3] < RELEVANCE_THRESHOLD:
            return _return(FALLBACK_ANSWER, debug_info)

        best_doc = scored[0][0]

        answer = best_doc.metadata["answer"]

        return _return(answer, debug_info)

    except FileNotFoundError as e:
        return _return(f"Setup error: {str(e)}")

    except Exception as e:
        return _return(f"Error: {str(e)}")
