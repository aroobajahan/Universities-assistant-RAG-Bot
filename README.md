# University RAG Chatbot

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]((https://ragchatbot-6cknbura9trdxhtfrukyqb.streamlit.app/))
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

A specialized AI chatbot that answers questions about universities in Pakistan using **Retrieval-Augmented Generation (RAG)**. This project uses LangChain, FAISS, and the FLAN-T5-small model to provide accurate, context-aware answers from a custom dataset.

## 🚀 Live Demo
Check out the live app here: (https://ragchatbot-6cknbura9trdxhtfrukyqb.streamlit.app/)

## ✨ Features
- **Contextual Search:** Uses FAISS vector storage to find the most relevant university data.
- **AI-Powered Answers:** Leverages a 2026-optimized LangChain pipeline.
- **Lightweight Architecture:** Optimized to run on a 1GB RAM cloud environment.


##⚙️ Installation & Local Setup
Clone the repo:
git clone https://github.com/aroobajahan/RAGchatbot.git

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run appui.py
## 🛠️ Tech Stack
- **Framework:** Streamlit
- **LLM Orchestration:** LangChain Classic
- **Model:** Google FLAN-T5-Small
- **Vector Database:** FAISS (CPU)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`

## 📂 Project Structure
```text
├── appui.py                # Streamlit UI & Chat Interface
├── botrag_aroobajahan.py   # RAG Logic, LLM loading, & Retrieval
├── data/
│   └── universities_dataset.json  # Custom University Data
├── requirements.txt        # Project Dependencies
└── README.md               # This file!
