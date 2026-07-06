# University RAG Chatbot

A chatbot that answers questions about universities in Pakistan using Retrieval-Augmented Generation (RAG), built with LangChain, FAISS, and FLAN-T5-small.

## Project structure
```
├── appui.py                        # Streamlit UI & chat interface
├── botrag_aroobajahan.py           # RAG logic, LLM loading, retrieval
├── data/
│   └── universities_dataset.json   # Q&A dataset
├── requirements.txt
└── README.md
```

## Installation & local setup
```
git clone https://github.com/aroobajahan/Universities-assistant-RAG-Bot.git
cd Universities-assistant-RAG-Bot
pip install -r requirements.txt
streamlit run appui.py
App link: https://universities-assistant-rag-bot-c4q6jgqg7edqushbm2uasw.streamlit.app/
```

## Tech stack
Framework: Streamlit
Retrieval: LangChain + FAISS (hybrid: dense embeddings + fuzzy keyword re-ranking)
Embeddings: HuggingFace all-MiniLM-L6-v2
