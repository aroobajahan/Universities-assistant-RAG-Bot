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
cd RAGchatbot
pip install -r requirements.txt
streamlit run appui.py
```

## Tech stack
- **Framework:** Streamlit
- **LLM orchestration:** LangChain 0.2
- **Model:** Google FLAN-T5-Small
- **Vector database:** FAISS (CPU)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`

## How relevance is handled
Before the LLM is ever called, the bot retrieves the top matches from the
dataset and checks a relevance score (0-1). If nothing clears the
`RELEVANCE_THRESHOLD` in `botrag_aroobajahan.py`, it replies with the
fallback message instead of asking the LLM to guess. This keeps answers
tied to the dataset instead of hallucinating on unrelated questions.

If you find it's saying "I don't know" too often, lower
`RELEVANCE_THRESHOLD`; if it's answering things it shouldn't, raise it.

## Deploying on Streamlit Community Cloud
1. Push this whole folder (including `data/universities_dataset.json`) to your GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), point the app at `appui.py`.
3. First load will be slow (downloading the FLAN-T5-small and MiniLM models) - this is normal.
4. Free tier is limited to 1GB RAM. If the app keeps crashing/rebooting after deploying, that's usually RAM exhaustion from loading both models at once - see notes below.
