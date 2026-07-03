import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

with open("universities_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

docs = []

for item in data:
    docs.append(
        Document(
            page_content=f"""
Instruction:
{item['instruction']}

Answer:
{item['output']}
"""
        )
    )

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

vectorstore.save_local("vectorstore")

print("Vector store created successfully!")
