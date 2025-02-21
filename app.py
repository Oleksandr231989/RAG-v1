import streamlit as st
import pandas as pd
import openai
import faiss
import numpy as np
import os
import logging
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Column Descriptions for Improved Context
COLUMN_DESCRIPTIONS = {
    "Country": "name of the country",
    "Date": "date in the format date by months",
    "Competitive market": "main category/group/market",
    "Submarket 1": "subcategory for competitive market",
    "Submarket 2": "subcategory 2 for competitive market",
    "Sales (local currency)": "total sales in local currency",
    "Sales (Euro)": "total sales in Euro",
    "Brand": "the name of the brand",
    "Product": "SKU or product name",
    "Corporation": "company, corporation, manufacturer name"
}

# Load Excel file directly in the code (Replace with actual file path)
FILE_PATH = "data/sales_data.xlsx"
df = pd.read_excel(FILE_PATH)

# Function to create embeddings and store in FAISS
@st.cache_resource(ttl=3600)
def create_faiss_index(df: pd.DataFrame) -> Tuple[faiss.Index, List[str], OpenAIEmbeddings]:
    try:
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai.api_key)
        
        texts = [" | ".join(f"{col}: {row[col]}" for col in df.columns) for _, row in df.iterrows()]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["\n", " | ", ". ", " ", ""])
        chunks = text_splitter.create_documents(texts)
        chunk_texts = [doc.page_content for doc in chunks]

        batch_size = 25
        embeddings = []
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i + batch_size]
            embeddings.extend(embedding_model.embed_documents(batch))
        
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))
        
        return index, chunk_texts, embedding_model
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        raise

index, texts, embedding_model = create_faiss_index(df)

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(3))
def query_faiss(user_query: str, index, texts: List[str], embedding_model) -> str:
    try:
        query_embedding = embedding_model.embed_query(user_query)
        k = 3
        _, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
        context = "\n".join([texts[i] for i in indices[0]])
        
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing sales data. Provide specific insights and numbers from the data."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error querying FAISS: {str(e)}")
        return f"Error processing query: {str(e)}"

st.title("Sales Data Analysis Assistant")
user_query = st.text_input("What would you like to know about the sales data?")
if user_query:
    response = query_faiss(user_query, index, texts, embedding_model)
    st.write(response)
