import streamlit as st
import pandas as pd
import openai
import faiss
import numpy as np
import time
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import sys
from typing import List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Error handling decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            return None
    return wrapper

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(3))
@handle_errors
def get_embeddings(texts: List[str], embedding_model) -> List[List[float]]:
    """Get embeddings with retry logic and error handling"""
    return embedding_model.embed_documents(texts)

@st.cache_resource(ttl=3600)  # Cache for 1 hour
@handle_errors
def create_faiss_index(df: pd.DataFrame) -> Tuple[faiss.Index, List[str], OpenAIEmbeddings]:
    """Create FAISS index with improved error handling and caching"""
    try:
        # Initialize embedding model
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=st.secrets["OPENAI_API_KEY"]
        )

        # Convert dataframe to text
        texts = []
        for _, row in df.iterrows():
            formatted_row = row.copy()
            for col in row.index:
                if pd.api.types.is_numeric_dtype(df[col]):
                    formatted_row[col] = f"{row[col]:,.2f}"
            
            text = " | ".join(f"{col}: {formatted_row[col]}" for col in df.columns)
            texts.append(text)

        # Create chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n", " | ", ". ", " ", ""]
        )
        chunks = text_splitter.create_documents(texts)
        chunk_texts = [doc.page_content for doc in chunks]

        # Process embeddings in smaller batches
        embeddings = []
        batch_size = 25  # Reduced batch size
        
        with st.spinner("Processing data... This may take a few minutes."):
            for i in range(0, len(chunk_texts), batch_size):
                batch = chunk_texts[i:i + batch_size]
                batch_embeddings = get_embeddings(batch, embedding_model)
                embeddings.extend(batch_embeddings)
                time.sleep(0.1)  # Rate limiting
                
                # Update progress
                progress = (i + batch_size) / len(chunk_texts)
                st.progress(min(progress, 1.0))

        # Create FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))

        return index, chunk_texts, embedding_model

    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        st.error("Failed to initialize the search index. Please try again.")
        raise

@st.cache_data(ttl=300)  # Cache for 5 minutes
def query_faiss(user_query: str, index, texts: List[str], embedding_model) -> str:
    """Query FAISS index with improved error handling"""
    try:
        # Generate query embedding
        query_embedding = embedding_model.embed_query(user_query)
        
        # Search index
        k = 3
        _, indices = index.search(np.array([query_embedding], dtype=np.float32), k)
        
        # Get context
        context = "\n".join([texts[i] for i in indices[0]])
        
        # Generate response
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

# Streamlit UI
try:
    st.title("Sales Data Analysis Assistant")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])
    
    if uploaded_file:
        # Load data
        df = pd.read_excel(uploaded_file)
        
        # Initialize FAISS index if not already done
        if not st.session_state.initialized:
            with st.spinner("Initializing AI model..."):
                st.session_state.index, st.session_state.texts, st.session_state.embedding_model = create_faiss_index(df)
                st.session_state.initialized = True
        
        # Query interface
        user_query = st.text_input("What would you like to know about the sales data?")
        if user_query:
            with st.spinner("Analyzing data..."):
                response = query_faiss(
                    user_query,
                    st.session_state.index,
                    st.session_state.texts,
                    st.session_state.embedding_model
                )
                st.write(response)

except Exception as e:
    logger.error(f"Application error: {str(e)}")
    st.error("An unexpected error occurred. Please refresh the page and try again.")
