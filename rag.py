"""
Excel Multi-Sheet RAG System with Streamlit UI and Google Gemini

Installation:
pip install streamlit pandas openpyxl sentence-transformers chromadb google-generativeai python-dotenv

Usage:
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
import google.generativeai as genai
from datetime import datetime
import os


class ExcelRAGSystem:
    """RAG system for Excel files with multiple sheets."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db"
    ):
        """Initialize the RAG system."""
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="excel_data",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.sheets_info = {}
    
    def load_excel(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Load all sheets from an Excel file."""
        excel_file = pd.ExcelFile(file_path)
        
        sheets = {}
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            sheets[sheet_name] = df
        
        return sheets
    
    def create_chunks(
        self,
        sheets: Dict[str, pd.DataFrame],
        chunk_size: int = 1
    ) -> List[Dict[str, Any]]:
        """Create text chunks from Excel data for indexing."""
        chunks = []
        chunk_id = 0
        
        for sheet_name, df in sheets.items():
            # Store sheet info
            self.sheets_info[sheet_name] = {
                'rows': len(df),
                'columns': list(df.columns)
            }
            
            # Process rows in chunks
            for start_idx in range(0, len(df), chunk_size):
                end_idx = min(start_idx + chunk_size, len(df))
                chunk_df = df.iloc[start_idx:end_idx]
                
                # Create text representation
                text_parts = [f"Sheet: {sheet_name}"]
                
                for idx, row in chunk_df.iterrows():
                    row_text_parts = []
                    for col in df.columns:
                        value = row[col]
                        if pd.notna(value):
                            row_text_parts.append(f"{col}: {value}")
                    
                    if row_text_parts:
                        text_parts.append(f"Row {idx + 2}: " + ", ".join(row_text_parts))
                
                text = "\n".join(text_parts)
                
                # Create chunk metadata
                chunk = {
                    'id': f"chunk_{chunk_id}",
                    'text': text,
                    'metadata': {
                        'sheet_name': sheet_name,
                        'start_row': start_idx,
                        'end_row': end_idx - 1,
                        'row_count': chunk_size,
                        'data': chunk_df.to_dict('records')
                    }
                }
                
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks
    
    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Create embeddings and index chunks in vector database."""
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        ids = [chunk['id'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Prepare metadata for ChromaDB
        metadatas = []
        for chunk in chunks:
            metadata = {
                'sheet_name': chunk['metadata']['sheet_name'],
                'start_row': chunk['metadata']['start_row'],
                'end_row': chunk['metadata']['end_row'],
                'data_json': json.dumps(chunk['metadata']['data'])
            }
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_sheet: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant chunks using semantic similarity."""
        # Create query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )[0]
        
        # Prepare filter
        where = {"sheet_name": filter_sheet} if filter_sheet else None
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'score': 1 - results['distances'][0][i],
                'metadata': results['metadatas'][0][i]
            }
            # Parse data back from JSON
            result['metadata']['data'] = json.loads(result['metadata']['data_json'])
            del result['metadata']['data_json']
            
            formatted_results.append(result)
        
        return formatted_results
    
    def generate_answer_with_gemini(
        self,
        query: str,
        results: List[Dict[str, Any]],
        api_key: str,
        model_name: str = "gemini-1.5-flash"
    ) -> str:
        """Generate an answer using Google Gemini API."""
        if not results:
            return "No relevant information found in the Excel data."
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Build context from results
        context_parts = []
        for i, result in enumerate(results[:5], 1):
            context_parts.append(f"\n--- Source {i} ---")
            context_parts.append(result['text'])
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are an AI assistant helping users understand their Excel data. 
Based on the following context from an Excel file, answer the user's question accurately and concisely.

Context from Excel:
{context}

User Question: {query}

Instructions:
1. Provide a clear, accurate answer based on the context
2. If you reference specific data, mention which sheet it came from
3. If the context doesn't contain enough information, say so
4. Be concise but comprehensive

Answer:"""
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer with Gemini: {str(e)}"
    
    def query(
        self,
        query: str,
        api_key: str,
        n_results: int = 5,
        filter_sheet: Optional[str] = None,
        use_gemini: bool = True
    ) -> Dict[str, Any]:
        """End-to-end query processing."""
        # Search for relevant chunks
        results = self.search(query, n_results, filter_sheet)
        
        # Generate answer
        if use_gemini and api_key:
            answer = self.generate_answer_with_gemini(query, results, api_key)
        else:
            answer = self._generate_simple_answer(query, results)
        
        return {
            'query': query,
            'answer': answer,
            'sources': results
        }
    
    def _generate_simple_answer(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Fallback answer generation without LLM."""
        if not results:
            return "No relevant information found in the Excel data."
        
        context_parts = ["Based on the Excel data, here's what I found:\n"]
        
        for i, result in enumerate(results[:3], 1):
            sheet = result['metadata']['sheet_name']
            score = result['score'] * 100
            
            context_parts.append(f"\n{i}. From sheet '{sheet}' (relevance: {score:.1f}%):")
            
            data = result['metadata']['data']
            for row in data[:2]:
                row_items = [f"{k}: {v}" for k, v in row.items() if pd.notna(v)]
                if row_items:
                    context_parts.append("   • " + ", ".join(row_items[:5]))
        
        context_parts.append(f"\n\nFound {len(results)} relevant records across the sheets.")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed data."""
        return {
            'total_chunks': self.collection.count(),
            'sheets': self.sheets_info
        }
    
    def clear_index(self):
        """Clear all indexed data."""
        try:
            self.client.delete_collection("excel_data")
            self.collection = self.client.get_or_create_collection(
                name="excel_data",
                metadata={"hnsw:space": "cosine"}
            )
            self.sheets_info = {}
        except Exception as e:
            st.error(f"Error clearing index: {str(e)}")


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Excel RAG System",
        page_icon="📊",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .stat-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .source-box {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">📊 Excel RAG System with Gemini AI</div>', unsafe_allow_html=True)
    st.markdown("Upload Excel files, ask questions, and get AI-powered answers!")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'indexed' not in st.session_state:
        st.session_state.indexed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        if api_key:
            st.success("API Key configured ✓")
        else:
            st.warning("Please enter your Gemini API key")
        
        st.divider()
        
        # Model selection
        model_name = st.selectbox(
            "Gemini Model",
            ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
            help="Choose the Gemini model to use"
        )
        
        # Search parameters
        st.subheader("Search Settings")
        n_results = st.slider("Number of results to retrieve", 1, 10, 5)
        
        st.divider()
        
        # System stats
        if st.session_state.indexed and st.session_state.rag_system:
            st.subheader("📈 System Stats")
            stats = st.session_state.rag_system.get_stats()
            st.metric("Total Chunks", stats['total_chunks'])
            st.metric("Sheets Indexed", len(stats['sheets']))
            
            with st.expander("Sheet Details"):
                for sheet_name, info in stats['sheets'].items():
                    st.write(f"**{sheet_name}**")
                    st.write(f"- Rows: {info['rows']}")
                    st.write(f"- Columns: {len(info['columns'])}")
        
        st.divider()
        
        # Clear button
        if st.button("🗑️ Clear Index", use_container_width=True):
            if st.session_state.rag_system:
                st.session_state.rag_system.clear_index()
                st.session_state.indexed = False
                st.session_state.chat_history = []
                st.success("Index cleared!")
                st.rerun()
    
    # Main content
    tab1, tab2 = st.tabs(["📤 Upload & Index", "💬 Chat"])
    
    with tab1:
        st.header("Upload Excel File")
        
        uploaded_file = st.file_uploader(
            "Choose an Excel file",
            type=['xlsx', 'xls'],
            help="Upload an Excel file with one or more sheets"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.success(f"File uploaded: {uploaded_file.name}")
            
            with col2:
                if st.button("🔄 Process & Index", use_container_width=True):
                    with st.spinner("Processing Excel file..."):
                        try:
                            # Initialize RAG system
                            st.session_state.rag_system = ExcelRAGSystem()
                            
                            # Load Excel
                            sheets = st.session_state.rag_system.load_excel(temp_path)
                            st.info(f"Loaded {len(sheets)} sheets")
                            
                            # Create chunks
                            chunks = st.session_state.rag_system.create_chunks(sheets)
                            st.info(f"Created {len(chunks)} chunks")
                            
                            # Index chunks
                            st.session_state.rag_system.index_chunks(chunks)
                            st.session_state.indexed = True
                            
                            st.success("✅ Indexing complete!")
                            st.balloons()
                            
                            # Clean up temp file
                            os.remove(temp_path)
                            
                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
            
            # Preview sheets
            if st.session_state.indexed:
                st.subheader("📋 Sheet Preview")
                
                rag = st.session_state.rag_system
                sheets = rag.load_excel(temp_path) if os.path.exists(temp_path) else {}
                
                if sheets:
                    sheet_name = st.selectbox("Select sheet to preview", list(sheets.keys()))
                    
                    df = sheets[sheet_name]
                    st.dataframe(df.head(10), use_container_width=True)
                    st.caption(f"Showing first 10 rows of {len(df)} total rows")
    
    with tab2:
        if not st.session_state.indexed:
            st.info("👈 Please upload and index an Excel file first")
        elif not api_key:
            st.warning("⚠️ Please enter your Gemini API key in the sidebar")
        else:
            st.header("Ask Questions About Your Data")
            
            # Display chat history
            for i, chat in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**🙋 You:** {chat['query']}")
                    st.markdown(f"**🤖 AI:** {chat['answer']}")
                    
                    with st.expander(f"📚 View Sources ({len(chat['sources'])} results)"):
                        for j, source in enumerate(chat['sources'], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {j}</strong> - Sheet: {source['metadata']['sheet_name']} 
                                (Relevance: {source['score']*100:.1f}%)
                                <pre>{source['text']}</pre>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.divider()
            
            # Query input
            with st.form(key="query_form", clear_on_submit=True):
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    query = st.text_input(
                        "Ask a question",
                        placeholder="e.g., What are the total sales in Q1?",
                        label_visibility="collapsed"
                    )
                
                with col2:
                    submit = st.form_submit_button("🔍 Ask", use_container_width=True)
                
                # Sheet filter
                sheet_filter = st.selectbox(
                    "Filter by sheet (optional)",
                    ["All sheets"] + list(st.session_state.rag_system.sheets_info.keys())
                )
            
            if submit and query:
                with st.spinner("Searching and generating answer..."):
                    try:
                        filter_sheet = None if sheet_filter == "All sheets" else sheet_filter
                        
                        response = st.session_state.rag_system.query(
                            query=query,
                            api_key=api_key,
                            n_results=n_results,
                            filter_sheet=filter_sheet,
                            use_gemini=True
                        )
                        
                        # Add to chat history
                        st.session_state.chat_history.append(response)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
            
            # Clear chat history
            if st.session_state.chat_history:
                if st.button("🧹 Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()


