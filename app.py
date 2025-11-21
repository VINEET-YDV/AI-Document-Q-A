import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- Modern LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
# The chains module is in the main 'langchain' package
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Chat with PDF (Groq)", page_icon="⚡", layout="wide")

# --- Header ---
st.title("⚡ Chat with PDF")
st.markdown("""
This app uses **Groq** (llama-3.1-8b-instant) for ultra-fast responses and **HuggingFace** for local embeddings. 
""")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    # Always show the input box
    default_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input("Enter Groq API Key", value=default_key, type="password")
    
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    else:
        st.warning("⚠️ Please enter your Groq API Key.")
    
    st.divider()
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    process_button = st.button("Process PDF")

# --- Logic: PDF Processing ---
@st.cache_resource
def get_embedding_model():
    # Cache the embedding model so it doesn't reload on every run
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    try:
        # Create a temporary file to save the uploaded bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        text_chunks = text_splitter.split_documents(documents)

        # Get Embedding Model
        embeddings = get_embedding_model()
        
        # Create Vector Store
        vector_store = FAISS.from_documents(text_chunks, embeddings)
        
        # Cleanup temp file
        os.remove(tmp_file_path)
        return vector_store

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# --- State Management ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Process Button Logic ---
if process_button and uploaded_file:
    if not api_key:
        st.sidebar.error("Please provide a Groq API Key to proceed.")
    else:
        with st.spinner("Processing PDF... (This may take a moment)"):
            st.session_state.vector_store = process_pdf(uploaded_file)
            if st.session_state.vector_store:
                st.success("PDF Processed! Ready to chat.")

# --- Chat Interface ---

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question..."):
    if not st.session_state.vector_store:
        st.warning("Please upload and process a PDF first.")
    elif not api_key:
        st.warning("Please enter your Groq API Key.")
    else:
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("Groq is thinking..."):
                try:
                    # Retrieval
                    docs = st.session_state.vector_store.similarity_search(prompt, k=3)
                    
                    # Prompt Template
                    prompt_template = """
                    Answer the question based ONLY on the provided context.
                    If the answer is not in the context, say "I cannot find the answer in the document."
                    
                    Context:
                    {context}
                    
                    Question: 
                    {question}
                    
                    Answer:
                    """
                    prompt_structure = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

                    # Initialize Groq LLM (Llama 3 70B)
                    llm = ChatGroq(
                        groq_api_key=api_key, 
                        model_name="llama-3.1-8b-instant", 
                        temperature=0.1
                    )
                    
                    # Run Chain
                    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_structure)
                    response = chain.run(input_documents=docs, question=prompt)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Error: {e}")