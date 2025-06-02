import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import GPT4All
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Streamlit page setup
st.set_page_config(page_title="BioMed Chatbot", page_icon="🧬", layout="centered")
st.title("🧬 BioMed Chatbot — Offline & Easy")
st.markdown("This chatbot answers questions from your medical PDF — all offline. No internet or API needed!")

# Paths
PDF_PATH = "./documents/example.pdf"
EMBEDDING_MODEL_PATH = "./models/all-MiniLM-L6-v2"
LLM_MODEL_PATH = "./models/ggml-gpt4all-j-v1.3-groovy.bin"

# File checks
if not os.path.exists(PDF_PATH):
    st.error(f"❌ PDF not found at {PDF_PATH}. Please add your file.")
    st.stop()

if not os.path.exists(EMBEDDING_MODEL_PATH):
    st.error(f"❌ Embedding model not found at {EMBEDDING_MODEL_PATH}")
    st.info("Download it using: `python download_model.py`")
    st.stop()

if not os.path.exists(LLM_MODEL_PATH):
    st.error(f"❌ LLM model not found at {LLM_MODEL_PATH}")
    st.info("Download it from https://gpt4all.io and place it in the models folder.")
    st.stop()

# Load chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load and embed PDF
@st.cache_resource
def load_documents():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Load local LLM
@st.cache_resource
def load_llm():
    return GPT4All(model=LLM_MODEL_PATH, backend='gpt4all', verbose=False)

# Load everything
with st.spinner("Loading models..."):
    vectorstore = load_documents()
    llm = load_llm()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    verbose=False
)

# Chat interface
st.markdown("### Ask your question about the PDF")
question = st.text_input("Your question:")

if question:
    with st.spinner("Thinking..."):
        response = qa_chain({"question": question})
        st.session_state.chat_history.append((question, response["answer"]))

        st.markdown("### Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")

# Footer
st.markdown("---")
st.markdown("""
### 📌 How to Use:
1. Add a PDF file as `documents/example.pdf`
2. Run `python download_model.py` to download the embedding model
3. Manually download the GPT4All model and put it in `models/`
4. Run the chatbot with `streamlit run app.py`
""")
