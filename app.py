# ============= STEP 1: LOAD MODULES ===============
import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ==================== STEP 2: UI & API CONFIG ======================
st.set_page_config(page_title="Flash Card Generator for Notes", layout="wide")

st.sidebar.title("SET API CONFIG")
st.title("Flash Card Generator for Notes 📚")

GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY", type="password")

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    st.sidebar.success("API key Loaded!!")
else:
    st.sidebar.info("Give API key")

# ======================= STEP 3: FILE UPLOAD ========================
uploaded_file = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])

if uploaded_file is not None and GOOGLE_API_KEY:
    save_dir = "pdf_files"
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"File uploaded: {uploaded_file.name}")

    # ===================== STEP 4: RESOURCE LOADERS ======================
    @st.cache_data
    def load_documents(path: str):
        loader = PyPDFLoader(path)
        return loader.load()

    @st.cache_resource
    def load_embedding():
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    @st.cache_data
    def get_splitted_chunks(_documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_documents(_documents)

    # Use @st.cache_resource for objects like VectorStore
    @st.cache_resource
    def create_vector_db(_chunks, _embeddings):
        return FAISS.from_documents(_chunks, _embeddings)

    # ===================== STEP 5: PROCESS PDF ====================
    with st.spinner("Processing PDF and building Vector Store..."):
        documents = load_documents(file_path)
        embeddings = load_embedding()
        chunks = get_splitted_chunks(documents)
        vectorstore = create_vector_db(chunks, embeddings)

    k_slider = st.sidebar.slider("Select Top K-Value", min_value=1, max_value=10, value=4)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_slider})

    # ======================== STEP 6: LCEL RAG CHAIN =======================
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # ================== STEP 7: USER INTERACTION ===============
    user_question = st.text_area("Ask a question about your PDF:")
    if user_question:
        if st.button("Get Answer"):
            with st.spinner("Generating answer..."):
                st.write_stream(rag_chain.stream(user_question))

elif not GOOGLE_API_KEY:
    st.warning("Please enter your Google API Key in the sidebar to proceed.")
elif not uploaded_file:
    st.info("Please upload a PDF file from the sidebar to start.")
