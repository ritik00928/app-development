import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="C++ ChatBot", page_icon="💻")

st.title("💻 C++ AI ChatBot")
st.write("Ask any question related to C++.")

# -------------------------------
# Load and Prepare Data
# -------------------------------
@st.cache_resource
def load_vector_database():

    loader = TextLoader(
        "cppTextData.txt",      # Keep the txt file in the same folder
        encoding="utf-8"
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    final_documents = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(final_documents, embedding)

    return db


db = load_vector_database()

# -------------------------------
# User Input
# -------------------------------

query = st.text_input("Enter your C++ Question")

if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a question.")
    else:

        docs = db.similarity_search(query, k=3)

        st.subheader("Relevant Answer")

        for i, doc in enumerate(docs):
            st.markdown(f"**Result {i+1}:**")
            st.write(doc.page_content)
            st.divider()