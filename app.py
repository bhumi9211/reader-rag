
import os
import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="Book RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book RAG Chatbot")
st.caption("Upload a PDF book and ask questions about its content.")

# -------------------- SESSION STATE --------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "book_hash" not in st.session_state:
    st.session_state.book_hash = None

if "book_name" not in st.session_state:
    st.session_state.book_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "llm" not in st.session_state:
    st.session_state.llm = None


# -------------------- MODELS --------------------

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Add it to your .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0.2
    )


# -------------------- PDF PROCESSING --------------------

def create_vectorstore(pdf_bytes):

    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:
        temp_file.write(pdf_bytes)
        pdf_path = temp_file.name

    try:
        # Load PDF pages
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            raise ValueError("No readable pages found in the PDF.")

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError(
                "No text could be extracted. "
                "The PDF may be scanned or image-only."
            )

        # Create embeddings
        embedding_model = get_embedding_model()

        # Use a separate database directory for this upload
        db_directory = tempfile.mkdtemp(prefix="book_chroma_")

        # Create ChromaDB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=db_directory
        )

        return vectorstore, len(documents), len(chunks)

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


# -------------------- RETRIEVER & PROMPT --------------------

template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful book assistant.

        Answer using ONLY the context provided below.

        Instructions:
        1. Explain concepts in simple English.
        2. Give a detailed answer using all relevant information
           available in the context.
        3. Use headings and bullet points when useful.
        4. Include examples if supported by the context.
        5. Do not invent information.
        6. If the answer is not available in the context,
           clearly say you could not find it in the uploaded book.

        Context:
        {context}
        """
    ),
    (
        "human",
        "Question: {question}"
    )
])


def get_answer(question):

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 15,
            "lambda_mult": 0.5
        }
    )

    docs = retriever.invoke(question)

    if not docs:
        return "I could not find relevant information in the book.", []

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = template.invoke({
        "context": context,
        "question": question
    })

    response = st.session_state.llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        answer = "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
        )
    else:
        answer = str(content)

    # Get page numbers for source references
    pages = sorted(set(
        doc.metadata.get("page", 0) + 1
        for doc in docs
    ))

    return answer, pages


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.header("📖 Upload Your Book")

    uploaded_file = st.file_uploader(
        "Choose a PDF book",
        type=["pdf"]
    )

    if uploaded_file:
        st.write("**Selected file:**")
        st.write(uploaded_file.name)

        pdf_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(pdf_bytes).hexdigest()

        if current_hash != st.session_state.book_hash:
            if st.button("Process Book", type="primary"):
                try:
                    with st.spinner(
                        "Reading book and creating embeddings..."
                    ):
                        vectorstore, page_count, chunk_count = (
                            create_vectorstore(pdf_bytes)
                        )

                        llm = get_llm()

                        st.session_state.vectorstore = vectorstore
                        st.session_state.llm = llm
                        st.session_state.book_hash = current_hash
                        st.session_state.book_name = uploaded_file.name
                        st.session_state.messages = []

                    st.success("Book processed successfully!")
                    st.info(f"Pages loaded: {page_count}")
                    st.info(f"Chunks created: {chunk_count}")

                except Exception as e:
                    st.error(f"Could not process book: {e}")

        else:
            st.success("Book is ready for questions.")

    st.divider()

    if st.session_state.book_name:
        st.caption(
            f"Current book: {st.session_state.book_name}"
        )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# -------------------- MAIN CHAT UI --------------------

if st.session_state.vectorstore is None:

    st.info(
        "👈 Upload a PDF book from the sidebar, "
        "then click **Process Book** to get started."
    )

else:

    st.subheader(f"Chatting with: {st.session_state.book_name}")

    # Display previous conversation
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("pages"):
                st.caption(
                    "Relevant PDF pages: "
                    + ", ".join(map(str, message["pages"]))
                )

    # User question input
    question = st.chat_input(
        "Ask something about your book..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:
                with st.spinner("Searching the book and generating answer..."):
                    answer, pages = get_answer(question)

                st.markdown(answer)

                if pages:
                    st.caption(
                        "Relevant PDF pages: "
                        + ", ".join(map(str, pages))
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "pages": pages
                })

            except Exception as e:
                st.error(f"Error generating answer: {e}")
