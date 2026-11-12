from typing import Iterable
import uuid

import streamlit as st
from docx import Document as DocxFile          # reads .docx files
from pypdf import PdfReader                    # reads .pdf files
from langdetect import detect

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("🌍 Multi-Language RAG")
st.caption("Upload documents in any language, then ask questions in any supported language.")

# ---------------------------------------------------------------
# 1. Same prompt as app.py
# ---------------------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.
Answer the question using ONLY the context below.
The context may be in a different language. Translate the facts as needed.

Write your answer in {language}. Write a full sentence in {language}.
Do not answer in the language of the context unless it is {language}.
Do not repeat the question. Do not mention the context.
If the context does not contain the answer, say so in {language}.

Context:
{context}

Question:
{question}

Answer in {language}:
""".strip()
)

# ---------------------------------------------------------------
# 2. Same language detection as app.py
# ---------------------------------------------------------------
LANGUAGE_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ar": "Arabic",
    "zh-cn": "Chinese", "ja": "Japanese",
    "hi": "Hindi",
}


def detect_language(question: str) -> str:
    try:
        code = detect(question)
    except Exception:
        return "English"
    return LANGUAGE_NAMES.get(code, "the same language as the question")


def is_supported(question: str) -> bool:
    try:
        return detect(question) in LANGUAGE_NAMES
    except Exception:
        return True  # if detection fails, let it through


def format_docs(docs: Iterable[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------
# 3. Models (loaded once, not on every Streamlit rerun)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


@st.cache_resource
def get_llm():
    return ChatOllama(model="qwen2.5", temperature=0)


# ---------------------------------------------------------------
# 4. NEW: read an uploaded file and return plain text
# ---------------------------------------------------------------
def read_file(uploaded_file) -> list[Document]:
    name = uploaded_file.name
    lower = name.lower()
    docs = []

    if lower.endswith(".pdf"):
        for page_number, page in enumerate(PdfReader(uploaded_file).pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": name, "page": page_number},
                ))

    elif lower.endswith(".docx"):
        paragraphs = [p.text for p in DocxFile(uploaded_file).paragraphs]
        text = "\n".join(paragraphs)
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": name}))

    else:  # .txt
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": name}))

    return docs


# ---------------------------------------------------------------
# 5. NEW: split into chunks, then build the same Chroma + chain as app.py
# ---------------------------------------------------------------
def build_chain(uploaded_files):
    # Read every file
    documents = []
    for f in uploaded_files:
        documents.extend(read_file(f))

    if not documents:
        return None, None, 0

    # Real documents are long, so cut them into small overlapping pieces
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Same as app.py: embed the chunks and store them in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"docs_{uuid.uuid4().hex}",  # fresh collection per upload
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Same chain as app.py
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "language": lambda q: detect_language(q),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return rag_chain, retriever, len(chunks)


# ---------------------------------------------------------------
# 6. Sidebar: upload and process files
# ---------------------------------------------------------------
if "chain" not in st.session_state:
    st.session_state.chain = None
    st.session_state.retriever = None
    st.session_state.history = []

# Upload area at the top of the page (collapsible, so it doesn't take space later)
with st.expander("📄 Upload documents", expanded=st.session_state.chain is None):
    files = st.file_uploader(
        "Upload PDF, DOCX or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Process documents", type="primary", disabled=not files):
        with st.spinner("Reading and indexing..."):
            chain, retriever, n_chunks = build_chain(files)
        if chain is None:
            st.error("No text found. Scanned PDFs (images) need OCR, which this app doesn't do.")
        else:
            st.session_state.chain = chain
            st.session_state.retriever = retriever
            st.session_state.history = []
            st.success(f"Indexed {len(files)} file(s) into {n_chunks} chunks.")

# ---------------------------------------------------------------
# 7. Chat area
# ---------------------------------------------------------------
if st.session_state.chain is None:
    st.info("🔝 Upload documents in the sidebar and click **Process documents**.")
    st.stop()

# Show earlier messages
for role, text in st.session_state.history:
    st.chat_message(role).write(text)

question = st.chat_input("Ask a question in any language...")

if question:
    st.chat_message("user").write(question)
    st.session_state.history.append(("user", question))

    if not is_supported(question):
        answer = "Sorry, this language isn't supported. Please ask in English, Hindi, or another supported language."
    else:
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.chain.invoke(question)
                # Show which files the answer came from
                found = st.session_state.retriever.invoke(question)
                sources = sorted({
                    f"{d.metadata['source']}" + (f" (page {d.metadata['page']})" if "page" in d.metadata else "")
                    for d in found
                })
                answer += f"\n\n*Sources: {', '.join(sources)}*"
            except Exception as e:
                answer = f"Could not reach the model. Is Ollama running?\n\n`{e}`"

    st.chat_message("assistant").write(answer)
    st.session_state.history.append(("assistant", answer))