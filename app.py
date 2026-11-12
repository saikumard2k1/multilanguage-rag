from typing import Iterable

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langdetect import detect

# Our multilingual knowledge base
documents = [
    Document(
        page_content="The company policy allows for up to 20 days of paid time off per year. Employees must request time off at least two weeks in advance.",
        metadata={"language": "English", "source": "HR_Policy_EN"},
    ),
    Document(
        page_content="La politique de l'entreprise accorde jusqu'à 20 jours de congés payés par an. Les employés doivent en faire la demande au moins deux semaines à l'avance.",
        metadata={"language": "French", "source": "HR_Policy_FR"},
    ),
    Document(
        page_content="El soporte técnico está disponible 24/7. Para problemas urgentes, llame al número de emergencia en lugar de enviar un correo electrónico.",
        metadata={"language": "Spanish", "source": "IT_Policy_ES"},
    ),
    Document(
        page_content="Spesenabrechnungen müssen innerhalb von 30 Tagen nach der Reise eingereicht werden. Hotelkosten werden bis zu 150 Euro pro Nacht erstattet.",
        metadata={"language": "German", "source": "Travel_Policy_DE"},
    ),
    Document(
        page_content="सभी कर्मचारियों को स्वास्थ्य बीमा मिलता है। बीमा की कवरेज सीमा 5 लाख रुपये प्रति वर्ष है।",
        metadata={"language": "Hindi", "source": "Benefits_Policy_HI"},
    ),
    Document(
        page_content="La mensa aziendale è aperta dal lunedì al venerdì, dalle 12:00 alle 14:30.",
        metadata={"language": "Italian", "source": "Facilities_Policy_IT"},
    ),
    Document(
        page_content="O horário de trabalho é das 9h às 18h, com uma hora de almoço.",
        metadata={"language": "Portuguese", "source": "Work_Hours_PT"},
    ),
    Document(
        page_content="Employees may work remotely up to three days per week with manager approval.",
        metadata={"language": "English", "source": "Remote_Policy_EN"},
    ),
]

# The prompt tells the LLM how to behave across languages
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


# Helper function to combine retrieved documents into a single string
def format_docs(docs: Iterable[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# A model trained to map different languages into the same vector space
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Load the documents into a local Chroma vector database
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="multilingual_hr_docs",
)

# Fetch the top 3 most relevant documents for each query
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize the local LLM (Ollama must be running)
llm = ChatOllama(model="qwen2.5", temperature=0)

# Build the pipeline: question -> retriever -> prompt -> LLM -> text
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


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "language": lambda q: detect_language(q),
    }
    | prompt
    | llm
    | StrOutputParser()
)


def is_supported(question: str) -> bool:
    try:
        return detect(question) in LANGUAGE_NAMES
    except Exception:
        return True  # if detection fails, let it through


def main() -> None:
    print("\nMulti-language RAG ready. Ask in any language. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not is_supported(question):
            print("AI: Sorry, this language isn't supported. Please ask in English, Hindi, or another supported language.\n")
            continue
        print(f"AI: {rag_chain.invoke(question)}\n")


if __name__ == "__main__":
    main()