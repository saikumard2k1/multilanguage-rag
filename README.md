# 🌍 Multi-Language RAG

A RAG (Retrieval-Augmented Generation) app that answers questions in the **user's language**, even when the documents are in a **different language**. No translation step is needed.

**Live demo:** [ADD YOUR APP URL HERE](ADD_YOUR_APP_URL_HERE)

> The demo sleeps when idle. If you see a "wake up" button, click it and wait a few seconds. Uploaded files are kept in memory only, so they reset when the app restarts.

## How it works

1. **Multilingual embeddings** (`paraphrase-multilingual-MiniLM-L12-v2`) place text from 50+ languages in one shared vector space. A Hindi question can find an English, French or German passage without translating anything.
2. **ChromaDB** stores the document chunks and returns the 3 closest to the question.
3. **Language detection** (`langdetect`) finds the language of the question in code, and the prompt names it explicitly. Small models follow "Answer in Hindi" much better than "answer in the user's language".
4. **An LLM** writes the answer from the retrieved text, in the question's language.
5. If the documents don't contain the answer, the app says so instead of guessing.

```
Question ──► detect language ──┐
   │                           ▼
   └─► embed ─► Chroma (top 3 chunks) ─► prompt ─► LLM ─► answer in your language + sources
```

## Files

| File | What it is |
|---|---|
| `app.py` | Command-line version with sample documents. Uses a local LLM through Ollama (Qwen 2.5). |
| `rag_web.py` | Streamlit web app for PDF, DOCX and TXT uploads. Uses Ollama locally. |
| `rag_web_groq.py` | The same web app using the Groq API. **This is the deployed version.** |
| `requirements.txt` | Python dependencies. |
| `.env.example` | Template for environment variables. Copy it to `.env` and add your real key. |

## Supported question languages

English, Hindi, German, French, Spanish, Italian, Portuguese, Arabic, Chinese and Japanese.

A question in any other language gets a "language isn't supported" message, because small models answer unreliably in languages they know less well.

## Run it locally

### 1. Set up Python

```bash
python -m venv venv
venv\Scripts\activate        # Windows (cmd)
# source venv/bin/activate   # Mac / Linux
pip install -r requirements.txt
```

### 2a. Local LLM with Ollama (no API key)

Install [Ollama](https://ollama.com/download), then:

```bash
ollama pull qwen2.5
python app.py                 # command-line chat
streamlit run rag_web.py      # web app
```

### 2b. Hosted LLM with Groq (no GPU or big download)

1. Create an API key at [console.groq.com](https://console.groq.com).
2. Copy `.env.example` to `.env` and fill it in:

   ```
   GROQ_API_KEY=your_key_here
   GROQ_MODEL=openai/gpt-oss-120b
   ```

3. Run:

   ```bash
   streamlit run rag_web_groq.py
   ```

Groq changes its model list from time to time. If you see "model not found", check the [supported models page](https://console.groq.com/docs/models) and update `GROQ_MODEL`.

## Using the web app

1. Open **Upload documents** and add PDF, DOCX or TXT files.
2. Click **Process documents**. The files are read, split into chunks and indexed.
3. Ask a question in any supported language. The answer comes back in your language, with the source file (and page for PDFs) listed underneath.

## Deployment

The app is deployed on **Streamlit Community Cloud** from `rag_web_groq.py`.

- Add `GROQ_API_KEY` and `GROQ_MODEL` under **Advanced settings > Secrets**.
- Use Python 3.12.
- Never put your API key in the code or commit your `.env` file. It is listed in `.gitignore`.

## Limitations

- Only `.docx` is supported, not the old `.doc` format. Save as `.docx` first.
- Scanned PDFs (images of pages) have no text layer, so they need OCR, which is not included.
- Text in `.docx` tables and text boxes is skipped. Only normal paragraphs are read.
- Only the top 3 chunks are retrieved per question. Questions about the whole document, such as "list all chapters", will give weak answers.
- The index is in memory and resets on restart.
- Smaller local models can produce broken characters in languages like Hindi. The Groq version handles these better.

## Future improvements

**Retrieval quality**
- **Whole-document questions.** Add a summary or "table of contents" mode, or a map-reduce step, so questions like "list all chapters" can use the entire file instead of only the top 3 chunks.
- **Hybrid search.** Combine keyword search (BM25) with vector search, which helps with exact terms, names and numbers.
- **Re-ranking.** Retrieve more chunks, then re-rank them with a cross-encoder before sending the best ones to the LLM.
- **Smarter chunking.** Split by headings and sections instead of a fixed 800 characters, and keep chapter titles in the chunk metadata.

**Documents**
- **OCR for scanned PDFs** (for example with Tesseract), so image-only files can be read.
- **More file types**: `.doc`, `.pptx`, `.xlsx`, `.csv`, HTML pages and web URLs.
- **Better `.docx` reading**, including tables and text boxes.
- **Persistent storage.** Save the index to disk or a hosted vector database (such as Chroma server, Qdrant or Pinecone) so documents survive restarts and users don't re-upload.

**Answers and chat**
- **Conversation memory**, so follow-up questions like "and what about page 3?" work.
- **Streaming answers**, so text appears word by word.
- **Highlighted citations** that show the exact passage used, not only the file and page.
- **Better handling of short questions.** Language detection can be unreliable on one or two words, so add a language selector or a stronger detector.
- **More languages**, tested one at a time with a larger model, including Indian regional languages such as Telugu and Tamil.

**App and deployment**
- **Model picker** in the UI to switch between Groq models, or between Groq and Ollama.
- **User accounts and private workspaces**, so each person has their own documents.
- **Usage limits and retry logic** for API rate limits.
- **Lower memory use** with a lighter embedding model (for example an ONNX-based one such as `fastembed`), so the app fits on smaller hosting plans.
- **Evaluation set.** A small set of test questions with expected answers in several languages, to catch regressions when changing models or prompts.
- **Docker image and CI**, for repeatable deployments and automatic tests on each push.

## Built with

[LangChain](https://www.langchain.com/) · [ChromaDB](https://www.trychroma.com/) · [Sentence Transformers](https://www.sbert.net/) · [Streamlit](https://streamlit.io/) · [Ollama](https://ollama.com/) · [Groq](https://groq.com/)
