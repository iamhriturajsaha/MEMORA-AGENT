# 📚MEMORA Agent

> Turn static documents into interactive knowledge.

MEMORA is a production-aligned RAG (Retrieval-Augmented Generation) application that lets you upload documents and explore them through natural conversation. Built with LangChain LCEL, FAISS, Hugging Face embeddings and a Groq-powered LLM.

## Features

**Core**
- Upload one or multiple PDFs directly in the UI.
- Semantic search via FAISS vector database.
- Context-aware answers powered by Groq (Llama 3.1).
- Conversational memory preserved across chat turns.
- Fast local embeddings — no OpenAI key required.
- Modern LangChain LCEL pipeline.

**UX**
- Clean chat-style interface built with Streamlit.
- Sidebar document management.
- Processing status indicators.
- Session-based conversation memory.
- Cached resources for performance.

## Architecture

```
PDF Upload → PyPDFLoader → RecursiveCharacterTextSplitter
    → HuggingFace Embeddings (all-MiniLM-L6-v2)
    → FAISS Vector Store → Top-k Retriever
    → Groq LLM (llama-3.1-8b-instant)
    → Streamlit Chat UI
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| LLM | Groq — Llama 3.1 8B Instant |
| Framework | LangChain (LCEL) |
| Embeddings | sentence-transformers |
| Vector DB | FAISS |
| UI | Streamlit |
| Deployment | Google Colab + ngrok |

## Usage

1. Launch the app.
2. Upload one or more PDF files from the sidebar.
3. Click **Process Documents**.
4. Ask questions in the chat box.
5. Continue the conversation — memory is preserved across turns.

## Quick Glance

<p align="center">
  <img src="Streamlit Images/1.png" alt="1" width="1000"/><br>
  <img src="Streamlit Images/2.png" alt="2" width="1000"/><br>
  <img src="Streamlit Images/3.png" alt="3" width="1000"/><br>
  <img src="Streamlit Images/4.png" alt="4" width="1000"/><br>
  <img src="Streamlit Images/5.png" alt="5" width="1000"/><br>
  <img src="Streamlit Images/6.png" alt="4" width="1000"/><br>
  <img src="Streamlit Images/7.png" alt="5" width="1000"/><br>
</p>

## Known Limitations

- Works best with text-based PDFs (no OCR for scanned documents).
- Vector-only retrieval (no hybrid search yet).
- Single-user session memory.

## Security

- Never hardcode API keys — always use environment variables.
- Revoke any keys accidentally committed to version control.
- Keep secrets out of your repository.

## Learning Outcomes

This project demonstrates hands-on application of - RAG architecture design, LangChain LCEL, vector database integration, prompt engineering, conversational memory, Streamlit development and LLM API integration.
