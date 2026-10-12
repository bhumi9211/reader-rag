# 📚 Book Reader RAG Project

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about PDF books and receive context-based answers.

## 🛠️ Tech Stack
- **Python** – Core language
- **LangChain** – RAG pipeline
- **Google Gemini API** – Answer generation
- **Hugging Face** – Text embeddings
- **ChromaDB** – Vector storage and semantic search

## ⚙️ How It Works
1. Load PDF documents.
2. Split text into chunks.
3. Generate embeddings using Hugging Face.
4. Store embeddings in ChromaDB.
5. Retrieve relevant content and generate answers using Gemini.

## 🚀 Setup

```bash
pip install -r requirements.txt
python create_database.py
python main.py
python app.py
```

Add your Gemini API key to `.env`:

```env
GOOGLE_API_KEY=your_api_key
```
