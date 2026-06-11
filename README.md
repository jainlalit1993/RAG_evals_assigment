
# 🏋️ RAG Fitness Agent - Complete RAG System Tutorial

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A production-ready Retrieval-Augmented Generation (RAG) system built for AI Builder Bootcamp students**

Built by **Nisarg Kadam** | AI Trainer & Educator | [LinkedIn](https://linkedin.com/in/nisargkadam) | [YouTube](https://youtube.com/@cognithicai)

---

## 🎯 What You'll Build

A complete RAG system that answers questions from your own documents with:

- ✅ **4 Advanced Chunking Strategies** (Token-based, Semantic, Agentic, Recursive)
- ✅ **Vector Database** (ChromaDB with local embeddings)
- ✅ **LangGraph Workflow** (Single-agent state machine)
- ✅ **Grounded Answers** (No hallucination - cites sources)
- ✅ **Production-Ready** (Proper error handling, logging, configuration)

---

## 🎬 Demo

**Question:** "How much protein in 1 cup cottage cheese?"

**Answer:**
> "According to the provided information, 1 cup of 1% milkfat cottage cheese contains **28 grams of protein**."
> 
> 📄 *Source: Nutrition_Guide.pdf (Page 1)*

**Question:** "How do I build a rocket ship?" *(Not in documents)*

**Answer:**
> "I don't have information about that in the provided documents."

---

## 🏗️ Architecture
```
┌─────────────┐
│   PDF Docs  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐     ┌──────────────┐
│  Text Chunking  │────▶│  Embeddings  │
│  (4 strategies) │     │  (Local AI)  │
└─────────────────┘     └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  ChromaDB    │
                        │ Vector Store │
                        └──────┬───────┘
                               │
    ┌──────────────────────────┘
    │
    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  User Query  │────▶│   Retrieve   │────▶│   Generate   │
└──────────────┘     │  (LangGraph) │     │  (GPT-3.5)   │
                     └──────────────┘     └──────┬───────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │    Answer    │
                                          │ with Sources │
                                          └──────────────┘
```

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one free](https://platform.openai.com/api-keys))
- 1GB free disk space

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/NisargKadam/rag-fitness-agent.git
cd rag-fitness-agent

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### Run the System
```bash
# Step 1: Add your PDF files to data/pdfs/ folder

# Step 2: Ingest documents (choose chunking strategy)
python app/ingest.py semantic

# Step 3: Ask questions interactively
python -m app.main
```

---

## 📚 Chunking Strategies Explained

| Strategy | Command | Best For | Chunks Created* | Speed |
|----------|---------|----------|-----------------|-------|
| **Token-based** | `python app/ingest.py token_based` | Precise token control, preventing LLM overflow | 62 | ⚡ Fast |
| **Semantic** | `python app/ingest.py semantic` | General Q&A, natural language | 106 | ⚡ Fast |
| **Agentic** | `python app/ingest.py agentic` | Highest quality, complex documents | 114 | 🐌 Slow† |
| **Recursive** | `python app/ingest.py recursive` | Structured/hierarchical documents | 106 | ⚡ Fast |

*Based on 57 pages of sample documents  
†Uses LLM for intelligent chunking (~$0.01 per 10 pages)

### Try Different Strategies
```bash
# Token-based: Best for strict context control
python app/ingest.py token_based
python -m app.main

# Semantic: Recommended default
python app/ingest.py semantic
python -m app.main

# Agentic: Best quality (uses API credits)
python app/ingest.py agentic
python -m app.main

# Recursive: Best for markdown/code
python app/ingest.py recursive
python -m app.main
```

---

## 🎓 Learning Objectives

After completing this project, you will understand:

### Core Concepts
- ✅ **RAG Architecture**: How retrieval improves LLM accuracy
- ✅ **Embeddings**: Semantic search with vector databases
- ✅ **LangGraph**: State machines for LLM workflows
- ✅ **Chunking Strategies**: Document splitting optimization
- ✅ **Grounding**: Preventing hallucination

### Technical Skills
- ✅ LangChain framework
- ✅ ChromaDB vector database
- ✅ OpenAI API integration
- ✅ Python async/await patterns
- ✅ Production-ready error handling

---

## 📁 Project Structure
```
rag_fitness_agent/
├── app/
│   ├── __init__.py
│   ├── ingest.py              # Document ingestion pipeline
│   ├── agent.py               # RAG agent logic
│   ├── graph.py               # LangGraph state definition
│   ├── main.py                # Interactive CLI
│   └── chunking_strategies.py # 4 chunking implementations
├── data/
│   └── pdfs/                  # 📥 Put your PDF files here
├── vectorstore/               # ChromaDB storage (auto-generated)
├── config.yaml                # Configuration settings
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

---

## 🧪 Example Usage

### Interactive Mode
```bash
python -m app.main
```
```
❓ Your question: How much protein in cottage cheese?

💡 Answer:
1 cup of 1% milkfat cottage cheese contains 28 grams of protein.

📚 Sources:
📄 Nutrition_Guide.pdf (pages: 1, 4, 6)
```

### Testing Grounding
```bash
❓ Your question: What is quantum physics?

💡 Answer:
I don't have information about that in the provided documents.
```

✅ **The system correctly refuses to answer questions outside its knowledge base!**

---

## ⚙️ Configuration

Edit `config.yaml` to customize:
```yaml
# Document chunking
chunking:
  chunk_size: 1000      # Characters per chunk
  chunk_overlap: 200    # Overlap between chunks

# Vector database
vectordb:
  collection_name: "fitness_docs"
  persist_directory: "./vectorstore"

# Retrieval
retrieval:
  top_k: 3              # Number of chunks to retrieve

# LLM settings
llm:
  model: "gpt-3.5-turbo"
  temperature: 0.0      # 0 = deterministic, 1 = creative
```

---

## 🔬 Advanced Features

### 1. Compare Chunking Strategies
```bash
python compare_all_chunking.py
```

Shows side-by-side comparison of all 4 strategies.

### 2. Inspect Vector Database
```bash
python inspect_chromadb.py
```

View:
- Total vectors stored
- Sample documents
- Source distribution
- Collection statistics

### 3. Test Similarity Scores
```bash
python check_similarity.py
```

See which chunks are most similar to your query.

---

## 🎯 Use Cases

Adapt this RAG system for:

| Domain | Example Documents | Sample Questions |
|--------|-------------------|------------------|
| **Education** | Textbooks, lecture notes | "Explain photosynthesis" |
| **Healthcare** | Medical guidelines | "What are diabetes symptoms?" |
| **Legal** | Contracts, regulations | "What are tenant rights?" |
| **Corporate** | Policies, reports | "What's our vacation policy?" |
| **Research** | Papers, articles | "Summarize recent AI trends" |

---

## 🛠️ Troubleshooting

### Issue: "No module named 'app'"
**Solution:**
```bash
# Use module mode
python -m app.main
```

### Issue: "OPENAI_API_KEY not found"
**Solution:**
1. Copy `.env.example` to `.env`
2. Add your API key to `.env`

### Issue: "No PDF files found"
**Solution:**
Add PDFs to `data/pdfs/` folder

### Issue: PDF has no text (scanned image)
**Solution:**
- Use OCR-enabled PDFs
- Or add OCR support (see advanced tutorials)

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Ingestion Time** | ~30 seconds (50 pages) |
| **Query Time** | ~2-3 seconds |
| **Embedding Model Size** | 80MB (local) |
| **Cost per Query** | ~$0.001 (OpenAI API) |
| **Accuracy** | 100% grounded (no hallucination) |

---

## 🚀 Next Steps

### For Beginners
1. Complete [STUDENT_GUIDE.md](STUDENT_GUIDE.md) assignments
2. Test with your own documents
3. Try all 4 chunking strategies

### For Intermediate Users
1. Switch to [Ollama](https://ollama.ai) (100% local, no API costs)
2. Add web interface (Streamlit/FastAPI)
3. Implement query history

### For Advanced Users
1. Deploy to Azure/AWS
2. Add multi-modal support (images + text)
3. Build multi-agent workflows
4. Create evaluation framework

---

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Tutorial](https://langchain-ai.github.io/langgraph/)
- [ChromaDB Guide](https://docs.trychroma.com/)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [My YouTube Channel](https://youtube.com/@cognithicai) - RAG tutorials

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👨‍💼 About the Author

**Nisarg Kadam**  
Lead Agentic AI Architect | AI Trainer & Educator

- 🏆 4x Global Hackathon Champion
- 🎓 AI Builder Bootcamp Instructor
- 🎤 International AI Keynote Speaker
- 💼 Enterprise AI Architect (UBS, GSDC)

### Connect With Me
- 🔗 [LinkedIn](https://linkedin.com/in/nisargkadam)
- 🎥 [YouTube - Cognithic AI Labs](https://youtube.com/@cognithicai)
- 🌐 [GitHub](https://github.com/NisargKadam)

---

## 🙏 Acknowledgments

- Built for **AI Builder Bootcamp** students
- Powered by [LangChain](https://langchain.com/) and [OpenAI](https://openai.com/)
- Inspired by real-world enterprise RAG implementations

---

## 📞 Support

- 📧 Issues: [GitHub Issues](https://github.com/NisargKadam/rag-fitness-agent/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/NisargKadam/rag-fitness-agent/discussions)
- 📺 Video tutorials: [YouTube Playlist](https://youtube.com/@cognithicai)

---

## ⭐ Star This Repo

If this project helped you learn RAG systems, please ⭐ star the repository!

---

**Happy Learning! 🚀**

*Built with ❤️ for the AI community*
# RAG_evals_assigment
# RAG_evals_assigment
# RAG_evals_assigment
# RAG_evals_assigment
