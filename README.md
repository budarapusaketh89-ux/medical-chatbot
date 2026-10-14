# Medical Chatbot — Medical Report Q&A

A medical document question-answering application that retrieves relevant information from uploaded PDF documents using NLP, transformer-based models, and semantic search.

## Overview

The Medical Chatbot helps users ask questions about medical documents and find relevant passages from their content. It combines document processing, semantic retrieval, and a web-based interface to make medical information easier to explore.

## Features

- **PDF-based question answering:** Retrieve relevant passages from uploaded medical documents.
- **Semantic search:** Find contextually relevant information rather than relying only on exact keyword matches.
- **Transformer-based NLP:** Uses BioBERT and MiniLM-related components for language understanding and retrieval workflows.
- **FAISS indexing:** Supports efficient similarity search over the prepared knowledge base.
- **Web interface:** Includes pages for interacting with the chatbot, logging in, and registering.
- **Backend API:** Python backend for handling application functionality and document-related workflows.

## Tech Stack

| Component | Technologies |
|---|---|
| Backend | Python, FastAPI (verify against the current backend implementation) |
| Frontend | HTML, CSS, JavaScript |
| NLP | BioBERT, MiniLM |
| Semantic Search | FAISS |
| Data Processing | NumPy, Pickle |
| Model Artifacts | Hugging Face-compatible model configuration and tokenizer files |

## Project Structure

```text
medical-chatbot/
├── backend/
│   ├── main.py
│   ├── auth.py
│   ├── database.py
│   ├── db_models.py
│   ├── reports.py
│   ├── schemas.py
│   └── test_*.py
├── data/
│   ├── kb_embeddings.npy
│   ├── kb_metadata.pkl
│   └── medquad_faiss.index
├── frontend/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── app.js
│   ├── auth.js
│   └── styles.css
├── models/
│   ├── config.json
│   ├── intent_classes.json
│   ├── label_encoder.pkl
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── .gitignore
└── README.md
```

## How It Works

1. **Document input:** The application accepts medical PDF documents through its document workflow.
2. **Processing and retrieval:** Document content is processed and matched against relevant knowledge-base information.
3. **Semantic search:** FAISS is used to retrieve relevant passages from the prepared knowledge base.
4. **Question answering:** The retrieved context supports the chatbot's response to the user's question.
5. **Web interaction:** Users interact with the application through its frontend.

*This describes the intended high-level workflow; exact processing steps depend on the implementation in `backend/main.py`.*

## Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by your dependencies
- Git
- A modern web browser

### 1. Clone the Repository

```bash
git clone https://github.com/budarapusaketh89-ux/medical-chatbot.git
cd medical-chatbot
```

### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install the Python packages required by the backend. A `requirements.txt` file should list the dependencies used by the current implementation.

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present yet, create it from the imports and tested package versions used by your project before running this command.

### 4. Start the Backend

If the backend uses FastAPI and exposes an ASGI application named `app` in `main.py`, run:

```bash
uvicorn main:app --reload
```

Run this command from the `backend` directory. Confirm the application variable and startup instructions in `backend/main.py` first.

### 5. Open the Frontend

Open `frontend/index.html` in a browser, or serve the frontend using a local development server if required by the backend API configuration.

## Repository Artifacts

The repository includes the prepared knowledge-base files and model configuration/tokenizer artifacts. The large model weights file is excluded from normal Git uploads because it exceeds GitHub's standard individual-file size limit.

To reproduce the complete environment, obtain the required model weights separately and place them at the path expected by the backend.

## Limitations

- The chatbot retrieves information from available documents and its configured knowledge base; responses should be checked against the original source.
- This project is for educational and informational purposes and is **not a substitute for professional medical advice, diagnosis, or treatment**.
- Actual setup requirements depend on the model files and dependencies configured in the current implementation.

## Future Improvements

- Add a verified, pinned dependency file and a complete setup guide.
- Improve source citations and passage-level references in answers.
- Add automated tests for document ingestion, retrieval, and question answering.
- Provide a reproducible method for downloading the required model weights.
- Improve deployment and error handling.

## Author

**Saketh Budarapu**

- GitHub: [@budarapusaketh89-ux](https://github.com/budarapusaketh89-ux)
- Project Repository: [medical-chatbot](https://github.com/budarapusaketh89-ux/medical-chatbot)

---

*Built as an NLP and semantic-search project for exploring medical documents.*
