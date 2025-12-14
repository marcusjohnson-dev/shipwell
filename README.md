# Shipwell
Shipwell Assessment

## Objective: 
Build an autonomous AI agent with LangChain that ingests a small folder of mixed-format documents and answers ad-hoc questions about their content.

## Features

- Supports multiple document formats: `.txt`, `.pdf`, `.docx`, `.jpg/.png` (with OCR)
- Semantic search using vector embeddings
- Answers grounded in retrieved context with source citations
- Simple CLI interface

## Setup

### Prerequisites

- Python 3.9+
- OpenAI API key
- Tesseract OCR (for image processing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/marcusjohnson-dev/shipwell.git
cd shipwell
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

Run the agent with your documents folder:
```bash
python app.py --docs ./sample_docs
```

Then ask questions interactively:
```
Your Question: When is CloudSync Pro launching?

Answer: CloudSync Pro is scheduled to launch on February 15, 2025.

Sources:
  [1] tech_launch.txt
      Preview: NEXWAVE TECHNOLOGIES Product Launch Strategy - CloudSync Pro Q1 2025 Launch Plan  Prepared by: Marke...
  [2] tech_launch.txt
      Preview: ═══════════════════════════════════════════════════════════  APPENDIX: BETA TEST RESULTS  Beta Perio...
  [3] tech_launch.txt
      Preview: ═══════════════════════════════════════════════════════════  CONTACT INFORMATION  NexWave Technologi....
```

Type `quit` or `exit` to stop.

```
Other Questions to test with:

    [1] Easy warm-up question
        Q: "When is CloudSync Pro launching?"
        A: February 15, 2025
    [2] Show different document types
        Q: "What supplies does Bella's Brew order?"
        A: Coffee beans from Mountain Peak Roasters, milk from Green Valley Dairy
    [3] OCR demonstration
        Q: "What trail is the hiking club doing next?"
        A: Eagle Peak Trail (from the image!)
    [4] Specific details
        Q: "How much does Apollo the eagle weigh?"
        A: 10.2 lbs
    [5] Cross-document comparison
        Q: "What are the phone numbers in these documents?"
        A: Should list multiple phone numbers from different organizations
    [6] Complex query
        Q: "Who are the managers or leaders mentioned across all documents?"
        A: Should mention Patty (garden), Victoria (hotel), Alex (tech company), etc.
```

## Adding Documents

Simply drop supported files (`.txt`, `.pdf`, `.docx`, `.jpg`, `.png`) into your documents folder and rerun the app.

## Architecture

### Components

1. **Document Loaders** (`src/loaders/`)
   - Handles multiple file formats
   - Extracts text from PDFs, Word docs, and images (OCR)
   - Converts to LangChain Document objects

2. **Vector Store** (`src/indexer/`)
   - Chunks documents into semantic units (1000 chars, 200 overlap)
   - Creates embeddings using OpenAI's `text-embedding-ada-002`
   - Stores in ChromaDB for fast retrieval

3. **QA Agent** (`src/agent/`)
   - Retrieves top 3 relevant chunks for each question
   - Uses GPT-3.5-turbo to generate answers grounded in context
   - Returns source documents for citation

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **ChromaDB** | No external dependencies, persists locally |
| **1000-char chunks** | Balances context vs. precision |
| **Top-3 retrieval** | Enough context without overwhelming the LLM |
| **GPT-3.5-turbo** | Cost-effective, good quality |
| **OpenAI embeddings** | High quality, well-supported (could use free alternatives like `sentence-transformers`) |

### Extensibility

- **Add new file formats**: Extend `DocumentLoader.SUPPORTED_FORMATS` and add loader logic
- **Swap LLM**: Change `ChatOpenAI` in `qa_agent.py` (e.g., to Claude, Llama)
- **Swap vector store**: Replace ChromaDB with FAISS, Pinecone, etc.
- **Add API**: Wrap in FastAPI for REST endpoints
- **Multi-turn conversations**: Add conversation memory to the chain

## Project Structure
```
rag-agent/
├── src/
│   ├── loaders/
│   ├── indexer/
│   └── agent/
├── sample_docs/
├── app.py
├── requirements.txt
└── README.md
```

## Troubleshooting

**"No module named 'langchain'"**
- Make sure virtual environment is activated: `source venv/bin/activate`

**"TesseractNotFoundError"**
- Install Tesseract OCR
**"OPENAI_API_KEY not found"**
- Create `.env` file with your API key

**"No documents found"**
- Check that your documents folder has supported file types
- Verify file extensions are lowercase (`.txt`, not `.TXT`)

**Deprecation warnings**
- These are mostly harmless