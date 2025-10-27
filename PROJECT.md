# Study Assistant Project Proposal

## Overview

A local Python-based study assistant that enables students to query their study materials using natural language. The system indexes documents from specified directories and uses retrieval-augmented generation (RAG) to provide contextual answers from the student's own files.

## Capabilities

A command-line application that:
- Automatically indexes study materials (PDF, DOCX, PPTX, TXT)
- Uses semantic search to find relevant content
- Provides conversational answers with source citations
- Runs entirely locally with student-provided API keys

## Technical Architecture

### Core Components

**Vector Database**: ChromaDB for document embeddings and similarity search
- Lightweight, no external dependencies
- Persistent local storage
- Fast retrieval (10K-50K chunks)

**Language Model Integration**: Multi-provider support (student-configured)
- OpenAI (gpt-3.5-turbo, gpt-4, gpt-4o)
- Google Gemini (gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro, gemini-2.0-flash)
- Configurable provider and model selection
- Falls back to local embeddings for search

**Document Processing Pipeline**:
- PyPDF2 for PDF extraction
- python-docx for Word documents
- python-pptx for PowerPoint slides
- Chunking strategy optimised for academic content

**CLI Interface**: Rich-based terminal UI
- Interactive setup wizard (first-time and quick start modes)
- Menu-driven workflow
- Real-time indexing progress
- Clear error messaging
- API key validation before indexing

### Environment Management

Self-bootstrapping entry point that:
1. Validates Python version (3.8+ required)
2. Creates virtual environment if absent
3. Installs dependencies automatically
4. Relaunches in isolated environment

No manual setup required beyond running `python3 main.py`.

## Project Structure

```
study-assistant/
├── main.py              # Self-bootstrapping entry point
├── requirements.txt     # Python dependencies
├── config.yaml          # User configuration (API keys, directories)
├── .venv/              # Auto-created virtual environment
├── data/               # ChromaDB persistent storage
└── src/
    ├── cli.py          # Terminal interface
    ├── indexer.py      # Document processing
    ├── vector_store.py # ChromaDB wrapper
    └── agent.py        # RAG implementation
```

## User Workflow

1. Clone repository into parent directory of study materials
2. Run `python3 main.py`
3. Choose setup mode:
   - First-time setup: Guided walkthrough of all options
   - Quick start: Minimal prompts (provider, API key, directories only)
4. System validates credentials and indexes documents automatically
5. Ask questions in natural language
6. Receive answers with source references
7. Exit cleanly with 'q' or 'exit'

Configuration stored in `config.yaml` for subsequent runs. Advanced users can edit directly for fine-tuning.

## Design Decisions

**Why ChromaDB over PostgreSQL + pgvector?**
- No container overhead or configuration complexity
- Comparable performance at student-scale workloads
- Zero external dependencies
- Single-command installation

**Why self-bootstrapping over manual setup?**
- Eliminates interpreter version conflicts
- Removes barrier to entry for non-technical users
- Ensures reproducible environments
- Handles dependency installation transparently

**Why CLI over web interface?**
- Lower resource usage
- Simpler architecture
- Terminal-native workflow for developers
- No port conflicts or server management

## Constraints and Limitations

- Requires internet connection for LLM API calls
- API costs borne by student (token tracking recommended)
- Local processing only; no collaborative features
- Limited to text-extractable document formats
- No real-time file watching (manual re-indexing required)

## Success Criteria

- First-time setup completes in under 2 minutes
- Quick start mode requires fewer than 4 user inputs
- API credentials validated before document processing
- Document indexing processes 100 pages per minute
- Query responses return within 3 seconds
- System handles 1,000+ documents without performance degradation
- Zero manual configuration file editing required

## Future Enhancements

- Local LLM support (Ollama integration)
- Incremental indexing for changed files
- Export conversation history
- Multi-language document support
- Citation export for academic writing

## Target Users

Undergraduate and postgraduate students who:
- Manage large volumes of study materials
- Need quick information retrieval during study sessions
- Prefer terminal-based workflows
- Have basic Python familiarity (can run a script)

## Dependencies

- Python 3.8+
- ChromaDB
- LangChain
- OpenAI Python SDK
- Google Generative AI SDK
- Document processing libraries (PyPDF2, python-docx, python-pptx)
- Rich (terminal UI)
