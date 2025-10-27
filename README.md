# Study Assistant

A terminal-based AI assistant that answers questions about your study materials using RAG (Retrieval-Augmented Generation).

## Features

- **Index documents** - PDF, DOCX, PPTX, TXT, MD
- **Semantic search** - ChromaDB vector store with intelligent retrieval
- **Multi-provider LLM** - OpenAI (GPT-4o, GPT-3.5) or Google Gemini
- **Conversation history** - Multi-turn context for natural follow-up questions
- **Source citations** - Answers include page numbers and document references
- **Zero-config setup** - Auto-installs dependencies, interactive wizard
- **Error resilience** - Automatic retry with exponential backoff

## Quick Start

1. **Clone repository:**
   ```bash
   git clone https://github.com/JoshuaDelosSantos/study-assistant.git
   cd study-assistant
   ```

2. **Run (auto-installs dependencies):**
   ```bash
   python3 main.py
   ```

3. **Follow setup wizard:**
   - Choose LLM provider (OpenAI or Gemini)
   - Enter API key
   - Select directories to index

4. **Ask questions!**
   ```
   [1]> What are the key concepts in machine learning?
   [2]> Explain supervised vs unsupervised learning
   [3]> How does that relate to neural networks?
   ```

## Requirements

- Python 3.8+
- API key for OpenAI or Google Gemini
- Internet connection (for LLM API calls)

## Commands

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `status` | Display configuration and system status |
| `history` | View conversation history |
| `clear` | Clear conversation history |
| `reindex` | Rebuild document index |
| `reconfig` | Reconfigure system (delete config and restart setup) |
| `q`, `exit` | Quit application |

## Configuration

Configuration is stored in `config.yaml`. Advanced users can edit directly:

```yaml
llm_provider: openai          # or 'gemini'
llm_model: gpt-4o             # or 'gemini-2.5-flash', 'gpt-3.5-turbo'
llm_api_key: sk-xxx...        # Your API key
directories:
  - /path/to/study/materials
chunk_size: 1000              # Characters per chunk
chunk_overlap: 200            # Overlap between chunks
top_k: 5                      # Number of sources to retrieve
track_tokens: true            # Show token usage
show_sources: true            # Show source citations
verbose: false                # Enable debug output
```

## Examples

### Multi-turn Conversation
```
[1]> What is gradient descent?
Answer: Gradient descent is an optimization algorithm used to minimize 
the cost function in machine learning models...

Sources: machine_learning.pdf (Page 42, Chunk 0) - similarity: 0.95

[2]> How does it relate to backpropagation?
Answer: Backpropagation uses gradient descent to update neural network 
weights by computing gradients layer by layer...

Sources: neural_networks.pdf (Page 15, Chunk 2) - similarity: 0.92
```

### Reindexing After Adding Documents
```
> reindex
Reindexing Warning
This will delete all existing indexed documents and rebuild from scratch.

Are you sure? [y/n]: y

Clearing existing index...
Recreating collection...
Indexing documents...

Indexed 50 files, 234 chunks in 12.3s

Reindexing complete!
```

### Viewing Conversation History
```
> history

Conversation History (3 turns):

[1] Q: What is gradient descent?
    A: Gradient descent is an optimization algorithm...
    (1 sources, 120 tokens)

[2] Q: How does it relate to backpropagation?
    A: Backpropagation uses gradient descent to update...
    (2 sources, 95 tokens)

[3] Q: Can you give an example?
    A: Here's a simple example of gradient descent...
    (1 sources, 150 tokens)
```

## Troubleshooting

### "API key invalid"
- **Verify your API key is correct**
  - OpenAI keys start with `sk-`
  - Get OpenAI key: https://platform.openai.com/api-keys
  - Get Gemini key: https://makersuite.google.com/app/apikey
- **Update configuration:** Run `reconfig` command in the app or delete `config.yaml` and restart

### "Rate limit exceeded"
- **Wait 60 seconds** and try again
- **Reduce query frequency** - Add delays between questions
- **Upgrade API plan** for higher rate limits
- The app will automatically retry with exponential backoff

### "No relevant context found"
- **Check documents are indexed:** Run `status` command
- **Verify directories:** Check `config.yaml` has correct paths
- **Try rephrasing:** Ask the question differently
- **Reindex documents:** Run `reindex` if you added new files

### "Out of memory"
- **Reduce `chunk_size`** in config (default: 1000)
- **Reduce `top_k`** in config (default: 5)
- **Index fewer documents** - Split into smaller batches

### "Network error"
- **Check internet connection**
- **Verify firewall settings** - Allow Python to access internet
- **Try again** - Temporary network issues resolve automatically

## Cost Estimation

API costs vary by provider and model:

| Provider | Model | Cost per Query* | Notes |
|----------|-------|----------------|-------|
| OpenAI | GPT-4o | ~$0.005 | Recommended for quality |
| OpenAI | GPT-3.5-turbo | ~$0.001 | Budget-friendly |
| Google | Gemini 2.5 Flash | ~$0.001** | Best price-performance, stable |
| Google | Gemini 2.5 Flash-Lite | ~$0.0005** | Ultra fast, most cost-efficient |

\* Typical query with 5 sources, 500 token context  
\** After free tier limit

**Cost Monitoring:**
- Enable `track_tokens: true` in config to see usage
- Token counts displayed after each query
- Example: `Tokens: 450 prompt + 120 completion = 570 total`

**Typical Usage:**
- Study session (20 queries): $0.10 - $0.20 with GPT-4o
- Heavy use (100 queries/day): $0.50 - $1.00 per day

## Project Structure

```
study-assistant/
├── main.py                  # Self-bootstrapping entry point
├── config.yaml              # User configuration (auto-generated)
├── data/                    # ChromaDB storage
├── src/
│   ├── agent.py             # RAG agent (query processing)
│   ├── cli.py               # Terminal interface
│   ├── config.py            # Configuration management
│   ├── conversation.py      # Conversation history
│   ├── indexer.py           # Document processing
│   ├── llm_providers.py     # LLM integrations (OpenAI, Gemini)
│   ├── setup_wizard.py      # Interactive setup
│   └── vector_store.py      # ChromaDB wrapper
├── tests/                   # Test suite (220+ tests)
└── agent/                   # Development documentation
```

## Technical Details

### How It Works

1. **Indexing Phase:**
   - Extracts text from documents (PDF, DOCX, PPTX, TXT, MD)
   - Splits into chunks (default: 1000 chars with 200 char overlap)
   - Generates embeddings using sentence-transformers
   - Stores in ChromaDB vector database

2. **Query Phase:**
   - User asks a question
   - System retrieves top-k most relevant chunks (semantic search)
   - Includes conversation history (last 3 turns) for context
   - Builds prompt with retrieved context + history
   - Sends to LLM (OpenAI or Gemini) for generation
   - Returns answer with source citations

3. **Conversation Management:**
   - Maintains rolling window of last 10 turns
   - Each turn includes: query, answer, sources, tokens
   - Context automatically included in subsequent queries
   - Use `clear` command to reset conversation

### Error Handling

The app includes **automatic retry logic** with exponential backoff:

- **Retryable errors:** Rate limits (429), server errors (500/502/503), timeouts
- **Non-retryable errors:** Authentication (401), bad requests (400)
- **Retry strategy:** 3 attempts with 1s, 2s delays
- **User-friendly messages:** Categorised errors with recovery suggestions

## Known Limitations

- **Text-only:** No image or diagram extraction from PDFs
- **Internet required:** API calls need active connection
- **Manual reindex:** No real-time file watching (run `reindex` after adding documents)
- **Single-user:** No collaboration or multi-user features
- **Session-based history:** Conversation resets when app closes

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_agent.py -v
```

### Project Status

- Phase 1: Foundation (self-bootstrapping, config, setup) - Complete
- Phase 2: Vector Store (ChromaDB, indexing, chunking) - Complete
- Phase 3: RAG Agent (LLM providers, query processing) - Complete
- Phase 4: Production Polish (conversations, error handling, docs) - Complete
- 220+ tests passing, 62% coverage

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Python 3.8+ with type hints
- Follow PEP 8 style guide
- Add docstrings to functions/classes
- Write tests for new features
- Keep functions focused and readable

## Support

- **Issues:** Report bugs or request features via [GitHub Issues](https://github.com/JoshuaDelosSantos/study-assistant/issues)
- **Questions:** Open a discussion or check existing issues
- **Documentation:** See `agent/` directory for development docs

## Acknowledgements

- Built with [ChromaDB](https://www.trychroma.com/) for vector storage
- Uses [OpenAI](https://openai.com/) and [Google Gemini](https://ai.google.dev/) for LLM
- Powered by [sentence-transformers](https://www.sbert.net/) for embeddings
- UI built with [Rich](https://rich.readthedocs.io/)

---

**Made for students who want AI-powered study assistance without the hassle.**
