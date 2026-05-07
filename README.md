# aiera-langchain-earnings

Basic LangChain scaffold for asking questions over a transcript excerpt from an earnings call, using a local Ollama backend.

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install and start Ollama.
4. Pull the chat model you want to use. The default in this scaffold is:

```bash
ollama pull llama3.1
```

5. Optionally set `OLLAMA_BASE_URL` or `OLLAMA_MODEL` in your environment or a local `.env` file.
6. No Ollama API key is required for a local default installation.

## Run

Retrieval-based Q&A:

```bash
python app.py --mode retrieval --transcript examples/transcript_excerpt.txt --question "What drove gross margin expansion?"
```

Conversation-style Q&A:

```bash
python app.py --mode conversation --transcript examples/transcript_excerpt.txt --question "What did management say about operating expenses?"
```

If you do not pass `--transcript`, the app uses a built-in sample excerpt.

The retrieval path uses a local TF-IDF retriever, so no embedding model needs to be pulled for this scaffold.
