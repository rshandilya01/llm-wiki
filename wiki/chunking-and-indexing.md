# Chunking and Indexing

**Summary**: How documents are split, embedded, and stored for retrieval — chunk size and overlap directly affect RAG quality.

**Sources**: Advanced RAG Cheat Sheet And Recipes Guide.md, Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron.md

**Last updated**: 2026-06-10

---

## Why chunking matters

LLMs have context length limits. Documents must be chunked for the external knowledge database. Chunks too large or too small cause inaccurate generation responses. (source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## Chunk size optimization

Grid-search hyperparameters using LlamaIndex `ParamTuner`:

- Define objective function scoring semantic similarity of RAG outputs vs reference answers
- Search over `chunk_size` values (e.g. 256, 512, 1024)
- Pick best-scoring configuration

(source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## Chunk overlap

Number of tokens shared between consecutive chunks. Preserves continuity and context across chunk boundaries. (source: Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron.md)

## NVIDIA workshop defaults

- `RecursiveCharacterTextSplitter`
- Chunk size: **800** tokens
- Chunk overlap: **120** tokens
- Embed with NeMo Retriever `llama-3.2-nv-embedqa-1b-v2`
- Store in FAISS via `FAISS.from_documents()`

(source: Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron.md)

## Recursive chunking pattern

Parent chunks (1024) + child sub-chunks (256, 512) linked via `IndexNode` for [[retrieval-techniques|recursive retrieval]]. (source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## Related pages

- [[retrieval-techniques]]
- [[basic-rag]]
- [[agentic-rag-and-react]]
