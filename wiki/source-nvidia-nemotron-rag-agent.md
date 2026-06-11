# NVIDIA Nemotron RAG Agent Workshop

**Summary**: Hands-on NVIDIA DevX workshop for building an agentic RAG IT help desk agent using Nemotron Nano 9B, LangGraph ReAct, FAISS, and NeMo Retriever embedding/reranking models.

**Sources**: Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron.md

**Last updated**: 2026-06-10

---

Published 2025-09-23 by Edward Li. Deployable via NVIDIA Brev Launchable.

## What you build

An [[agentic-rag-and-react|agentic RAG]] system where a ReAct agent decides **when** to retrieve vs respond directly, using a knowledge-base retriever tool.

## Stack

| Component | Model / Tool |
|-----------|-------------|
| LLM | `nvidia/nvidia-nemotron-nano-9b-v2` via ChatNVIDIA |
| Embeddings | NeMo Retriever `llama-3.2-nv-embedqa-1b-v2` |
| Reranker | NeMo Retriever `llama-3.2-nv-rerankqa-1b-v2` |
| Vector DB | In-memory FAISS |
| Agent framework | LangGraph `create_react_agent` |
| Chunking | `RecursiveCharacterTextSplitter` (800 tokens, 120 overlap) |

## Pipeline steps implemented

1. Load markdown docs from `./data/it-knowledge-base`
2. [[chunking-and-indexing|Split and embed]] chunks into FAISS
3. Build retriever (top-6 similarity) + [[generation-and-postprocessing|reranker]] via `ContextualCompressionRetriever`
4. Wrap as `create_retriever_tool` for the agent
5. Configure ReAct agent with grounding-focused system prompt

## Production path

Workshop uses NVIDIA API Catalog; production migrates to local [[agentic-rag-and-react|NIM microservices]] via Docker. LangSmith optional for tracing.

## Related pages

- [[agentic-rag-and-react]]
- [[basic-rag]]
- [[chunking-and-indexing]]
- [[generation-and-postprocessing]]
