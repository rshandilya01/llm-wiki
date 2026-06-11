# Agentic RAG and ReAct

**Summary**: Agentic RAG gives the LLM control over when and how to retrieve, using ReAct (reasoning + acting) agents with tool calling — implemented via LangGraph and NVIDIA Nemotron in production workshops.

**Sources**: Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron.md, Advanced RAG Cheat Sheet And Recipes Guide.md

**Last updated**: 2026-06-10

---

## Basic RAG vs agentic RAG

| Basic RAG | Agentic RAG |
|-----------|-------------|
| Fixed retrieval every query | Agent decides **when** to retrieve |
| No data source selection | Dynamic tool/source selection |
| No quality control | Agent can skip bad retrievals |
| Single-shot pipeline | Iterative reasoning + tool calls |

(source: Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron.md)

## ReAct architecture

**Reasoning + Acting** via tool-calling LLMs:

1. LLM receives prompt
2. If tool call needed → run tool, append result to chat history
3. Send back to LLM for next reasoning step
4. Repeat until final response

For RAG: expose retrieval chain as a **tool** the agent invokes on demand. (source: Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron.md)

## LangGraph implementation

```python
from langgraph.prebuilt import create_react_agent

AGENT = create_react_agent(
    model=llm,
    tools=[RETRIEVER_TOOL],
    prompt=SYSTEM_PROMPT,
)
```

Agent graph = dynamic workflow (vs linear "chain"). Run with `langgraph dev`. (source: Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron.md)

## Grounding prompt principles

Reliable agentic RAG prompts specify:

- **Role** — agent expertise and scope
- **Tool utilization** — which tool for which task
- **Grounding** — answer from sources, admit uncertainty
- **Citation** — inline source tags (e.g. `[KB]`)
- **Style** — brief, conversational

(source: Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron.md)

## NVIDIA stack

| Layer | Component |
|-------|-----------|
| LLM | Nemotron Nano 9B V2 via NIM |
| Embeddings | NeMo Retriever EmbedQA 1B V2 |
| Reranker | NeMo Retriever RerankQA 1B V2 |
| Vector DB | FAISS (dev) → production vector DB TBD |
| Tracing | LangSmith (optional) |
| Deployment | NVIDIA Brev Launchable → local NIM Docker containers |

Production: migrate from API Catalog to local NIM microservices for unlimited performance. (source: Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron.md)

## Related pages

- [[basic-rag]]
- [[chunking-and-indexing]]
- [[generation-and-postprocessing]]
- [[source-nvidia-nemotron-rag-agent]]
- [[query-enhancement]]
