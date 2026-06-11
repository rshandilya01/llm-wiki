# Basic RAG

**Summary**: Standard Retrieval-Augmented Generation architecture — retrieve documents from external knowledge, pass them with the user query to an LLM for response generation.

**Sources**: Advanced RAG Cheat Sheet And Recipes Guide.md, From Basic to Advanced RAG every step of the way.md, Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron.md

**Last updated**: 2026-06-10

---

## Three components

1. **Retrieval** — find relevant documents from external knowledge
2. **External knowledge database** — indexed document store (vector DB, etc.)
3. **Generation** — LLM produces answer using retrieved context + query

(source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## Basic flow

```
User query → Embed query → Similarity search → Retrieve top-k chunks → LLM generates answer
```

Expanded 5-stage view in [[rag-pipeline-stages]].

## Minimal LlamaIndex implementation

```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex

documents = SimpleDirectoryReader(input_dir="...").load_data()
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine()
response = query_engine.query("A user's query")
```

(source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## Why RAG exists

LLMs limited by training data cutoff. RAG injects up-to-date, domain-specific, or confidential information at inference time. (source: Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron.md)

## Limitations of basic RAG

- Fixed retrieval pipeline — LLM cannot control how/when data is retrieved
- No data source selection
- No quality control over retrieved content
- Each [[rag-pipeline-stages|pipeline stage]] is a failure point

Leads to [[advanced-rag-overview]] and [[agentic-rag-and-react]].

## Related pages

- [[rag-pipeline-stages]]
- [[advanced-rag-overview]]
- [[agentic-rag-and-react]]
