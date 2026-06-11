# RAG Pipeline Stages

**Summary**: The five stages from user query to final answer, each with distinct failure modes that advanced RAG techniques address.

**Sources**: From Basic to Advanced RAG every step of the way.md

**Last updated**: 2026-06-10

---

## The five stages

```
query → query embedding → similarity search → retrieval → context → answering
```

(source: From Basic to Advanced RAG every step of the way.md)

## Failure points per stage

| Stage | Failure mode | Advanced fix |
|-------|-------------|--------------|
| **Query** | Poorly framed, ambiguous, or multi-step analytical questions | [[query-enhancement|Query transformation]], decomposition |
| **Query embedding** | General-purpose embedder doesn't capture query intent | HyDE, DPR, embedding fine-tuning — see [[retrieval-techniques]] |
| **Similarity search** | Misses relevant info or source lacks the answer | ColBERT, BM25 hybrid, fusion retrievers — see [[retrieval-techniques]] |
| **Retrieval** | Irrelevant chunks or missing context | Recursive retrieval, reranking — see [[generation-and-postprocessing]] |
| **Context / answering** | LLM ignores context, hallucinates, or mishandles conflicts | Chain of Note, compression, grounding prompts — see [[generation-and-postprocessing]] |

## Design principle

Don't settle for basic RAG. Interject improvements at **every** stage arrow in the pipeline. (source: From Basic to Advanced RAG every step of the way.md)

## Related pages

- [[basic-rag]]
- [[query-enhancement]]
- [[retrieval-techniques]]
- [[generation-and-postprocessing]]
