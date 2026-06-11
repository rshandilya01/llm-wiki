# Retrieval-Generation Synergy

**Summary**: Techniques using the interplay between retrieval and generation to improve both document finding and answer quality.

**Sources**: Advanced RAG Cheat Sheet And Recipes Guide.md

**Last updated**: 2026-06-10

---

## Generator-enhanced retrieval (FLARE)

`FLAREInstructQueryEngine` — the generator LLM plays an active role **before** retrieval:

- LLM reasons about what information it needs
- Elicits retrieval instructions
- Then performs retrieval with refined intent

Useful when raw user query poorly indicates what documents are needed. (source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## Iterative retrieval-generation (Retry)

`RetryQueryEngine` — runs retrieval-generation cycles in a loop:

1. Retrieve + generate
2. `RelevancyEvaluator` critiques the result
3. Repeat until passing evaluation or max cycles reached

Handles complex queries needing multi-step reasoning. (source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## Distinction from single-component techniques

| Retrieval-only | Generation-only | Synergy |
|---------------|-----------------|---------|
| Chunk optimization | Compression | FLARE |
| Recursive retrieval | Reranking | RetryQueryEngine |
| ColBERT | Chain of Note | |

See [[advanced-rag-overview]] for the full categorization.

## Related pages

- [[query-enhancement]]
- [[advanced-rag-overview]]
- [[rag-evaluation]]
