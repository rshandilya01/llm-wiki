# Advanced RAG Overview

**Summary**: Advanced RAG applies sophisticated techniques to meet two success requirements — accurate retrieval and effective generation — organized by which component they target.

**Sources**: Advanced RAG Cheat Sheet And Recipes Guide.md

**Last updated**: 2026-06-10

---

## Success requirements

A RAG system succeeds when:

1. **Retrieval** finds the most relevant documents for the user query
2. **Generation** makes good use of retrieved documents to answer the query

(source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## Technique categories

| Category | Goal | Examples |
|----------|------|----------|
| Retrieval-only | Better document finding | [[chunking-and-indexing]], [[retrieval-techniques]] |
| Generation-only | Better use of retrieved docs | [[generation-and-postprocessing]] |
| Synergy | Both simultaneously | [[retrieval-generation-synergy]] |

## Mental model

Anchor all advanced RAG decisions against the two success requirements. Ask: does this technique improve retrieval accuracy, generation quality, or both?

Inspired by Gao, Yunfan, et al. "Retrieval-Augmented Generation for Large Language Models: A Survey" (2023). (source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## Related pages

- [[basic-rag]]
- [[rag-pipeline-stages]]
- [[retrieval-techniques]]
- [[generation-and-postprocessing]]
- [[retrieval-generation-synergy]]
- [[rag-evaluation]]
