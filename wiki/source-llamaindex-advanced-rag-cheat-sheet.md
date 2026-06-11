# LlamaIndex Advanced RAG Cheat Sheet

**Summary**: Comprehensive cheat sheet from LlamaIndex covering basic RAG, advanced retrieval/generation techniques, synergy methods, and evaluation — with LlamaIndex code recipes for each.

**Sources**: Advanced RAG Cheat Sheet And Recipes Guide.md

**Last updated**: 2026-06-10

---

Inspired by Gao et al.'s 2023 RAG survey paper. Organizes advanced RAG around two [[advanced-rag-overview|success requirements]]: accurate retrieval and effective generation.

## Basic RAG recipe

Standard flow: load documents → build `VectorStoreIndex` (chunk + embed) → `as_query_engine()` → query. See [[basic-rag]].

## Retrieval techniques covered

- [[chunking-and-indexing|Chunk-size optimization]] via `ParamTuner` grid search
- [[retrieval-techniques|Structured external knowledge]]: recursive retrieval (small chunks retrieve, large chunks generate)
- Knowledge graphs, auto/mixed retrievers, fusion retrievers, embedding fine-tuning, HyDE

## Generation techniques covered

- [[generation-and-postprocessing|Information compression]]: `LongLLMLinguaPostprocessor`
- [[generation-and-postprocessing|Result re-ranking]]: `CohereRerank` to combat "lost in the middle"

## Synergy techniques covered

- [[retrieval-generation-synergy|Generator-enhanced retrieval]]: `FLAREInstructQueryEngine`
- [[retrieval-generation-synergy|Iterative retrieval-generation]]: `RetryQueryEngine` with `RelevancyEvaluator`

## Evaluation

References 7 measurement aspects from the survey. LlamaIndex provides evaluation abstractions and RAGAs integrations. See [[rag-evaluation]].

## Related pages

- [[basic-rag]]
- [[advanced-rag-overview]]
- [[retrieval-techniques]]
- [[generation-and-postprocessing]]
- [[retrieval-generation-synergy]]
- [[rag-evaluation]]
