# RAG Evaluation

**Summary**: How to measure whether a RAG system meets retrieval and generation success requirements.

**Sources**: Advanced RAG Cheat Sheet And Recipes Guide.md

**Last updated**: 2026-06-10

---

## Why evaluate

RAG has two high-level success requirements (see [[advanced-rag-overview]]). Evaluation quantifies how well each is met.

## Seven measurement aspects

Gao, Yunfan, et al. (2023) survey identifies **7 measurement aspects** for RAG evaluation (referenced in LlamaIndex cheat sheet). *Specific aspect names not enumerated in source — needs verification from original survey paper.*

(source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## LlamaIndex tooling

- `SemanticSimilarityEvaluator` — compare generated vs reference responses
- `BatchEvalRunner` — batch evaluation across question sets
- `RelevancyEvaluator` — critique retrieval-generation cycles (used in `RetryQueryEngine`)
- RAGAs integrations

(source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## Hyperparameter tuning as evaluation

`ParamTuner` uses evaluation scores as objective function for chunk size / top-k optimization. Treats RAG tuning like traditional ML grid search. (source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## Related pages

- [[advanced-rag-overview]]
- [[retrieval-generation-synergy]]
- [[chunking-and-indexing]]
