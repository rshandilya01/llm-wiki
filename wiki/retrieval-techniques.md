# Retrieval Techniques

**Summary**: Advanced methods for finding the most relevant documents — beyond basic vector similarity search.

**Sources**: Advanced RAG Cheat Sheet And Recipes Guide.md, From Basic to Advanced RAG every step of the way.md

**Last updated**: 2026-06-10

---

## Dense Passage Retrieval (DPR)

Uses **separate encoders** for queries and documents, trained with Siamese objective on QA pairs. General-purpose shared embedders may not represent query-document relevance well. (source: From Basic to Advanced RAG every step of the way.md)

HyDE (hypothetical document embeddings) attempts to work around this but suffers from hallucinations. (source: From Basic to Advanced RAG every step of the way.md)

## ColBERT (late interaction)

Instead of single-vector compression:

1. Tokenize query and document separately through encoders → word embedding sets
2. Build pairwise similarity matrix between query tokens and document tokens
3. **MaxSim**: max similarity per query token row, sum for final score

More fine-grained than cosine on whole-document embeddings. Handles queries like "Gross Revenue of all products of Company X in 2005" where dense embeddings are noisy. >2x retrieval performance vs BM25. (source: From Basic to Advanced RAG every step of the way.md)

## Recursive retrieval

Retrieve with **small chunks** for precision, pass associated **larger parent chunks** to generation for context. Uses `IndexNode` linking in LlamaIndex. (source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## Structured external knowledge

Beyond flat vector index:

- Knowledge graphs
- Mixed/auto retrievers
- Fusion retrievers (combine multiple retrieval methods)
- Embedding model fine-tuning

(source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## Hybrid retrieval

Combine dense embeddings with classical methods like **BM25**. Fuse multiple top-k result sets via [[generation-and-postprocessing|Reciprocal Rank Fusion]]. (source: From Basic to Advanced RAG every step of the way.md)

## Related pages

- [[chunking-and-indexing]]
- [[generation-and-postprocessing]]
- [[advanced-rag-overview]]
