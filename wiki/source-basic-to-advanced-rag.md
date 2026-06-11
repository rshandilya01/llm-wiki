# From Basic to Advanced RAG

**Summary**: Rahul Deora's walkthrough of the basic RAG pipeline's failure points and the advanced techniques added at each stage — query transformation, routing, retrieval, and post-processing.

**Sources**: From Basic to Advanced RAG every step of the way.md

**Last updated**: 2026-06-10

---

Published 2024-02-25. Frames RAG as a 5-stage pipeline where each stage can fail independently. Advanced RAG = interjecting improvements at every arrow in the pipeline diagram.

## Pipeline framing

`query → query embedding → similarity search → retrieval → context → answering`

Each stage has distinct failure modes documented in [[rag-pipeline-stages]].

## Techniques covered

| Stage | Techniques |
|-------|------------|
| Query | [[query-enhancement|Query transformation]], multi-step query decomposition |
| Routing | [[query-enhancement|Query routing]] across multiple data sources or tools |
| Retrieval | [[retrieval-techniques|Dense Passage Retrieval]], [[retrieval-techniques|ColBERT]] late interaction |
| Post-processing | [[generation-and-postprocessing|Reciprocal Rank Fusion]], [[generation-and-postprocessing|Chain of Note]] |
| Augmentation | Corrective RAG (CRAG) — internet search for missing content |

## Key insight

Analytical queries (e.g. comparing Berlin temperatures across years) may have no matching sentence in documents — direct vector search fails. Decomposition into sub-queries is essential. (source: From Basic to Advanced RAG every step of the way.md)

## Related pages

- [[rag-pipeline-stages]]
- [[query-enhancement]]
- [[retrieval-techniques]]
- [[generation-and-postprocessing]]
