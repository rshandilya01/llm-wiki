# Generation and Post-processing

**Summary**: Techniques to improve how retrieved documents are used at generation time — compression, reranking, fusion, and note-taking.

**Sources**: Advanced RAG Cheat Sheet And Recipes Guide.md, From Basic to Advanced RAG every step of the way.md, Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron.md

**Last updated**: 2026-06-10

---

## Information compression

Retrieved docs may carry noise. LLMs also degrade with too much irrelevant context.

`LongLLMLinguaPostprocessor` compresses retrieved nodes before generation — targets token budget, reorders context. (source: Advanced RAG Cheat Sheet And Recipes Guide.md)

## Result reranking

**"Lost in the middle"** — LLMs focus on prompt extremes. Re-rank retrieved docs so most relevant appear at edges.

- `CohereRerank` postprocessor (retrieve top-10, rerank to top-2)
- NVIDIA NeMo `llama-3.2-nv-rerankqa-1b-v2` via `NVIDIARerank`
- Wrapped in `ContextualCompressionRetriever`

(source: Advanced RAG Cheat Sheet And Recipes Guide.md, Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron.md)

## Reciprocal Rank Fusion (RRF)

When multiple retrieval methods produce separate top-k sets (e.g. BM25 + dense), fuse rankings into single top-k. Alternative: custom heuristics with similarity cutoffs. (source: From Basic to Advanced RAG every step of the way.md)

## Chain of Note (CoN)

LLM writes **notes** on each retrieved passage before answering. Helps with:

- Reducing hallucinations
- Conflicting or partial information in documents
- Combining retrieval context with LLM knowledge

Similar to chain-of-thought prompting applied per retrieval. (source: From Basic to Advanced RAG every step of the way.md)

## Corrective RAG (CRAG)

Extension of CoN — when retrievals are ambiguous or insufficient, search the internet to fill gaps. (source: From Basic to Advanced RAG every step of the way.md)

## Related pages

- [[advanced-rag-overview]]
- [[retrieval-techniques]]
- [[agentic-rag-and-react]]
