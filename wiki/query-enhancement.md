# Query Enhancement

**Summary**: Techniques to improve user queries before retrieval — transformation for clarity and decomposition/routing for complex multi-source questions.

**Sources**: From Basic to Advanced RAG every step of the way.md, Advanced RAG Cheat Sheet And Recipes Guide.md

**Last updated**: 2026-06-10

---

## Query transformation

When user query is poorly framed, ambiguous, or grammatically incorrect → use LLM to reframe it before retrieval. (source: From Basic to Advanced RAG every step of the way.md)

## Multi-step query decomposition

For analytical or logical questions requiring multiple reasoning steps:

1. Break query into sub-queries
2. Answer each sub-query via basic [[basic-rag|RAG flow]]
3. Feed sub-answers + original question to LLM for final synthesis

Sub-queries can be generated sequentially or in parallel.

**Example**: "Is Berlin's average summer temperature higher than in 2005?"
- Sub-query 1: Current average temperature in Berlin
- Sub-query 2: Average temperature in Berlin in 2005
- Synthesize comparison

No single document sentence may state the comparison directly — decomposition is required. (source: From Basic to Advanced RAG every step of the way.md)

## Query routing

When multiple data sources exist (or documents are grouped):

1. Each source gets a human-written description (e.g. "All data about sports teams in Houston")
2. LLM selects relevant source(s) and generates sub-queries per source
3. Answers combined for final response

Sources can be replaced with agents/APIs/tools — LLM generates parameters for tool calls. (source: From Basic to Advanced RAG every step of the way.md)

## Generator-enhanced retrieval (synergy)

`FLAREInstructQueryEngine` — LLM actively elicits retrieval instructions before searching. See [[retrieval-generation-synergy]].

## Related pages

- [[rag-pipeline-stages]]
- [[retrieval-generation-synergy]]
- [[agentic-rag-and-react]]
