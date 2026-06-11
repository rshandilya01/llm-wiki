---
title: "From Basic to Advanced RAG every step of the way"
source: "https://rahuld3eora.medium.com/from-basic-to-advanced-rag-every-step-of-the-way-dee3a3a1aae9"
author:
  - "[[Rahul Deora]]"
published: 2024-02-25
created: 2026-06-11
description: "More"
tags:
  - "clippings"
---
RAG or Retrieval Augmented Generation, provides LLMs with the ability to retrieve information from one or multiple data sources and use those pieces of information to answer a user query. Setting up a basic RAG system is relatively straightforward, yet developing one that is both robust and reliable presents numerous challenges and pitfalls, especially when optimizing for computational efficiency.

In this blog, we’ll explore common pitfalls in developing RAG systems and introduce advanced techniques aimed at enhancing retrieval quality, minimizing hallucinations, and tackling complex queries. By the conclusion of this post, you’ll gain a deeper understanding of the intricacies involved in constructing RAG systems and learn how you can get started in tacking them.

## Basic to Advanced RAG

Below is a diagram of the basic flow of RAG which most people use. It shows the basic flow of query -> retrieval -> answer

![Basic RAG](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*tAGA8bIvsul5hNyUXyib7w.png)

Basic RAG flow. Diagram credits

Above we see that this basic flow of query -> retrieval -> answer contains many steps. Particularly it involves converting the query into an embedding by an embedding model, then finding similarity with an existing database of your data source selecting the top relevant documents, and finally feeding this back into an LLM to answer the initial query. We can write this as

query -> query embedding -> similarity search-> retrieval -> context -> answering

This flow above has 5 stages and each stage comes with its points of failure, for example:

- **query**: The user may not write a good query
- **query embedding**: The query model does not create a good representative embedding
- **similarity search**: Similarity search misses the most relevant information from the document source or the document source does not have the relevant information
- **retrieval**: Most of the retrieved information is irrelevant or lacking context
- **context**: While answering the query may use our past knowledge or not properly use the information from the retrieved context

Thus to create a strong and reliable RAG system we must interject at each of these stages, which are represented by arrows in the above diagram. This is what we call an Advanced RAG system:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Kr5D1XD6ryst1i5fEnU7XA.png)

Advanced RAG flow. Diagram credits

Don’t be afraid to look at this! Each of those new components added can be explained very simply and naturally.

We will cover each new component in green added above and its motivations and benefits in detail below.

## Query Transformation

Many times the user’s input query is not well framed, grammatically incorrect, or ambiguous, and answering the query may require multiple steps of reasoning.

In case of the query not being framed well or ambiguous we can simply use an LLM and ask it to better frame the query. For dealing with complex queries that require multiple reasoning steps we can do [**Multi-Step Query Decomposition**](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/query_transformations.html#multi-step-query-transformations):

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*nWYVnvkqqJEhWOsXnVmWOQ.png)

Breaking a query into sub-queries. Source

Here the query is broken into sub-queries. Each sub-query is then answered using the basic RAG flow introduced at the start. Each answer is then fed into an LLM again along with the original question to synthesise the final answer.

We can generate the sub-queries sequentially as shown above or all at the same time. For example, for the query ‘Is the average temperature in summer in Berlin more than its average temperature in 2005’ the sub-queries generated would be ‘Current Average temperature in Berlin’, ‘ Average temperature in Berlin in 2005’, then using the answer of these two queries an LLM will be able to give us the final answer.

Query decomposition is important as user queries that pose analytical or logical questions may not be directly stated in the document list. For example, there may be no sentence that says ‘Berlin’s average temperature today is less than it was in 2005', so doing a direct vector search with the user query would yield only irrelevant documents.

## Query Routing

Often we have multiple data sources or have grouped our documents to make them more compact to retrieve. The user’s answer will often combine the knowledge of multiple sources so we need to know which document source to retrieve from and what information we could obtain from the source.

As an example below we see that we query Houston and Boston databases separately with different sub-queries and utilize the answers to queries for the final answer.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*HzegPilty6hGnj7ucpbEJg.png)

Query Routing with Houston and Boston data sources

Each document source has a description which we provide the LLM that broadly tells the LLM what information is contained in that document source. This description is normally user generated eg) The Huston document source above could have a description as ‘All data about sports teams in Houston’.

Then we ask an LLM which document source is relevant and to generate a list of sub-queries over that document source which would be helpful to answer the user’s question.

Over here we can also replace data sources with an agent/API/tool and ask the LLM to come up with parameters to make that agent perform a specific task and then use the output downstream.

You can go over these references for a working example: [Colab](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/usecases/10k_sub_question.ipynb#scrollTo=8775650f-b164-478c-8129-9a8e6a0cdc97) and [YouTube](https://www.youtube.com/watch?v=GT_Lsj3xj1o&ab_channel=LlamaIndex)

## Retrieval

Now that we can better handle the user’s query, let’s go over two advanced methods of retrieval in this section to understand how we can better identify relevant documents over just doing basic RAG

### Dense Passage Retrieval

Normally we use the same embedding model to encode the documents and query. However, this embedding model is a general-purpose embedding model and may not be able to transform the query in an embedding space that represents relevance to what is required to answer the query. Methods like [HyDE](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/query_transformations.html#hyde-hypothetical-document-embeddings) try to circumvent this but are not very effective due to hallucinations.

To improve the performance of retrieval we can use [Dense Passage Retrievers](https://arxiv.org/abs/2004.04906) which use different encoders to encode the query and documents. They have been trained using a Siamese objective on question-answer pairs.

![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*x9Xr9whVR3uQsgj8bQvk1Q.png)

Dense Passage Retrieval

## CoBERT

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*_ruKh6sM0mAzNlcK)

Retrieval Methods

CoBERT offers a different way of ranking and retrieval which is more fine-grained than the vanilla cosine embedding comparison and even the Dense Passage Retrieval. The main intuition of CoBERT is to do a comparison across word embeddings instead of compressing the query and document to a limited one-dimensional embedding vector which may not be able to summarise all the content in the query and document.

The advantage of such retrieval is fine-grained similarity from different query tokens, for example, consider a query like “Gross Revenue of all the products of Company X in the year 2005”. The key terms here are ‘Gross Revenue’, ‘all products’, and ‘year 2005’. The overall query shares high similarity with the passage speaking about Company X’s revenue, Company X’s revenue in particular years, a few of the products of Company X’s revenue, Gross Revenue of the company in 2005, and so on. You see there are so many things around the exact query of interest so doing dense embeddings would be extremely noisy.

Instead of condensing our query and document into a single embedding vector, we tokenize and pass these through separate encoders to obtain a set of enriched word embeddings. So for the input query tokens we go from q1, q2, q3… -> x1, x2, x3 …. and for a document tokens we go from d1,d2,d3…. -> y1, y2,y3…..

Now we do a pairwise similarity between each of these vectors to come up with a correlation matrix and we take the maximum across each row and sum those values to obtain a score called MaxSim.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*4PsEC660gZ25MroL.png)

Query-Document token correlation matrix. Standford CS224U

In essence, we compare each word embedding in the query to every word embedding in the document and pick the score with the max similarity as it measures a match of the query content. CoBERT obtains more than 2x the retrieval performance as compared to [BM25](https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever.html). The query/document encoders can also be fine-tuned for our task for slightly better results.

To learn more about CoBERT check out their [paper](https://arxiv.org/pdf/2004.12832.pdf) or this [article](https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/)

## Reranking And Post-processing

Now that we are better able to retrieve our documents, lets look at some ways to better re-rank and post-process our obtained documents before using them to generate our final answer.

### Reranking

Just like the above two methods, there are multiple methods we can use to retrieve content from our document source. We can use multiple dense embedding models or more classical methods like [BM25](https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever.html) to obtain multiple sets of top-k retrievals.

Once we obtain these different sets of top-k results we must do re-ranking within these sets to obtain a single set of top-k results using methods like [Reciprocal Rank Fusion](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking). There are many ways you could fuse multiple top-k sets including coming up with your heuristics based on experimenting with your retrieval methods with similarity cutoffs etc

## Post-processing: Chain of Note

Chain of Note(CoN) is a powerful and very popular method to post-process our retrievals to make them better suited for question answering. Chain of Note helps in three main areas:

- Reducing LLM hallucinations
- Cases where a document has conflicting or impartial information
- Cases where the LLM must use its knowledge along with the content to answer the query

CoN asks an LLM to make notes for each retrieval. This allows the LLM the carefully inspect each retrieval’s relevance and focus on key information relevant to the query. CoN adds a layer of note-taking in between which is similar to chain of thought prompting.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*2LDAOBF-XTU8RwOuOhQVYA.png)

Chain Of Note

In the above diagram we see how CoN falls in an intersection of RAG and using Contextual Insights along with LLM Knowledge to better handle points of failure.

Consider the below example:

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*1z1rqUc7jTK552R0NQwviw.png)

Chain Of Note Example

In the first example above we see how CoN helps in dealing with conflicting information, in the second how retrievals that are very close to relevance but still miss the required answer are dealt and finally when the retrievals are completely irrelevant.

Chain Of Note is widely used to improve retrieval performance. You can read about it [here](https://praveengovindaraj.com/cutting-through-the-noise-chain-of-notes-con-robust-approach-to-super-power-your-rag-pipelines-0df5f1ce7952).

We can hook up the internet also in CoN, check out a more advanced version of CoN: [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) which refines ambiguous snippets by searching the internet to fill in the missing content.

## Conclusion

We saw above that every point along the basic RAG pipeline of:  
query -> query embedding -> similarity search-> retrieval -> context -> answering

has pitfalls and can be augmented to improve the reliability of our RAG system. We covered query transformation and query routing to improve our query and better use our data sources/tools. We covered advanced retrieval techniques like Dense Passage Retrieval and CoBERT to improve the quality of our retrieved data and finally, we covered re-ranking and CoN post-processing to better utilize the retrieved passages for question answering.

By better understanding the points of failure and these techniques, we are better equipped to build highly reliable RAG systems. Don’t settle for basic RAG! Hope you found this blog helpful:)