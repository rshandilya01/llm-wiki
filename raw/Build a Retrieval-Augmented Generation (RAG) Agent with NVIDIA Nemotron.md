---
title: "Build a Retrieval-Augmented Generation (RAG) Agent with NVIDIA Nemotron"
source: "https://developer.nvidia.com/blog/build-a-rag-agent-with-nvidia-nemotron/?ncid=pa-srch-goog-984264&_bt=800923065558&_bk=rag%20architecture&_bm=b&_bn=g&_bg=200126387731&gad_source=1&gad_campaignid=23669821421&gbraid=0AAAAAD4XAoHt8_ow7Ir9b9h9axQP1TVfI&gclid=Cj0KCQjwlqTRBhCBARIsANrkrxgll3qbOgf9i8UT1Pu53Fz-4JCWOHD71lbW7_e4keQA6cNJ-sqn7hgaAtBVEALw_wcB"
author:
  - "[[Edward Li]]"
published: 2025-09-23
created: 2026-06-10
description: "Unlike traditional LLM-based systems that are limited by their training data, retrieval-augmented generation (RAG) improves text generation by incorporating…"
tags:
  - "clippings"
---
Unlike traditional LLM-based systems that are limited by their training data, [retrieval-augmented generation (RAG)](https://www.nvidia.com/en-us/glossary/retrieval-augmented-generation/) improves text generation by incorporating relevant external information. [Agentic RAG](https://developer.nvidia.com/blog/traditional-rag-vs-agentic-rag-why-ai-agents-need-dynamic-knowledge-to-get-smarter/) goes a step further by leveraging autonomous systems integrated with [LLMs](https://www.nvidia.com/en-us/glossary/large-language-models/) and retrieval mechanisms. This allows these systems to make decisions, adapt to changing requirements, and perform complex reasoning tasks dynamically.

In this guide to the [self-paced workshop for building a RAG agent](https://brev.nvidia.com/launchable/deploy?launchableID=env-32kC34ErT9wsqTcJyaKMxBEuhr2), you’ll gain:

- Understanding of the core principles of agentic RAG, including NVIDIA [Nemotron](https://developer.nvidia.com/ai-models#:~:text=NVIDIA%20Nemotron,-The%20NVIDIA%20Nemotron), an open model family with open data and weights.
- Knowledge of how to build agentic RAG systems using LangGraph.
- A turnkey, portable development environment.
- Your own customized agentic RAG system, ready to share as an [NVIDIA Launchable](https://developer.nvidia.com/brev).

## Video walkthrough

![](https://www.youtube.com/watch?v=Mw9_ZK3tnrQ)

*Video 1. Build a RAG Agent with NVIDIA Nemotron*

## Opening the workshop

Launch the workshop as an [**NVIDIA Launchable**](https://brev.nvidia.com/launchable/deploy?launchableID=env-32kC34ErT9wsqTcJyaKMxBEuhr2):

![Button of the “Deploy Now” button for NVIDIA DevX  Workshop](https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/Figure-2.-Button-to-deploy-the-NVIDIA-DevX-Workshop-in-the-cloud-with-NVIDIA-Brev-1-png-1.webp)

Figure 1. Click on the ‘Deploy Now’ button to deploy the NVIDIA DevX Workshop in the cloud

With your Jupyter Lab environment running, locate the **NVIDIA DevX Learning Path** section of the Jupyterlab Launcher. Select the **Agentic RAG** tile to open up the lab instructions and get started.

![A screenshot of the 2. Agentic RAG tile](https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/Figure-2.-Agentic-RAG-tile-in-NVIDIA-DevX-Learning-Path-to-open-lab-instructions-png.webp)

Figure 2. Click on the “Agentic RAG” tile in NVIDIA DevX Learning Path to open lab instructions.

## Setting up secrets

In order to follow along with this workshop, you’ll need to gather and configure a few project secrets.

- [**NGC API Key**](https://org.ngc.nvidia.com/setup/api-key): This enables access to NVIDIA software, models, containers, and more
- (optional) [**LangSmith API Key**](https://smith.langchain.com/): This connects the workshop to LangChain’s platform for tracing and debugging your AI Agent

You can utilize the **Secrets Manager** tile under **NVIDIA DevX Learning Path** of the Jupyterlab Launcher to configure these secrets for your workshop development environment. Verify in the logs tab that the secrets have been added successfully.

![A screenshot of the Secrets Manager tile under NVIDIA DevX Learning Path.](https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/Figure-3.-Use-the-Secrets-Manager-tile-under-the-NVIDIA-DevX-Learning-Path-section-to-configure-project-secrets-1-png.webp)

Figure 3. Use the “Secrets Manager” tile under the NVIDIA DevX Learning Path section to configure project secrets (API keys).

## Introduction to RAG architecture

Once your workshop environment has been set up, the next step is understanding the architecture of the agentic RAG system you’ll build.

RAG enhances the capabilities of LLMs by incorporating relevant external information during output text generation. Traditional language models generate responses based solely on the knowledge captured in their training data, which can be a limiting factor, especially when dealing with rapidly changing information, highly specialized knowledge domains, or enterprise confidential data. RAG, on the other hand, is a powerful tool for generating responses based on relevant unstructured data retrieved from an external knowledge base.

![A flow chart showing the path of a user prompt takes from the retrieval chain, to the LLM, to the final generated response.](https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/Figure-4.-Traditionally-RAG-utilizes-a-user-prompt-to-retrieve-contextually-relevant-documents-providing-them-as-context-to-the-LLM-for-a-more-informed-response-png.webp)

Figure 4. Traditionally, RAG utilizes a user prompt to retrieve contextually-relevant documents, providing them as context to the LLM for a more informed response.

The typical flow for a RAG system is:

1. **Prompt**: A user generates a natural language query.
2. **Embedding Model**: The prompt is converted into vectors
3. **Vector Database Search**: After a user’s prompt is embedded into a vector, the system searches a vector database filled with semantically indexed document chunks, enabling fast retrieval of contextually relevant data chunks.”
4. **Reranking Model**: The retrieved data chunks are reranked to prioritize the most relevant data.
5. **LLM**: The LLM generates responses informed by the retrieved data.

This approach ensures that the language model can access up-to-date and specific information beyond its training data, making it more versatile and effective.

## Understanding ReAct agent architecture

Unlike traditional LLM-based applications, agents can dynamically choose tools, incorporate complex [reasoning](https://www.nvidia.com/en-us/glossary/ai-reasoning/), and adapt their analysis approach based on the situation at hand.

![A flow chart showing the path a user prompt takes inside of a ReAct agent to iteratively utilize tool calling.](https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/Figure-5.-A-ReAct-agent-can-iteratively-reason-and-call-out-to-user-defined-tools-to-generate-a-higher-quality-RAG-based-response-png.webp)

Figure 5. A ReAct agent can iteratively reason and call out to user-defined tools to generate a higher quality RAG-based response.

ReAct Agents are a simple agentic architecture that use “reasoning and acting” via tool calling supported LLMs. If the LLM requests any tool calls after taking in the prompt, those tools will be run, added to the chat history, and sent back to the model to be invoked again.

RAG works well, but it’s limited because the LLM can’t determine how data is retrieved, control for data quality, or choose between data sources. Agentic RAG takes the concept of RAG a step further by combining the strengths of LLMs such as language comprehension, contextual reasoning, and flexible generation, with dynamic tool usage, and advanced retrieval mechanisms such as semantic search, hybrid retrieval, reranking, and data source selection. Making a ReAct Agent for RAG just requires giving it the Retrieval Chain as a tool so the agent can decide when and how to search for information.

![A flow chart showing the path a user prompt takes between the ReAct agent and the Retrieval Chain.](https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/Figure-6.-The-full-agentic-RAG-pipeline-will-involve-adding-the-ReAct-agent-to-the-Retrieval-Chain-where-the-contextual-documents-are-stored-jpg.webp)

Figure 6. The full agentic RAG pipeline will involve adding the ReAct agent to the Retrieval Chain where the contextual documents are stored.

Agentic RAG employs a ReAct agent architecture in which the reasoning LLM systematically decides whether to retrieve information via tool calling or respond directly, activating the retrieval pipeline only when additional context is needed to better address the user’s request.

## Learn and implement the code

Now that we understand the concepts, let’s dive into the technical implementation. We’ll start with the foundational components before building up to the complete agentic RAG system:

1. Models
2. Tools
3. Data Ingestion
4. Text Splitting
5. Vector Database Ingestion
6. Document Retriever and Reranker
7. Retriever Tool Creation
8. Agent Configuration

### Foundations: the models

The workshop relies on [NVIDIA NIM endpoints](https://developer.nvidia.com/nim) for the core model powering the agent. NVIDIA NIM provides high-performance inference capabilities, including:

- **Tool binding**: Native support for function calling.
- **Structured output**: Built-in support for Pydantic models.
- **Async operations**: Full async/await support for concurrent processing.
- **Enterprise reliability**: Production-grade inference infrastructure.

This example shows the ChatNVIDIA LangChain connector using NVIDIA NIM.

```python
fromlangchain_nvidia_ai_endpoints importChatNVIDIA
LLM_MODEL ="nvidia/nvidia-nemotron-nano-9b-v2"
llm =ChatNVIDIA(model=LLM_MODEL, temperature=0.6, top_p=0.95, max_tokens=8192)
```

To ensure the quality of the LLM-based application, it’s crucial that the agent receives clear instructions to clarify decision-making, remove ambiguity, and clarify how it should treat retrieved documents. One such example from `code/rag_agent.py` is provided as follows:

```python
SYSTEM_PROMPT =(
    "You are an IT help desk support agent.\n"
    "- Use the 'company_llc_it_knowledge_base' tool for questions likely covered by the internal IT knowledge base.\n"
    "- Always write grounded answers. If unsure, say you don't know.\n"
    "- Cite sources inline using [KB] for knowledge base snippets.\n"
    "- If the knowledge base doesn't contain sufficient information, clearly state what information is missing.\n"
    "- Keep answers brief, to the point, and conversational."
)
```

This prompt shows a few key principles of reliable LLM prompting for RAG-based applications:

- **Role specification**: Clear definition of the agent’s expertise and responsibilities.
- **Tool Utilization**: Instruct the agent on which tools to use for specific tasks.
- **Grounding:** Emphasize the importance of providing answers based on reliable sources and the importance of admitting to uncertainty.
- **Source Citation:** Provide guidelines for citing sources to ensure transparency.
- **Communication Style:** Specify the desired communication style.

In `code/rag_agent.py` we define the models necessary for the IT Help Desk agent to answer user queries by utilizing the Knowledge Base.

- The LLM Model, **Nemotron Nano 9b V2**, is the primary reasoning model used for generating responses.
- The NVIDIA NeMo Retriever Embedding Model, **Llama 3.2 EmbedQA 1b V2**, is used for converting documents into vector embedding representations for storage and retrieval.
- The NeMo Retriever Reranking Model, **Llama 3.2 RerankQA 1b V2**, is used for reranking for the most relevant retrieved documents and data.

These open models collectively enable the IT Help Desk agent to answer user queries accurately by leveraging a combination of language generation, document retrieval, and reranking capabilities.

### Foundations: the tools

Our RAG agent will have access to the knowledge base provided at `./data/it-knowledge-base` that contains markdown files documenting common IT-related procedures. The retriever tool enables the agent to search the internal IT knowledge base for documents relevant to the user’s query.

A vector database stores, indexes, and queries numerical representations of vectorized embeddings, allowing for fast similarity searches of unstructured data like text, images, and audio. For our purposes, we use an in-memory FAISS database, which is efficient for spinning up small databases. In terms of data ingestion to‌ utilize the data in the knowledge base, we’ll focus on text ingestion. Additional features like multimodality should be considered for production use cases.

### Foundations: data ingestion

The embedding model utilized is [NeMo Retriever llama-3.2-nv-embedqa-1b-v2](https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2). This model creates embeddings for documents and queries that help in efficiently retrieving relevant documents from the knowledge base by comparing the semantic similarity between the query and the documents.

To ingest the documents, we’ll chunk the documents, embed those chunks into vectors, and then insert the vectors into the database. Before doing that, we need to load the data from our `./data/it-knowledge-base` directory using the LangChain DirectoryLoader.

```python
fromlangchain_community.document_loaders importDirectoryLoader, TextLoader
# Read the data
_LOGGER.info(f"Reading knowledge base data from {DATA_DIR}")
data_loader =DirectoryLoader(
    DATA_DIR,
    glob="**/*",
    loader_cls=TextLoader,
    show_progress=True,
)
docs =data_loader.load()
```

### Foundations: text splitting

Document splitting is controlled by two things: chunk size and chunk overlap.

Chunk size defines the maximum length of each text chunk. This ensures that each chunk is of an optimized size for processing by language models and retrieval systems. A chunk size that is too large may contain information less relevant to specific queries, while one too small may miss important context.

Chunk overlap defines the number of tokens that overlap between consecutive chunks. The goal is to ensure continuity and preserve context across chunks, thereby maintaining coherence in the retrieved information.

To perform text splitting efficiently, we use the RecursiveCharacterTextSplitter. This tool recursively splits documents into smaller chunks based on character length, so each chunk adheres to the defined chunk size and overlap parameters. It’s particularly useful for processing large documents, improving the information retrieval’s overall accuracy.

```python
fromlangchain.text_splitter importRecursiveCharacterTextSplitter
CHUNK_SIZE =800
CHUNK_OVERLAP =120
 
_LOGGER.info(f"Ingesting {len(docs)} documents into FAISS vector database.")
splitter =RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
chunks =splitter.split_documents(docs)
```

### Foundations: vector database ingestion

To facilitate efficient retrieval of relevant information, we need to ingest our large corpus of documents into a vector database. Now that we have broken down our documents into manageable chunks, we utilize the embedding model to generate vector embeddings for each document chunk.

These embeddings are numerical representations of the semantic content of the chunks. High-quality embeddings enable efficient similarity searches, allowing the system to quickly identify and retrieve the most relevant chunks in response to a user’s query.

The next step is to store the generated embeddings in an in-memory FAISS database, which ensures fast indexing and querying capabilities for real-time information retrieval. In this example, we leverage the fact that LangChain’s FAISS \`from\_documents\` method conveniently generates the embeddings for the document chunks and also stores them in the FAISS vector store in one function call.

```python
fromlangchain_community.vectorstores importFAISS
fromlangchain_nvidia_ai_endpoints importNVIDIAEmbeddings,
 
embeddings =NVIDIAEmbeddings(model=RETRIEVER_EMBEDDING_MODEL, truncate="END")
vectordb =FAISS.from_documents(chunks, embeddings)
```

By following these steps and taking advantage of the power of the embedding model, we ensure that the IT Help Desk agent can efficiently retrieve and process relevant information from the knowledge base.

**Foundations: document retriever and reranker**

With our vector database populated, we can build a chain for content retrieval. This involves creating a seamless workflow that includes both the embedding step and the lookup step.

![A flow chart showing the path ingested document chunks take to get stored in a vector database.](https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/Figure-7.-A-basic-retrieval-chain-consists-of-an-embedding-model-and-a-database-to-store-the-converted-vector-embeddings-png.webp)

Figure 7. A basic retrieval chain consists of an embedding model and a database to store the converted vector embeddings.

In the embedding step, user queries are converted into embeddings using the same model that we previously used for document chunks. This ensures that both the queries and document chunks are represented in the same semantic space, enabling accurate similarity comparisons.

To initialize the retriever in this example, we’ll use semantic similarity and search for the top six returned results compared to our query.

```python
# imports already handled
kb_retriever =vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})
```

The embeddings of the user’s queries are compared against the embeddings stored in the vector database during the lookup step. The system retrieves the most similar document chunks, which are then used to generate responses.

![A flow chart showing the path ingested document chunks take to get stored in and retrieved from a vector database.](https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/Figure-8.-A-more-complex-retrieval-chain-consists-of-attaching-a-Reranking-model-to-reorganize-retrieved-context-to-put-the-most-relevant-chunks-first-png.webp)

Figure 8. A more complex retrieval chain consists of attaching a Reranking model to reorganize retrieved context to put the most relevant chunks first.

For both the embedding and the reranking models, we’ll use NIM microservices from [NVIDIA NeMo Retriever](https://developer.nvidia.com/nemo-retriever). LangChain allows us to easily create a basic retrieval chain from our Vector Database object that has both the embedding step and the lookup step.

For improving the relevance and order of retrieved documents, we can utilize the [NVIDIA Rerank](https://build.nvidia.com/nvidia/llama-3_2-nv-rerankqa-1b-v2?snippet_tab=LangChain) class, built on the NVIDIA NeMo Retriever Reranker model. The Reranker model evaluates and ranks the retrieved document chunks based on their relevance to the user’s query so that the most pertinent information is presented to the user first. In this example, we initialize the Reranker as follows:

```python
fromlangchain_nvidia_ai_endpoints importNVIDIARerank
reranker =NVIDIARerank(model=RETRIEVER_RERANK_MODEL)
```

### Foundations: Retriever tool creation

Taking the document retriever and the documenter reranker, we can now create the final document retriever as below:

```python
RETRIEVER =ContextualCompressionRetriever(
    base_retriever=kb_retriever,
    base_compressor=reranker,
)
```

The LangChain ContextualCompressionRetriever makes it easy to combine a retriever with additional processing steps, attaching the retrieval chain to the reranking model. Now we can create the retriever tool that enables our ReAct Agent.

In this example, we can initialize the retriever tool by using the LangChain tools package below, passing in our initialized retriever:

```python
fromlangchain.tools.retriever importcreate_retriever_tool
RETRIEVER_TOOL =create_retriever_tool(
    retriever=RETRIEVER,
    name="company_llc_it_knowledge_base",
    description=(
        "Search the internal IT knowledge base for Company LLC IT related questions and policies."
    ),
)
```

### Foundations: agent configuration

With our vector database and retriever chain in place, we’re ready to construct the agent graph. This agent graph acts as a kind of flowchart, mapping out the possible steps the model can take to accomplish its task. In traditional, step-by-step LLM applications, these are called “chains.” When the workflow involves more dynamic, non-linear decision-making, we refer to them as “graphs.” The agent can choose different paths based on the context and requirements of the task at hand, branching out into different decision nodes.

Given the prevalence of the ReAct agent architecture, LangGraph provides a function that’ll create ReAct Agent Graphs. In this example, we utilized as below:

```python
fromlanggraph.prebuilt importcreate_react_agent
AGENT =create_react_agent(
    model=llm,
    tools=[RETRIEVER_TOOL],
    prompt=SYSTEM_PROMPT,
)
```

By constructing an agent graph, we create a dynamic and flexible workflow that enables our IT Help Desk agent to handle complex decision-making processes. This approach ensures that the agent can efficiently retrieve and process information, provide accurate responses, and adapt to various scenarios.

### Running your agent

**Congratulations! You have successfully built your agent! Now, the next step is to try it out.**

To get started with running your agent from your terminal, cd into the code directory that has the Python file containing your code for the agent. Once there, start your Agent API with the LangGraph CLI. Your agent will automatically reload as you make changes and save your code.

```python
langraph dev
```

To chat with your agent, a simple Streamlit app has been included in the **Simple Agents Client***.* You can also access the Streamlit Client from the Jupyter Launcher page. In the sidebar, ensure the rag\_agent client is selected and try chatting!

![A screenshot of the Simple Agents Client tile](https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/Figure-9.-Simple-Agents-Client-tile-in-NVIDIA-DevX-Learning-Path-to-open-the-Streamlit-chat-application-png.webp)

Figure 9. Click on the “Simple Agents Client” tile in NVIDIA DevX Learning Path to open the Streamlit chat application.

As your agents become more sophisticated, managing their internal complexity can become difficult. Tracing helps visualize each step your agent takes, which makes it easier to debug and optimize your agent’s behavior. In the workshop, you can optionally configure the LANGSMITH\_API\_KEY and view traces on the [LangSmith dashboard](https://smith.langchain.com/).

## Migrate to local NIM microservices

This workshop utilizes the [nvidia-nemotron-nano-9b-v2](https://build.nvidia.com/nvidia/nvidia-nemotron-nano-9b-v2) LLM from the [NVIDIA API Catalog](https://build.nvidia.com/). These APIs are useful for evaluating many models, quick experimentation, and getting started is free. However, for the unlimited performance and control needed in production, deploy models locally with NVIDIA NIM microservice containers.

In a typical development workflow, both your agent and NIM containers would run in the background, allowing you to multitask and iterate quickly. For this exercise, we can run the NIM in the foreground to easily monitor its output and ensure proper start up.

First, you need to log in to the NGC container registry as follows:

The next step is to create a location for NIM containers to save their downloaded model files.

```python
docker volume create nim-cache
```

Now, we need to use a Docker run command to pull the NIM container image and model data files before hosting the model behind a local, OpenAI-compliant API.

```python
docker run -it --rm \
    --name nemotron \
    --network workbench \
    --gpus 1\
    --shm-size=16GB\
    -e NGC_API_KEY=$NVIDIA_API_KEY \
    -v nim-cache:/opt/nim/.cache \
    -u $(id-u) \
    -p 8000:8000\
    nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2:latest
```

After letting it run for a few minutes, you’ll know the NIM is ready for inference when it says *Application startup complete*.

```python
INFO 2025-09-1016:31:52.7on.py:48] Waiting forapplication startup.
INFO 2025-09-1016:31:52.239on.py:62] Application startup complete.
INFO 2025-09-1016:31:52.240server.py:214] Uvicorn running on http://0.0.0.0:8000(Press CTRL+C to quit)
...
INFO 2025-09-1016:32:05.957metrics.py:386] Avg prompt throughput: 0.2tokens/s, Avg generation throughput: 1.1tokens/s, Running: 0reqs, Swapped: 0reqs, Pending: 0reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
```

Now that your NIM is running locally, we need to update the agent you created in `rag_agent.py` to use it.

```python
llm =ChatNVIDIA(
    base_url="http://nemotron:8000/v1",
    model=LLM_MODEL,
    temperature=0.6,
    top_p=0.95,
    max_tokens=8192
)
```

With your *langgraph* server still running, go back to our *Simple Agents Client* and try prompting the agent again. If everything was successful, you should notice no change!

**Congratulations! You have now migrated to using Local NIM microservices for your LangGraph Agent!**

## Conclusion and next steps

This workshop provides a comprehensive path from basic concepts to sophisticated agentic systems, emphasizing hands-on learning with production-grade tools and techniques.

By completing this workshop, developers gain practical experience with:

- **Fundamental concepts:** Understanding the difference between standard and agentic RAG.
- **State management:** Implementing complex state transitions and persistence.
- **Tool integration:** Creating and managing agentic tool-calling capabilities.
- **Modern AI stack:** Working with LangGraph, NVIDIA NIM, and associated tooling.

### Learn more

For hands-on learning, tips, and tricks, watch our [Nemotron Labs](https://www.addevent.com/calendar/Og917781) livestream replay, “ [Build a RAG Agent with NVIDIA Nemotron](https://www.youtube.com/watch?v=f0utW0eLueY) ”.

- Try NVIDIA Nemotron on [Hugging Face](https://huggingface.co/nvidia/collections?search=nemotron). See the workshop on [GitHub](https://github.com/brevdev/workshop-build-an-agent).
- Ask questions on the [Nemotron developer forum](https://forums.developer.nvidia.com/c/ai-data-science/nvidia-nemotron/669) or the [Nemotron channel](https://discord.com/channels/1019361803752456192/1407781691698708682) on [Discord](https://discord.com/invite/nvidiadeveloper)
- Want to learn more? “ [Build a Report Generator AI Agent with NVIDIA Nemotron on OpenRouter](https://developer.nvidia.com/blog/build-a-report-generator-ai-agent-with-nvidia-nemotron-on-openrouter/) ” in the same NVIDIA DevX Workshop.

*Stay up to date on* [*NVIDIA Nemotron*](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) *by subscribing to* [*NVIDIA news*](https://www.nvidia.com/en-us/ai-data-science/generative-ai/news/) *and following NVIDIA AI on* [*LinkedIn*](https://www.linkedin.com/showcase/nvidia-ai/posts/?feedView=all)*,* [*X*](https://x.com/NVIDIAAIDev)*,* [*Discord*](https://discord.com/invite/nvidiadeveloper)*, and* [*YouTube*](https://www.youtube.com/@NVIDIADeveloper)*.*

- *Visit our* [*Nemotron developer page*](https://developer.nvidia.com/nemotron) *for all the essentials you need to get started with the most open, smartest-per-compute reasoning model.*
- *Explore new open Nemotron models and datasets on* [*Hugging Face*](https://huggingface.co/collections/nvidia/nvidia-nemotron-689f6d6e6ead8e77dd641615) *and* [*NIM microservices*](https://build.nvidia.com/models?filters=publisher%3Anvidia&q=Nemotron) *and* [*Blueprints*](https://build.nvidia.com/blueprints) *on* [*build.nvidia.com*](http://build.nvidia.com/)*.*
- [*Share your ideas*](http://nemotron.ideas.nvidia.com/?nvid=nv-int-tblg-905992) *and vote on features to help shape the future of Nemotron.*
- *Tune into upcoming* [*Nemotron livestreams*](https://www.addevent.com/calendar/Og917781) *and connect with the NVIDIA Developer community through* [*the Nemotron developer forum*](https://forums.developer.nvidia.com/c/ai-data-science/nvidia-nemotron/669) *and the* [*Nemotron channel*](https://discord.com/channels/1019361803752456192/1407781691698708682) *on* [*Discord*](https://discord.com/invite/nvidiadeveloper)

*Browse* [*video tutorials and livestreams*](https://www.youtube.com/playlist?list=PL5B692fm6--vdRKB14FImVi7MTJ77zjn4) *to get the most out of NVIDIA Nemotron*