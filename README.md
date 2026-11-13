# RAG Retrieval Benchmark on SQuAD

I wanted to know how good different embedding models score on a given taks and how RAG retrieval tricks can actually help.
That's why I tested on SQuAD v1.1: BM25, dense retrieval with different embedding models, hybrid search (BM25 + dense, merged with Reciprocal Rank Fusion) and several rerankers on top.

(Work in progress, more to come.)

## What I did

- Corpus: all 20,958 SQuAD paragraphs (train + validation), each one a document
- Task: for each question, find the paragraph it was written for
- Embeddings: Qwen3-Embedding-8B / 4B, Voyage-4-Large, OpenAI text-embedding-3-large, BGE-M3 (using OpenRouter, FAISS exact search)
- Rerankers: Qwen3-Reranker-8B, Voyage Rerank-3-Lite, Cohere Rerank 3.5, applied to the top 20 hybrid candidates
- As this is more of a validation evaluation (deciding which model/technique to use), I decided to use 1500 questions from the train split of the dataset

## Results

![Final pipeline comparison](figures/final_pipeline.png)

### Embedding models: Dense vs. Hybrid

| Model | Dense R@1 | Dense MRR | Hybrid R@1 | Hybrid MRR | MRR gain |
|---|---:|---:|---:|---:|---:|
| Voyage-4-Large | **0.724** | **0.807** | **0.759** | **0.830** | +0.023 |
| Qwen3-Embedding-8B | 0.709 | 0.792 | 0.737 | 0.813 | +0.021 |
| e5-small-v2 (local) | 0.662 | 0.756 | 0.731 | 0.809 | +0.053 |
| Qwen3-Embedding-4B | 0.689 | 0.775 | 0.731 | 0.806 | +0.031 |
| BGE-M3 | 0.647 | 0.726 | 0.716 | 0.791 | +0.065 |
| OpenAI text-embedding-3-large | 0.607 | 0.704 | 0.705 | 0.785 | +0.081 |
| BM25 (reference) | 0.675 | 0.746 | | | |

### Rerankers (top 20 hybrid Qwen3-8B candidates)

| Reranker | Runs | R@1 | R@5 | MRR |
|---|---|---:|---:|---:|
| Qwen3-Reranker-8B | API | **0.896** | 0.961 | **0.926** |
| Voyage Rerank-3-Lite | API | 0.895 | **0.962** | 0.925 |
| Cohere Rerank 3.5 | API | 0.881 | 0.956 | 0.915 |
| BGE-Reranker-v2-M3 | local | 0.880 | 0.956 | 0.913 |
| MiniLM-L6 (ms-marco) | local | 0.818 | 0.941 | 0.871 |
| mxbai-rerank-xsmall | local | 0.809 | 0.935 | 0.865 |
| No reranking | | 0.737 | 0.902 | 0.813 |

R@20 is 0.965 for all rerankers, since they only reorder the 20 candidates.

- Reranking is a big factor for good rankings at the very top: Recall@1 goes from 0.74 to 0.90 and MRR from 0.81 to 0.93
- Hybrid beats dense alone (in MRR) for every embedding model, and helps the weaker ones most
- Hybrid beats dense alone for every embedding model, and helps the weaker ones most
- BM25 is already quite hard to beat on SQuAD: two of the five embedding models were worse than BM25 on their own

## Run it

Put your `OPENROUTER_API_KEY` in a `.env`, install the packages imported at the top of `main_nb.ipynb` and run the notebook. Embeddings and reranker responses are cached, so reruns are cheap. In order for the local models to work, you need sufficiently good hardware. If your hardware is as bad as mine I recommend to run this notebook as a Kaggle notebook.
