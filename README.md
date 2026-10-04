# llama-index-embeddings-forge

LlamaIndex embeddings for [**Forge**](https://voxell.ai/forge), Voxell's hosted text-embedding API.

Voxell's Ingot-8B-R3 ranks #1 for English on the public MTEB leaderboard (English v2), with a 75.98
mean task score across 41 tasks. It is the top usable English embedding model. See the
[model card](https://huggingface.co/JCorners/Ingot-8B-R3), or try Forge with no signup on the
[playground](https://playground.voxell.ai).

## Retrieval, measured on public documents

Voxell publishes a retrieval receipt for each of four public corpora: 7,817 documents and 980,885
passages in total. Each corpus has 200 questions (800 in all), measured 2026-10-02.

| Corpus | Documents | First result answers the question | An answer in the top three |
| ------ | --------: | --------------------------------: | -------------------------: |
| SEC filings | 2,010 | 91% | 95% |
| USPTO patents | 4,008 | 86.5% | 92% |
| NASA technical reports | 1,210 | 72% | 80% |
| arXiv technical papers | 589 | 92% | 98% |
| All four | 7,817 | 85% | 91% |

The right document is in the top ten for 95% of the questions.

What these numbers are: the questions were written by a model from the documents, and each result
was judged against the passage text. That is easier than a test set written by people. The numbers
describe what Voxell's hosted retrieval does on these four corpora. They are not a comparison with
any other system, and they are not a promise about your corpus.

They measure the whole hosted pipeline, of which embedding is one step. This package gives
LlamaIndex the embeddings. It does not run the rest of that pipeline for you.

The receipts, with sample questions including ones the hosted retrieval got wrong, are at
[voxell.ai/retrieval](https://voxell.ai/retrieval/).

## Why Forge

One API, three tiers. Pick your point on the quality and cost curve:

| Tier | Dim | Notes |
| ----- | --- | ----- |
| `turbo` | 1024 | fast, low cost |
| `pro` | 2560 | balanced quality and cost |
| `ultra` | 4096 | highest quality, top tier |

Matryoshka (MRL) dimensions are real: truncated vectors are re-normalized, so a shorter `dim` is a
unit-norm prefix of the full vector, which means a smaller index with minimal quality loss. Forge
logs request metadata only (model, tokens, latency), never your text or vectors.

## Install

```bash
pip install llama-index-embeddings-forge
```

## Usage

```python
from llama_index.embeddings.forge import ForgeEmbedding

# FORGE_API_KEY is read from the environment; or pass api_key=...
embed_model = ForgeEmbedding(model="turbo")

vector = embed_model.get_text_embedding("the quick brown fox")
query = embed_model.get_query_embedding("fast animal")
batch = embed_model.get_text_embedding_batch(["doc one", "doc two"])
```

### As the global embed model

```python
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.embeddings.forge import ForgeEmbedding

Settings.embed_model = ForgeEmbedding(model="pro")
index = VectorStoreIndex.from_documents([Document(text="hello world")])
```

### Async

```python
vec = await embed_model.aget_text_embedding("doc")
q = await embed_model.aget_query_embedding("a search query")
```

### Matryoshka (shorter vectors)

```python
embed_model = ForgeEmbedding(model="turbo", dimensions=256)  # re-normalized 256-d vectors
```

## Configuration

| Arg | Default | Notes |
| --- | ------- | ----- |
| `model` | `"turbo"` | `turbo` \| `pro` \| `ultra` (stored as `model_name`) |
| `api_key` | `FORGE_API_KEY` env | get one at [dash.voxell.ai](https://dash.voxell.ai) |
| `base_url` | `https://api.voxell.ai` | |
| `dimensions` | `None` | Matryoshka truncation, e.g. `256` |
| `timeout` | `30.0` | seconds |

## License

MIT © Voxell, Inc.
