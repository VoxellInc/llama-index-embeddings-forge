# llama-index-embeddings-forge

LlamaIndex embeddings for [**Forge**](https://voxell.ai/forge), Voxell's hosted text-embedding API.

Voxell's Ingot-8B-R3 ranks #1 for English on the public MTEB leaderboard (English v2), with a 75.98
mean task score across 41 tasks. It is the top usable English embedding model. See the
[model card](https://huggingface.co/JCorners/Ingot-8B-R3), or try Forge with no signup on the
[playground](https://playground.voxell.ai).

## Why Forge

One API, three tiers — pick your point on the quality/cost curve:

| Model | Dim | Notes |
| ----- | --- | ----- |
| `turbo` | 1024 | fast, low cost |
| `pro` | 2560 | balanced quality and cost |
| `ultra` | 4096 | highest quality, top tier |

Matryoshka (MRL) dimensions are real: truncated vectors are re-normalized, so a shorter `dim` is a
unit-norm prefix of the full vector — smaller index, minimal quality loss. Forge logs request
metadata only (model, tokens, latency) — never your text or vectors.

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
