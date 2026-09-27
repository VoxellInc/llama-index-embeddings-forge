"""Forge embeddings for LlamaIndex.

Forge is Voxell's hosted text-embedding API. ``ForgeEmbedding`` is a LlamaIndex
:class:`~llama_index.core.base.embeddings.base.BaseEmbedding` implementation with
sync and async support.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field

DEFAULT_BASE_URL = "https://api.voxell.ai"
__version__ = "0.1.0"
_USER_AGENT = f"llama-index-embeddings-forge/{__version__}"


class ForgeError(RuntimeError):
    """Raised when the Forge API returns a non-success response."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"Forge API error {status_code}: {message}")


class ForgeEmbedding(BaseEmbedding):
    """LlamaIndex embeddings backed by the Forge API.

    Example::

        from llama_index.embeddings.forge import ForgeEmbedding

        emb = ForgeEmbedding(model="turbo")              # FORGE_API_KEY from env
        vec = emb.get_text_embedding("hello world")

    Args:
        model: Forge tier — ``"turbo"`` (1024d), ``"pro"`` (2560d), or
            ``"ultra"`` (4096d, highest quality). Stored as ``model_name``.
        api_key: Forge API key. Defaults to the ``FORGE_API_KEY`` env var.
        base_url: API base URL. Defaults to ``https://api.voxell.ai``.
        dimensions: Optional Matryoshka truncation (re-normalized), e.g. ``256``.
        timeout: Per-request timeout in seconds.
    """

    api_key: str = Field(description="Forge API key.")
    base_url: str = Field(default=DEFAULT_BASE_URL, description="Forge API base URL.")
    dimensions: Optional[int] = Field(
        default=None, description="Optional Matryoshka truncation dimension."
    )
    timeout: float = Field(default=30.0, description="Per-request timeout (seconds).")

    def __init__(
        self,
        model: str = "turbo",
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        dimensions: Optional[int] = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        key = api_key or os.environ.get("FORGE_API_KEY")
        if not key:
            raise ValueError(
                "Forge API key missing: pass api_key=... or set FORGE_API_KEY. "
                "Create one at https://dash.voxell.ai."
            )
        super().__init__(
            model_name=model,
            api_key=key,
            base_url=base_url.rstrip("/"),
            dimensions=dimensions,
            timeout=timeout,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "ForgeEmbedding"

    @property
    def _url(self) -> str:
        return f"{self.base_url}/v1/embed"

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }

    def _body(self, texts: List[str], input_type: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "texts": texts,
            "model": self.model_name,
            "input_type": input_type,
        }
        if self.dimensions is not None:
            body["dim"] = self.dimensions
        return body

    @staticmethod
    def _parse(resp: httpx.Response) -> List[List[float]]:
        if resp.status_code != 200:
            raise ForgeError(resp.status_code, resp.text[:200])
        embeddings = resp.json().get("embeddings")
        if not isinstance(embeddings, list):
            raise ForgeError(resp.status_code, "unexpected response shape")
        return embeddings

    # --- transport ---
    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        if not texts:
            return []
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self._url, headers=self._headers, json=self._body(texts, input_type))
        return self._parse(resp)

    async def _aembed(self, texts: List[str], input_type: str) -> List[List[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self._url, headers=self._headers, json=self._body(texts, input_type)
            )
        return self._parse(resp)

    # --- BaseEmbedding overrides ---
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query], "query")[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text], "document")[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(list(texts), "document")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._aembed([query], "query"))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._aembed([text], "document"))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self._aembed(list(texts), "document")
