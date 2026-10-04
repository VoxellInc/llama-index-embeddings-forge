"""Tests for llama-index-embeddings-forge.

Shape tests run with no network. Live tests run only when FORGE_API_KEY is set
(against api.voxell.ai). Run: python -m unittest discover -s test
"""
import os
import unittest

from llama_index.embeddings.forge import ForgeEmbedding

LIVE = os.environ.get("FORGE_API_KEY")
DIMS = {"turbo": 1024, "pro": 2560, "ultra": 4096}


class ShapeTests(unittest.TestCase):
    def test_missing_key_raises(self):
        env_key = os.environ.pop("FORGE_API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                ForgeEmbedding(model="turbo")
        finally:
            if env_key is not None:
                os.environ["FORGE_API_KEY"] = env_key

    def test_class_name(self):
        self.assertEqual(ForgeEmbedding.class_name(), "ForgeEmbedding")

    def test_fields_and_body(self):
        e = ForgeEmbedding(model="pro", api_key="test-key", dimensions=256)
        self.assertEqual(e.model_name, "pro")
        self.assertEqual(e._url, "https://api.voxell.ai/v1/embed")
        self.assertEqual(e._headers["Authorization"], "Bearer test-key")
        self.assertTrue(e._headers["User-Agent"].startswith("llama-index-embeddings-forge/"))
        body = e._body(["a", "b"], "document")
        self.assertEqual(body, {"texts": ["a", "b"], "model": "pro", "input_type": "document", "dim": 256})

    def test_no_dim_omits_field(self):
        e = ForgeEmbedding(model="turbo", api_key="test-key")
        self.assertNotIn("dim", e._body(["x"], "query"))

    def test_version_matches_package_metadata(self):
        # The User-Agent and __version__ must report the installed package version.
        from importlib.metadata import version

        from llama_index.embeddings import forge

        installed = version("llama-index-embeddings-forge")
        self.assertEqual(forge.__version__, installed)
        e = ForgeEmbedding(model="turbo", api_key="test-key")
        self.assertEqual(e._headers["User-Agent"], f"llama-index-embeddings-forge/{installed}")


@unittest.skipUnless(LIVE, "FORGE_API_KEY not set — skipping live tests")
class LiveTests(unittest.TestCase):
    def test_text_and_query_dims_per_tier(self):
        for tier, dim in DIMS.items():
            e = ForgeEmbedding(model=tier)
            self.assertEqual(len(e.get_text_embedding("a document")), dim, f"{tier} text")
            self.assertEqual(len(e.get_query_embedding("a query")), dim, f"{tier} query")

    def test_batch(self):
        e = ForgeEmbedding(model="turbo")
        vecs = e.get_text_embedding_batch(["one", "two", "three"])
        self.assertEqual(len(vecs), 3)
        self.assertTrue(all(len(v) == DIMS["turbo"] for v in vecs))

    def test_matryoshka(self):
        e = ForgeEmbedding(model="turbo", dimensions=256)
        self.assertEqual(len(e.get_text_embedding("truncate me")), 256)


@unittest.skipUnless(LIVE, "FORGE_API_KEY not set — skipping live tests")
class LiveAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_aget_text_and_query(self):
        e = ForgeEmbedding(model="pro")
        self.assertEqual(len(await e.aget_text_embedding("doc")), DIMS["pro"])
        self.assertEqual(len(await e.aget_query_embedding("q")), DIMS["pro"])


if __name__ == "__main__":
    unittest.main()
