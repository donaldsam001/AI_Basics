from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.vector_store.faiss_store import FAISSStore
from src.vector_store.metadata_store import MetadataStore
from src.vector_store.cv_search import CVSearch


class FAISSStoreTests(unittest.TestCase):
    def test_add_embeddings(self):
        store = FAISSStore(2)
        store.add(np.array([[3.0, 4.0], [1.0, 0.0]]))
        self.assertEqual(store.size, 2)

    def test_search_returns_closest_vector_first(self):
        store = FAISSStore(2)
        store.add(np.array([[1.0, 0.0], [0.0, 1.0]]))
        scores, ids = store.search(np.array([0.9, 0.1]))
        self.assertEqual(ids[0], 0)
        self.assertAlmostEqual(float(scores[0]), 0.9938837, places=6)

    def test_inner_product_uses_cosine_ordering(self):
        store = FAISSStore(2)
        store.add(np.array([[10.0, 0.0], [1.0, 1.0], [-5.0, 0.0]]))
        scores, ids = store.search(np.array([2.0, 0.0]), 3)
        self.assertEqual(ids.tolist(), [0, 1, 2])
        self.assertTrue(np.all(np.diff(scores) <= 0))

    def test_save_load_preserves_search(self):
        with TemporaryDirectory() as directory:
            store = FAISSStore(2)
            store.add(np.array([[1.0, 0.0], [0.0, 1.0]]))
            path = f"{directory}/cv.index"
            store.save(path)
            expected = store.search(np.array([1.0, 0.1]))
            actual = FAISSStore.load(path).search(np.array([1.0, 0.1]))
            np.testing.assert_allclose(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])

    def test_empty_index_is_safe(self):
        scores, ids = FAISSStore(2).search(np.array([1.0, 0.0]))
        self.assertEqual(len(scores), 0)
        self.assertEqual(len(ids), 0)

    def test_dimension_validation(self):
        store = FAISSStore(2)
        with self.assertRaisesRegex(ValueError, "dimension"):
            store.add(np.array([1.0, 0.0, 0.0]))


class MetadataStoreTests(unittest.TestCase):
    def test_candidate_mapping(self):
        with TemporaryDirectory() as directory:
            path = f"{directory}/metadata.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump([{"faiss_id": 0, "candidate_id": "c1", "candidate_name": "Ada"}], handle)
            metadata = MetadataStore(path)
            self.assertEqual(metadata.get_candidate(0)["candidate_name"], "Ada")
            metadata.validate_index_size(1)

    def test_index_metadata_mismatch_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = f"{directory}/metadata.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump([{"faiss_id": 0}, {"faiss_id": 1}], handle)
            with self.assertRaisesRegex(ValueError, "contains 2 records"):
                MetadataStore(path).validate_index_size(1)


class CVSearchTests(unittest.TestCase):
    def test_search_combines_similarity_and_candidate_metadata(self):
        class FakeModel:
            def encode(self, texts, show_progress_bar=True):
                return np.array([[1.0, 0.0]])

        with TemporaryDirectory() as directory:
            path = f"{directory}/metadata.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump([
                    {"faiss_id": 0, "candidate_id": "java", "candidate_name": "Java Dev", "source_index": 0},
                    {"faiss_id": 1, "candidate_id": "python", "candidate_name": "Python Dev", "source_index": 1},
                ], handle)
            store = FAISSStore(2)
            store.add(np.array([[1.0, 0.0], [0.0, 1.0]]))
            results = CVSearch(FakeModel(), store, MetadataStore(path)).search("Java developer", k=2)
            self.assertEqual([result["candidate_id"] for result in results], ["java", "python"])
            self.assertTrue(results[0]["score"] >= results[1]["score"])


if __name__ == "__main__":
    unittest.main()
