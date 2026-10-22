import numpy as np
import pytest

from niyam import search
from niyam.ingest import Chunk
from niyam.search import INDEX_FILE, DenseIndex, Retriever, load_chunks, read_index, write_index


class FakeModel:
    """Stands in for the e5 model: every text embeds to the same vector, so no model download is needed."""

    def __init__(self, vector):
        self.vector = np.array(vector, dtype=np.float32)

    def embed(self, texts, batch_size=16):
        for _ in texts:
            yield self.vector


def test_index_file_round_trips_and_ranks_by_cosine(tmp_path):
    chunks = [Chunk(f"1-{i}", 1, "1/1/2017-GST", "2017-06-26", "Proper officer", 1, 2, f"passage {i}") for i in range(3)]
    vectors = np.eye(3, 4, dtype=np.float32)
    path = tmp_path / "index.npz"
    write_index(path, chunks, vectors)
    key, loaded, loaded_vectors = read_index(path)
    assert key == "e5" and loaded == chunks and np.array_equal(loaded_vectors, vectors)
    index = DenseIndex.from_file(path, model=FakeModel([0.1, 1.0, 0.2, 0.0]))
    assert [h.chunk.chunk_id for h in index.search("which officer?", k=3)] == ["1-1", "1-2", "1-0"]


def test_committed_index_file_matches_the_committed_passages():
    key, chunks, vectors = read_index(INDEX_FILE)
    assert key == "e5" and chunks == load_chunks()  # if not, re-run scripts/export_index.py
    assert vectors.shape == (len(chunks), 1024) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_niyam_index_env_var_selects_the_backend(monkeypatch):
    monkeypatch.setattr(search, "TextEmbedding", lambda name: FakeModel(np.ones(1024)))
    monkeypatch.setenv("NIYAM_INDEX", "file")
    assert Retriever().dense.chunks == load_chunks()
    monkeypatch.setenv("NIYAM_INDEX", "postgres")
    with pytest.raises(ValueError, match="NIYAM_INDEX"):
        Retriever()
