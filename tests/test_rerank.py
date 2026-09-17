from __future__ import annotations

import numpy as np

from ragdesk.rerank import RERANK_BATCH, OnnxReranker
from ragdesk.search import Hit


class _Encoding:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1] * len(ids)
        self.type_ids = [0] * len(ids)


class _Tokenizer:
    def __init__(self):
        self.calls: list[int] = []

    def encode_batch(self, pairs):
        self.calls.append(len(pairs))
        return [_Encoding([1, 2]) for _ in pairs]


class _Output:
    def __init__(self, name):
        self.name = name
        self.shape = ["batch", 1]


class _Session:
    """Scores by position so batch order can be verified end to end."""

    def __init__(self):
        self.next = 0
        self.batches = 0

    def get_inputs(self):
        return []

    def get_outputs(self):
        return [_Output("logits")]

    def run(self, _outputs, feed):
        size = len(feed["input_ids"])
        self.batches += 1
        scores = [[float(self.next + i)] for i in range(size)]
        self.next += size
        return [np.array(scores)]


def _hit(text: str, index: int) -> Hit:
    return Hit(
        chunk_id=index,
        doc_id=index,
        path=f"doc{index}.md",
        source="test",
        ordinal=0,
        text=text,
        score=1.0,
        cosine=1.0,
        lanes="dense",
    )


def test_rerank_batches_and_keeps_order():
    reranker = OnnxReranker(repo="someone/custom-reranker")
    session = _Session()
    reranker._tokenizer = _Tokenizer()
    reranker._session = session
    reranker._input_names = ["input_ids", "attention_mask"]

    hits = [_hit(f"text {i}", i) for i in range(20)]
    ranked = reranker.rerank("question", hits)

    assert session.batches == 3  # 8 + 8 + 4
    assert reranker._tokenizer.calls == [RERANK_BATCH, RERANK_BATCH, 4]
    assert reranker.loaded
    # Scores grew with position, so the last document wins — batching must not
    # shuffle the mapping between a score and its hit.
    assert ranked[0].path == "doc19.md"
    assert ranked[-1].path == "doc0.md"


def test_unload_drops_the_session():
    reranker = OnnxReranker(repo="someone/custom-reranker")
    reranker._tokenizer = _Tokenizer()
    reranker._session = _Session()
    reranker.unload()
    assert not reranker.loaded
    assert reranker._tokenizer is None


def test_rerank_empty_is_free():
    reranker = OnnxReranker(repo="someone/custom-reranker")
    assert reranker.rerank("question", []) == []
    assert not reranker.loaded  # never even loads the model
