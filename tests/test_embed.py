from __future__ import annotations

import numpy as np

from ragdesk.embed import OnnxEmbedder, onnx_prompts, position_limit


def test_prompts_are_per_family():
    query, doc = onnx_prompts("onnx-community/embeddinggemma-300m-ONNX")
    assert query.startswith("task: search result") and doc.startswith("title: none")

    assert onnx_prompts("Xenova/multilingual-e5-small") == ("query: ", "passage: ")
    assert onnx_prompts("Xenova/multilingual-gte-base") == ("", "")
    assert onnx_prompts("someones/custom-bert") == ("", "")  # never guess


class _Encoding:
    def __init__(self, ids, mask):
        self.ids = ids
        self.attention_mask = mask


class _Tokenizer:
    def encode_batch(self, texts):
        return [_Encoding([1, 2, 3, 4, 0, 0], [1, 1, 1, 1, 0, 0]) for _ in texts]


class _Output:
    def __init__(self, shape, name="last_hidden_state"):
        self.shape = shape
        self.name = name


class _Input:
    def __init__(self, name):
        self.name = name


class _Session:
    """Two tokens carry [3.0, 0.0] / [0.0, 4.0]; padding must not count."""

    def __init__(self, inputs=("input_ids", "attention_mask")):
        self._outputs = [_Output(["batch", "tokens", 2])]
        self._inputs = [_Input(name) for name in inputs]

    def get_outputs(self):
        return self._outputs

    def get_inputs(self):
        return self._inputs

    def run(self, _outputs, feed):
        assert "token_type_ids" in feed, "declared inputs must all be fed"
        rows = np.array(
            [
                [[3.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 4.0], [9.0, 9.0], [9.0, 9.0]],
                [[3.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 4.0], [9.0, 9.0], [9.0, 9.0]],
            ]
        )
        return [rows]


def test_mean_pool_fallback_ignores_padding_and_normalises():
    embedder = OnnxEmbedder(repo="someone/custom-bert")
    embedder._session = _Session(  # bypass load: no download in tests
        inputs=("input_ids", "attention_mask", "token_type_ids")
    )
    embedder._tokenizer = _Tokenizer()
    embedder._pool = True
    embedder._output_index = 0

    vectors = embedder.embed(["a", "b"])
    assert len(vectors) == 2
    assert embedder.dim == 2
    for vec in vectors:
        assert abs(sum(v * v for v in vec) - 1.0) < 1e-6
    # mean of [3,0]x2 and [0,4]x2 = [1.5, 2.0] -> normalised
    norm = (1.5**2 + 2.0**2) ** 0.5
    assert abs(vectors[0][0] - 1.5 / norm) < 1e-6
    assert abs(vectors[0][1] - 2.0 / norm) < 1e-6


def test_dim_comes_from_the_model_shape():
    embedder = OnnxEmbedder(repo="Xenova/multilingual-e5-small")
    embedder._session = _Session(inputs=("input_ids", "attention_mask", "token_type_ids"))
    embedder._tokenizer = _Tokenizer()
    embedder._pool = True
    embedder._output_index = 0
    embedder._dim = 2
    assert embedder.dim == 2


def test_position_limit_respects_the_model():
    assert position_limit({"max_position_embeddings": 512}) == 512
    assert position_limit({"max_position_embeddings": 32768}) == 2048  # ours, capped
    assert position_limit({}) == 2048
    assert position_limit({"max_position_embeddings": "nonsense"}) == 2048
    assert position_limit({"max_position_embeddings": 8}) == 64  # never degenerate
