from __future__ import annotations

import sys
import types

import pytest

from ragdesk import llm as llm_module
from ragdesk.llm import (
    LLMUnavailable,
    MlxLLM,
    OllamaLLM,
    OpenAICompatLLM,
    llm_status,
    ollama_has_model,
    resolve_llm,
)


@pytest.fixture()
def no_mlx(monkeypatch):
    monkeypatch.setattr(llm_module, "mlx_available", lambda: False)


def test_auto_reuses_ollama_when_model_present(monkeypatch):
    monkeypatch.setattr(llm_module, "ollama_has_model", lambda tag, host: True)
    backend = resolve_llm("auto", preset="light")
    assert isinstance(backend, OllamaLLM)
    assert backend.model == "qwen3.5:4b"
    assert backend.kind == "ollama"


def test_auto_falls_back_to_mlx_without_touching_ollama(monkeypatch):
    monkeypatch.setattr(llm_module, "ollama_has_model", lambda tag, host: False)
    monkeypatch.setattr(llm_module, "mlx_available", lambda: True)
    backend = resolve_llm(None, preset="quality")
    assert isinstance(backend, MlxLLM)
    assert backend.model == "mlx-community/Qwen3.5-9B-4bit"
    assert backend.host == "in-process"


def test_auto_explains_when_nothing_is_available(no_mlx, monkeypatch):
    monkeypatch.setattr(llm_module, "ollama_has_model", lambda tag, host: False)
    with pytest.raises(LLMUnavailable) as excinfo:
        resolve_llm("auto", preset="light")
    assert "ollama pull qwen3.5:4b" in str(excinfo.value)
    assert "ragdesk[mlx]" in str(excinfo.value)


def test_explicit_specs_parse(no_mlx):
    assert resolve_llm("ollama", preset="light").model == "qwen3.5:4b"
    assert resolve_llm("ollama:custom:tag", preset="light").model == "custom:tag"
    assert resolve_llm("mlx:some/repo", preset="light").model == "some/repo"
    with pytest.raises(LLMUnavailable):
        resolve_llm("cloud:gpt", preset="light")


def test_env_spec_wins_over_default(no_mlx, monkeypatch):
    monkeypatch.setenv("RAGDESK_LLM", "mlx:env/repo")
    assert resolve_llm(None, preset="light").model == "env/repo"


def test_ollama_tags_normalise_latest(monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "_get_json",
        lambda host, path, timeout=1.5: {
            "models": [{"name": "qwen3.5:4b"}, {"name": "glm:latest"}]
        },
    )
    assert ollama_has_model("qwen3.5:4b", "http://x")
    assert ollama_has_model("glm", "http://x")
    assert not ollama_has_model("missing", "http://x")


def test_ollama_unreachable_is_not_an_error(monkeypatch):
    def boom(host: str, path: str, timeout: float = 1.5) -> dict:
        raise OSError("connection refused")

    monkeypatch.setattr(llm_module, "_get_json", boom)
    assert ollama_has_model("qwen3.5:4b", "http://127.0.0.1:9") is False


def test_mlx_backend_maps_options_and_streams(monkeypatch):
    calls: dict = {}

    class FakeTokenizer:
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True, **kwargs
        ) -> str:
            calls.setdefault("template_kwargs", []).append(kwargs)
            if kwargs:
                raise TypeError("enable_thinking unsupported")
            return "PROMPT"

    def fake_generate(model, tokenizer, prompt, *, max_tokens, verbose, **kwargs) -> str:
        calls["generate"] = {"prompt": prompt, "max_tokens": max_tokens, **kwargs}
        return "an answer"

    def fake_stream_generate(model, tokenizer, *, prompt, max_tokens):
        calls["stream"] = {"prompt": prompt, "max_tokens": max_tokens}
        yield types.SimpleNamespace(text="an ")
        yield types.SimpleNamespace(text="answer")

    mlx = types.ModuleType("mlx_lm")
    mlx.load = lambda repo: ("MODEL", FakeTokenizer())
    mlx.generate = fake_generate
    mlx.stream_generate = fake_stream_generate
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx)

    backend = MlxLLM("fake/repo")
    assert backend.generate("hi", {"num_predict": 7, "temperature": 0.1}) == "an answer"
    assert calls["generate"]["prompt"] == "PROMPT"
    assert calls["generate"]["max_tokens"] == 7
    assert calls["generate"]["temp"] == 0.1
    assert calls["template_kwargs"] == [{"enable_thinking": False}, {}]
    assert "".join(backend.generate_stream("hi", {"num_predict": 9})) == "an answer"
    assert calls["stream"]["max_tokens"] == 9


def test_mlx_model_cached_scans_snapshots(tmp_path):
    from ragdesk.llm import mlx_model_cached

    repo = "mlx-community/Qwen3.5-4B-MLX-4bit"
    cache = tmp_path / "hub"
    folder = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc"
    folder.mkdir(parents=True)
    assert mlx_model_cached(repo, str(cache)) is False  # dir exists, no weights yet
    (folder / "model.safetensors").write_bytes(b"x")
    assert mlx_model_cached(repo, str(cache)) is True
    assert mlx_model_cached("other/repo", str(cache)) is False
    assert mlx_model_cached("", str(cache)) is False


def test_llm_status_reports_none_with_hint(no_mlx, monkeypatch):
    monkeypatch.setattr(llm_module, "ollama_has_model", lambda tag, host: False)
    status = llm_status("auto", preset="light")
    assert status["kind"] == "none"
    assert "ollama pull" in status["note"]


def test_openai_backend_generate_and_stream(monkeypatch):
    import json as json_module

    calls: dict = {}

    class FakeStream:
        def __init__(self, lines):
            self.lines = lines

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(self.lines)

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        calls["url"] = request.full_url
        calls["auth"] = request.headers.get("Authorization")
        body = json_module.loads(request.data)
        calls["body"] = body
        if body.get("stream"):
            return FakeStream(
                [
                    b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
                    b"data: [DONE]\n",
                ]
            )
        return FakeResponse(
            json_module.dumps({"choices": [{"message": {"content": "an answer"}}]}).encode()
        )

    monkeypatch.setattr("ragdesk.llm.urllib.request.urlopen", fake_urlopen)
    llm = OpenAICompatLLM("qwen2.5-7b-instruct", "http://127.0.0.1:1234/v1", "sk-test")
    assert llm.generate("hi", {"num_predict": 10}) == "an answer"
    assert calls["url"] == "http://127.0.0.1:1234/v1/chat/completions"
    assert calls["auth"] == "Bearer sk-test"
    assert calls["body"]["max_tokens"] == 10
    assert "".join(llm.generate_stream("hi", {})) == "Hello"


def test_openai_unreachable_raises_llm_unavailable(monkeypatch):
    def boom(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("ragdesk.llm.urllib.request.urlopen", boom)
    backend = OpenAICompatLLM("m", "http://127.0.0.1:9/v1")
    with pytest.raises(LLMUnavailable) as excinfo:
        backend.generate("hi", {})
    assert "127.0.0.1:9" in str(excinfo.value)


def test_preference_and_openai_config(monkeypatch, tmp_path):
    from ragdesk import credentials, settings

    monkeypatch.setenv("RAGDESK_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(llm_module, "ollama_has_model", lambda tag, host: True)
    settings.save(
        {
            "openai_host": "http://127.0.0.1:1234/v1",
            "openai_model": "local-model",
            "llm_preference": "openai",
        }
    )
    credentials.set_provider("openai", {"api_key": "sk-test"})

    backend = resolve_llm(None, preset="light")  # preference beats the ladder order
    assert isinstance(backend, OpenAICompatLLM)
    assert backend.model == "local-model"
    assert backend.api_key == "sk-test"

    settings.save({"llm_preference": ""})  # auto: Ollama with the model wins again
    assert isinstance(resolve_llm(None, preset="light"), OllamaLLM)

    explicit = resolve_llm("openai:gpt-x", preset="light")
    assert explicit.model == "gpt-x"


def test_openai_explicit_without_config_explains(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGDESK_CONFIG_DIR", str(tmp_path / "config2"))
    with pytest.raises(LLMUnavailable) as excinfo:
        resolve_llm("openai:whatever", preset="light")
    assert "Settings" in str(excinfo.value)


def test_llm_status_reports_openai(monkeypatch, tmp_path):
    from ragdesk import settings

    monkeypatch.setenv("RAGDESK_CONFIG_DIR", str(tmp_path / "config3"))
    settings.save(
        {
            "openai_host": "http://127.0.0.1:1234/v1",
            "openai_model": "local-model",
            "llm_preference": "openai",
        }
    )
    status = llm_status(None, preset="light")
    assert status["kind"] == "openai"
    assert "1234" in status["note"]


def test_is_local_host():
    from ragdesk.llm import is_local_host

    assert is_local_host("")  # not configured: nothing leaves
    assert is_local_host("http://127.0.0.1:1234/v1")
    assert is_local_host("http://localhost:11434")
    assert is_local_host("http://[::1]:8080/v1")
    assert is_local_host("127.0.0.1:11434")
    assert not is_local_host("https://api.openai.com/v1")
    assert not is_local_host("http://192.168.1.50:8000/v1")
    assert not is_local_host("http://gpu-box.lan:8000/v1")
