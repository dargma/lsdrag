"""config.reader_config() — Reader 프로바이더 선택(openai|anthropic/claude) + override 검증."""
from src.config import Config


def _cfg(reader: dict):
    return Config({"reader": reader, "engine": {"root": "."}}, base_dir=".")


def test_default_openai():
    rc = _cfg({}).reader_config()
    assert rc["provider"] == "openai"
    assert rc["api_key_env"] == "OPENAI_API_KEY"
    assert "openai.com" in rc["endpoint"] and "gpt-4.1" in rc["model"]


def test_anthropic_preset():
    rc = _cfg({"provider": "anthropic"}).reader_config()
    assert rc["api_key_env"] == "ANTHROPIC_API_KEY"
    assert "anthropic.com" in rc["endpoint"] and rc["model"].startswith("claude")


def test_claude_alias():
    assert _cfg({"provider": "claude"}).reader_config()["api_key_env"] == "ANTHROPIC_API_KEY"


def test_explicit_override_wins():
    rc = _cfg({"provider": "anthropic", "model": "claude-haiku-4-5",
               "api_key_env": "MY_KEY"}).reader_config()
    assert rc["model"] == "claude-haiku-4-5" and rc["api_key_env"] == "MY_KEY"


def test_unknown_provider_falls_back_openai():
    rc = _cfg({"provider": "bogus"}).reader_config()
    assert rc["api_key_env"] == "OPENAI_API_KEY"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("[config] reader provider 선택 통과")
