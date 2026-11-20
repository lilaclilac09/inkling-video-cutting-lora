import pytest

from eval.tinker_env import setup_tinker


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_key(monkeypatch, key):
    if key is None:
        monkeypatch.delenv("TINKER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TINKER_API_KEY", key)
    with pytest.raises(SystemExit, match="Set TINKER_API_KEY"):
        setup_tinker()


def test_setup_cleans_proxy_without_exposing_key(monkeypatch, capsys):
    monkeypatch.setenv("TINKER_API_KEY", "test-private-value")
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "localhost,[::1],example.com")
    setup_tinker()
    import os
    assert os.environ["TINKER_API_KEY"] == "test-private-value"
    assert os.environ["no_proxy"] == os.environ["NO_PROXY"] == "localhost,example.com"
    assert capsys.readouterr() == ("", "")
