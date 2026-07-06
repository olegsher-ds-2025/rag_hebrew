from config import Settings, settings


def test_defaults_present():
    assert settings.embedding_model == "intfloat/multilingual-e5-large"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 200
    assert settings.top_k == 10
    assert settings.rrf_k == 60
    assert settings.embedding_batch_size == 32
    assert settings.mavat_insecure_ssl is False


def test_env_overrides_field(monkeypatch):
    # Field maps to the uppercase env var (case-insensitive), no prefix.
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "8")
    fresh = Settings()
    assert fresh.top_k == 3
    assert fresh.embedding_batch_size == 8


def test_legacy_env_names_still_work(monkeypatch):
    monkeypatch.setenv("LLAMACPP_URL", "http://example:9/completion")
    monkeypatch.setenv("API_TOKEN", "secret")
    monkeypatch.setenv("MAVAT_INSECURE_SSL", "1")
    fresh = Settings()
    assert fresh.llamacpp_url == "http://example:9/completion"
    assert fresh.api_token == "secret"
    assert fresh.mavat_insecure_ssl is True   # "1" parses to True


def test_raw_dir_property(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/srv/docs")
    fresh = Settings()
    assert fresh.raw_dir == "/srv/docs/raw"
    assert fresh.processed_dir == "/srv/docs/processed"
