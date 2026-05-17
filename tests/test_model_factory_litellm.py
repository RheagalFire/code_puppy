"""Tests for LiteLLM model type in ModelFactory."""

import os
from unittest.mock import patch, MagicMock

import pytest


class TestLiteLLMFactoryDispatch:
    """Test that ModelFactory.get_model dispatches litellm type correctly."""

    def test_litellm_creates_model_with_base_url(self):
        from code_puppy.model_factory import ModelFactory

        config = {
            "litellm-claude": {
                "type": "litellm",
                "provider": "litellm",
                "name": "anthropic/claude-sonnet-4-20250514",
                "base_url": "http://localhost:4000/v1",
                "context_length": 200000,
            }
        }
        model = ModelFactory.get_model("litellm-claude", config)
        assert model is not None
        assert model.model_name == "anthropic/claude-sonnet-4-20250514"

    def test_litellm_uses_env_base_url(self):
        from code_puppy.model_factory import ModelFactory

        config = {
            "litellm-model": {
                "type": "litellm",
                "provider": "litellm",
                "name": "openai/gpt-4o",
            }
        }
        with patch.dict(os.environ, {"LITELLM_BASE_URL": "http://proxy:4000/v1"}):
            model = ModelFactory.get_model("litellm-model", config)
            assert model is not None
            assert model.model_name == "openai/gpt-4o"

    def test_litellm_missing_base_url_returns_none(self):
        from code_puppy.model_factory import ModelFactory

        config = {
            "litellm-model": {
                "type": "litellm",
                "provider": "litellm",
                "name": "openai/gpt-4o",
            }
        }
        with patch.dict(os.environ, {}, clear=True):
            # Remove LITELLM_BASE_URL if it exists
            os.environ.pop("LITELLM_BASE_URL", None)
            model = ModelFactory.get_model("litellm-model", config)
            assert model is None

    def test_litellm_with_api_key_env_ref(self):
        from code_puppy.model_factory import ModelFactory

        config = {
            "litellm-hosted": {
                "type": "litellm",
                "provider": "litellm",
                "name": "anthropic/claude-sonnet-4-20250514",
                "base_url": "https://litellm.example.com/v1",
                "api_key": "$MY_LITELLM_KEY",
            }
        }
        with patch.dict(os.environ, {"MY_LITELLM_KEY": "sk-hosted-key"}):
            model = ModelFactory.get_model("litellm-hosted", config)
            assert model is not None

    def test_litellm_with_raw_api_key(self):
        from code_puppy.model_factory import ModelFactory

        config = {
            "litellm-direct": {
                "type": "litellm",
                "provider": "litellm",
                "name": "openai/gpt-4o",
                "base_url": "http://localhost:4000/v1",
                "api_key": "sk-direct-key",
            }
        }
        model = ModelFactory.get_model("litellm-direct", config)
        assert model is not None

    def test_litellm_api_key_defaults_to_unused(self):
        from code_puppy.model_factory import ModelFactory

        config = {
            "litellm-local": {
                "type": "litellm",
                "provider": "litellm",
                "name": "ollama/llama3",
                "base_url": "http://localhost:4000/v1",
            }
        }
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LITELLM_API_KEY", None)
            model = ModelFactory.get_model("litellm-local", config)
            assert model is not None


class TestLiteLLMProviderIdentity:
    """Test provider identity resolution for litellm."""

    def test_litellm_in_type_overrides(self):
        from code_puppy.provider_identity import _TYPE_PROVIDER_OVERRIDES

        assert "litellm" in _TYPE_PROVIDER_OVERRIDES
        assert _TYPE_PROVIDER_OVERRIDES["litellm"] == "litellm"

    def test_resolve_identity_with_explicit_provider(self):
        from code_puppy.provider_identity import resolve_provider_identity

        config = {"type": "litellm", "provider": "litellm"}
        assert resolve_provider_identity("litellm-claude", config) == "litellm"

    def test_resolve_identity_from_type_override(self):
        from code_puppy.provider_identity import resolve_provider_identity

        config = {"type": "litellm"}
        assert resolve_provider_identity("litellm-claude", config) == "litellm"
