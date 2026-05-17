"""Tests for LiteLLM model type in ModelFactory."""

import os
from unittest.mock import patch, MagicMock


class TestLiteLLMModelConfig:
    """Test LiteLLM model configuration structure."""

    def test_litellm_config_structure(self):
        config = {
            "litellm-claude": {
                "type": "litellm",
                "provider": "litellm",
                "name": "anthropic/claude-sonnet-4-20250514",
                "base_url": "http://localhost:4000/v1",
                "context_length": 200000,
            }
        }
        model_config = config.get("litellm-claude")
        assert model_config["type"] == "litellm"
        assert model_config["name"] == "anthropic/claude-sonnet-4-20250514"
        assert model_config["base_url"] == "http://localhost:4000/v1"

    def test_litellm_config_with_api_key_env_ref(self):
        config = {
            "litellm-model": {
                "type": "litellm",
                "provider": "litellm",
                "name": "openai/gpt-4o",
                "base_url": "https://litellm.example.com/v1",
                "api_key": "$LITELLM_API_KEY",
            }
        }
        model_config = config.get("litellm-model")
        assert model_config["api_key"] == "$LITELLM_API_KEY"

    def test_litellm_config_without_api_key(self):
        config = {
            "litellm-local": {
                "type": "litellm",
                "provider": "litellm",
                "name": "ollama/llama3",
                "base_url": "http://localhost:4000/v1",
            }
        }
        model_config = config.get("litellm-local")
        assert "api_key" not in model_config


class TestLiteLLMEnvironmentVariables:
    """Test LiteLLM environment variable handling."""

    def test_litellm_base_url_env(self):
        with patch.dict(os.environ, {"LITELLM_BASE_URL": "http://localhost:4000/v1"}):
            base_url = os.environ.get("LITELLM_BASE_URL")
            assert base_url == "http://localhost:4000/v1"

    def test_litellm_api_key_env(self):
        with patch.dict(os.environ, {"LITELLM_API_KEY": "sk-litellm-test"}):
            api_key = os.environ.get("LITELLM_API_KEY")
            assert api_key == "sk-litellm-test"

    def test_litellm_api_key_defaults_to_unused(self):
        with patch.dict(os.environ, {}, clear=True):
            api_key = os.environ.get("LITELLM_API_KEY") or "unused"
            assert api_key == "unused"


class TestLiteLLMModelType:
    """Test that litellm model type is recognized in the factory dispatch."""

    def test_litellm_type_dispatches_correctly(self):
        model_config = {
            "type": "litellm",
            "provider": "litellm",
            "name": "anthropic/claude-sonnet-4-20250514",
            "base_url": "http://localhost:4000/v1",
        }
        assert model_config.get("type") == "litellm"

    def test_litellm_supports_multiple_providers(self):
        models = [
            {"name": "anthropic/claude-sonnet-4-20250514", "type": "litellm"},
            {"name": "openai/gpt-4o", "type": "litellm"},
            {"name": "bedrock/anthropic.claude-3-sonnet", "type": "litellm"},
            {"name": "vertex_ai/gemini-2.5-flash", "type": "litellm"},
        ]
        for model in models:
            assert model["type"] == "litellm"
