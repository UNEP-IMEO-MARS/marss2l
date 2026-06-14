"""Tests for marss2l.config — env-based credential loading.

Mirrors the georeader credential-config test pattern: credentials are read
purely from environment variables, exercised here with ``patch.dict``.
"""

import json
import os
from unittest.mock import patch

from marss2l.config import (
    DEFAULT_AZURE_ACCOUNT_NAME,
    DEFAULT_WANDB_PROJECT,
    ENV_AZURE_ACCOUNT_NAME,
    ENV_AZURE_CONTAINER_NAME,
    ENV_AZURE_SAS_TOKEN,
    ENV_GEE_PROJECT,
    ENV_GEE_SERVICE_ACCOUNT_KEY,
    ENV_WANDB_API_KEY,
    ENV_WANDB_PROJECT,
    AzureConfig,
    GEEConfig,
    WandbConfig,
)


class TestWandbConfig:
    def test_from_env_reads_values(self):
        env = {ENV_WANDB_API_KEY: "key123", ENV_WANDB_PROJECT: "my-project"}
        with patch.dict(os.environ, env, clear=True):
            cfg = WandbConfig.from_env()
        assert cfg.api_key == "key123"
        assert cfg.project == "my-project"

    def test_project_defaults_when_unset(self):
        with patch.dict(os.environ, {ENV_WANDB_API_KEY: "key123"}, clear=True):
            cfg = WandbConfig.from_env()
        assert cfg.project == DEFAULT_WANDB_PROJECT

    def test_is_available(self):
        with patch.dict(os.environ, {ENV_WANDB_API_KEY: "key123"}, clear=True):
            assert WandbConfig.is_available() is True
        with patch.dict(os.environ, {}, clear=True):
            assert WandbConfig.is_available() is False


class TestGEEConfig:
    def test_from_env_reads_values(self):
        key = json.dumps({"client_email": "sa@example.com", "private_key": "x"})
        env = {ENV_GEE_SERVICE_ACCOUNT_KEY: key, ENV_GEE_PROJECT: "proj"}
        with patch.dict(os.environ, env, clear=True):
            cfg = GEEConfig.from_env()
        assert cfg.project == "proj"
        assert cfg.is_configured is True
        assert cfg.service_account_dict()["client_email"] == "sa@example.com"

    def test_not_configured_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = GEEConfig.from_env()
        assert cfg.is_configured is False
        assert GEEConfig.is_available() is False

    def test_is_available_requires_both(self):
        with patch.dict(os.environ, {ENV_GEE_SERVICE_ACCOUNT_KEY: "{}"}, clear=True):
            assert GEEConfig.is_available() is False
        env = {ENV_GEE_SERVICE_ACCOUNT_KEY: "{}", ENV_GEE_PROJECT: "proj"}
        with patch.dict(os.environ, env, clear=True):
            assert GEEConfig.is_available() is True


class TestAzureConfig:
    def test_from_env_reads_values(self):
        env = {
            ENV_AZURE_ACCOUNT_NAME: "myaccount",
            ENV_AZURE_CONTAINER_NAME: "mycontainer",
            ENV_AZURE_SAS_TOKEN: "sv=token",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = AzureConfig.from_env()
        assert cfg.account_name == "myaccount"
        assert cfg.container_name == "mycontainer"
        assert cfg.sas_token == "sv=token"
        assert cfg.is_configured is True

    def test_account_name_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = AzureConfig.from_env()
        assert cfg.account_name == DEFAULT_AZURE_ACCOUNT_NAME
        assert cfg.sas_token is None
        assert cfg.is_configured is False
        assert AzureConfig.is_available() is False

    def test_is_available_requires_sas_token(self):
        with patch.dict(os.environ, {ENV_AZURE_SAS_TOKEN: "sv=token"}, clear=True):
            assert AzureConfig.is_available() is True
