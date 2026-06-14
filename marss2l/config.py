"""
config.py
=========

Environment-based credential configuration for ``marss2l``.

All external credentials are read **exclusively from environment variables** —
no secrets are dropped into the package directory. Each external service is
represented by a small dataclass that knows which environment variables it
needs (:attr:`required_env_vars`), how to build itself from the environment
(:meth:`from_env`), and whether it is currently configured
(:meth:`is_available`).

These same dataclasses are reused by the test-suite to declare the
dependencies of the notebook / integration tests (see ``tests/conftest.py``):
a test is skipped unless every config it depends on is configured.

Credentials governed here
-------------------------
* **Weights & Biases** — ``WANDB_API_KEY`` (key), ``WANDB_PROJECT`` (project name).
* **Google Earth Engine** — ``GEE_SERVICE_ACCOUNT_KEY`` (service-account JSON),
  ``GEE_PROJECT`` (project id).
* **Azure blob storage** — ``AZURE_STORAGE_ACCOUNT_NAME`` (account name),
  ``AZURE_STORAGE_CONTAINER_NAME`` (container), ``AZURE_STORAGE_SAS_TOKEN`` (SAS token).

The MARS-S2L Hugging Face dataset is public, so no token is required to read it.

See ``.env.sample`` at the repo root for an example with mock values.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------

ENV_WANDB_API_KEY = "WANDB_API_KEY"
ENV_WANDB_PROJECT = "WANDB_PROJECT"

ENV_GEE_SERVICE_ACCOUNT_KEY = "GEE_SERVICE_ACCOUNT_KEY"
ENV_GEE_PROJECT = "GEE_PROJECT"

ENV_AZURE_ACCOUNT_NAME = "AZURE_STORAGE_ACCOUNT_NAME"
ENV_AZURE_CONTAINER_NAME = "AZURE_STORAGE_CONTAINER_NAME"
ENV_AZURE_SAS_TOKEN = "AZURE_STORAGE_SAS_TOKEN"

# ---------------------------------------------------------------------------
# Defaults (non-secret values that are safe to keep in the source tree)
# ---------------------------------------------------------------------------

DEFAULT_WANDB_PROJECT = "s2l89-model"
DEFAULT_AZURE_ACCOUNT_NAME = "unepazeconomyadlsstorage"


@dataclass
class WandbConfig:
    """Weights & Biases credentials.

    The W&B client itself reads :envvar:`WANDB_API_KEY` automatically; this
    config simply centralises the project name and the availability check.
    """

    required_env_vars: ClassVar[Tuple[str, ...]] = (ENV_WANDB_API_KEY,)

    api_key: Optional[str] = None
    project: str = DEFAULT_WANDB_PROJECT

    @classmethod
    def from_env(cls) -> "WandbConfig":
        return cls(
            api_key=os.environ.get(ENV_WANDB_API_KEY),
            project=os.environ.get(ENV_WANDB_PROJECT, DEFAULT_WANDB_PROJECT),
        )

    @classmethod
    def is_available(cls) -> bool:
        return all(os.environ.get(v) for v in cls.required_env_vars)


@dataclass
class GEEConfig:
    """Google Earth Engine service-account credentials.

    ``service_account_key`` holds the *contents* of the service-account JSON
    (as a string), read from the environment — never a path to a file in the
    package.
    """

    required_env_vars: ClassVar[Tuple[str, ...]] = (
        ENV_GEE_SERVICE_ACCOUNT_KEY,
        ENV_GEE_PROJECT,
    )

    service_account_key: Optional[str] = None
    project: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GEEConfig":
        return cls(
            service_account_key=os.environ.get(ENV_GEE_SERVICE_ACCOUNT_KEY),
            project=os.environ.get(ENV_GEE_PROJECT),
        )

    @classmethod
    def is_available(cls) -> bool:
        return all(os.environ.get(v) for v in cls.required_env_vars)

    @property
    def is_configured(self) -> bool:
        return self.service_account_key is not None

    def service_account_dict(self) -> dict:
        """Parse the service-account JSON string into a dict."""
        if self.service_account_key is None:
            raise ValueError(
                f"{ENV_GEE_SERVICE_ACCOUNT_KEY} is not set; cannot parse the "
                "GEE service-account key."
            )
        return json.loads(self.service_account_key)


@dataclass
class AzureConfig:
    """Azure blob storage credentials.

    ``account_name`` falls back to the well-known public MARS account so that
    anonymous reads keep working when no SAS token is provided.
    """

    required_env_vars: ClassVar[Tuple[str, ...]] = (ENV_AZURE_SAS_TOKEN,)

    account_name: str = DEFAULT_AZURE_ACCOUNT_NAME
    container_name: Optional[str] = None
    sas_token: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AzureConfig":
        return cls(
            account_name=os.environ.get(ENV_AZURE_ACCOUNT_NAME, DEFAULT_AZURE_ACCOUNT_NAME),
            container_name=os.environ.get(ENV_AZURE_CONTAINER_NAME),
            sas_token=os.environ.get(ENV_AZURE_SAS_TOKEN),
        )

    @classmethod
    def is_available(cls) -> bool:
        return all(os.environ.get(v) for v in cls.required_env_vars)

    @property
    def is_configured(self) -> bool:
        return self.sas_token is not None
