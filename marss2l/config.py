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
* **Google Earth Engine** — ``EARTHENGINE_SERVICE_ACCOUNT_KEY`` (service-account
  JSON). ``EARTHENGINE_PROJECT`` (project id) is optional.
* **Azure blob storage** — ``AZURE_STORAGE_ACCOUNT`` (account name),
  ``CONTAINER_NAME`` (container), ``SAS_TOKEN`` (SAS token).

The MARS-S2L Hugging Face dataset is public, so no token is required to read it.

On import this module loads a ``.env`` file from the current working directory
(if present) into ``os.environ`` via :func:`load_dotenv`. Real environment
variables take precedence over the file. See ``.env.sample`` at the repo root
for an example with mock values.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


def load_dotenv(path: str = ".env") -> None:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into ``os.environ``.

    Real environment variables take precedence: a key already present in the
    environment is never overwritten (so the order of preference is env vars
    first, then the ``.env`` file in the current working directory if it
    exists). Blank lines and ``#`` comments are ignored. Only the first ``=``
    splits a line, so values may contain ``=`` (e.g. SAS tokens); optional
    surrounding single/double quotes are stripped.
    """
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, sep, value = line.partition("=")
                if not sep:
                    continue
                key = key.strip()
                if not key:
                    continue
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                os.environ.setdefault(key, value)
    except OSError:
        pass


# Load a .env from the current working directory on import so that the config
# classes below (and anything reading os.environ) see those values.
load_dotenv()

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------

ENV_WANDB_API_KEY = "WANDB_API_KEY"
ENV_WANDB_PROJECT = "WANDB_PROJECT"

# Earth Engine / Azure variable names follow the georeader convention.
ENV_EARTHENGINE_SERVICE_ACCOUNT_KEY = "EARTHENGINE_SERVICE_ACCOUNT_KEY"
ENV_EARTHENGINE_PROJECT = "EARTHENGINE_PROJECT"

ENV_AZURE_ACCOUNT = "AZURE_STORAGE_ACCOUNT"
ENV_AZURE_CONTAINER = "CONTAINER_NAME"
ENV_AZURE_SAS_TOKEN = "SAS_TOKEN"

# ---------------------------------------------------------------------------
# Defaults (non-secret values that are safe to keep in the source tree)
# ---------------------------------------------------------------------------

DEFAULT_WANDB_PROJECT = "s2l89-model"


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
    package. ``project`` is optional: if unset, Earth Engine is initialised with
    ``project=None``.
    """

    required_env_vars: ClassVar[Tuple[str, ...]] = (ENV_EARTHENGINE_SERVICE_ACCOUNT_KEY,)

    service_account_key: Optional[str] = None
    project: Optional[str] = None

    @classmethod
    def from_env(cls) -> "GEEConfig":
        return cls(
            service_account_key=os.environ.get(ENV_EARTHENGINE_SERVICE_ACCOUNT_KEY),
            project=os.environ.get(ENV_EARTHENGINE_PROJECT),
        )

    @classmethod
    def is_available(cls) -> bool:
        return all(os.environ.get(v) for v in cls.required_env_vars)

    @property
    def is_configured(self) -> bool:
        # An empty string (e.g. an unset GitHub Actions secret) counts as unset.
        # The project is optional, so it is not required here.
        return bool(self.service_account_key)

    def service_account_dict(self) -> dict:
        """Parse the service-account JSON string into a dict."""
        if not self.service_account_key:
            raise ValueError(
                f"{ENV_EARTHENGINE_SERVICE_ACCOUNT_KEY} is not set; cannot parse the "
                "GEE service-account key."
            )
        return json.loads(self.service_account_key)


@dataclass
class AzureConfig:
    """Azure blob storage credentials.

    Everything is read from the environment — there is no hardcoded account or
    container, and no anonymous fallback. Public data is read from Hugging Face;
    Azure access always requires an account name and a SAS token.
    """

    required_env_vars: ClassVar[Tuple[str, ...]] = (
        ENV_AZURE_ACCOUNT,
        ENV_AZURE_SAS_TOKEN,
    )

    account_name: Optional[str] = None
    container_name: Optional[str] = None
    sas_token: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AzureConfig":
        return cls(
            account_name=os.environ.get(ENV_AZURE_ACCOUNT),
            container_name=os.environ.get(ENV_AZURE_CONTAINER),
            sas_token=os.environ.get(ENV_AZURE_SAS_TOKEN),
        )

    @classmethod
    def is_available(cls) -> bool:
        return all(os.environ.get(v) for v in cls.required_env_vars)

    @property
    def is_configured(self) -> bool:
        return bool(self.account_name) and bool(self.sas_token)
