from typing import Optional

from marss2l.config import GEEConfig

import os

ee_initialized = False


def ee_initialize(project: Optional[str] = None):
    global ee_initialized
    if ee_initialized:
        return

    import ee

    # https://developers.google.com/earth-engine/guides/service_account#use-a-service-account-with-a-private-key
    cfg = GEEConfig.from_env()
    # Explicit project argument wins, then EARTHENGINE_PROJECT env var; None is fine.
    project = project or cfg.project or None
    if not cfg.is_configured:
        ee.Authenticate()
        ee.Initialize(project=project)
    else:
        print("Using service account for EE")
        if os.path.isfile(cfg.service_account_key):
            credentials = ee.ServiceAccountCredentials(
                email=None, key_file=cfg.service_account_key
            )
        else:
            credentials = ee.ServiceAccountCredentials(
                email=None, key_data=cfg.service_account_key
            )
        ee.Initialize(credentials)

    ee_initialized = True
