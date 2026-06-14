from marss2l.config import GEEConfig

ee_initialized = False


def ee_initialize():
    global ee_initialized
    if ee_initialized:
        return

    import ee

    # https://developers.google.com/earth-engine/guides/service_account#use-a-service-account-with-a-private-key
    cfg = GEEConfig.from_env()
    if not cfg.is_configured:
        ee.Authenticate()
        ee.Initialize(project=cfg.project)
    else:
        print("Using service account for EE")
        service_account = cfg.service_account_dict()["client_email"]
        credentials = ee.ServiceAccountCredentials(
            service_account, key_data=cfg.service_account_key
        )
        ee.Initialize(credentials, project=cfg.project)

    ee_initialized = True
