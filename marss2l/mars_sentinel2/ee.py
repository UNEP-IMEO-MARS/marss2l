import os

ee_initialized = False

account_key_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/GCP_token.json"


def ee_initialize():
    global ee_initialized
    if ee_initialized:
        return

    import ee

    # https://developers.google.com/earth-engine/guides/service_account#use-a-service-account-with-a-private-key
    path_to_credentials = account_key_file
    if not os.path.exists(path_to_credentials):
        ee.Authenticate()
        ee.Initialize()
    else:
        # read "account" field from json
        import json

        print("Using service account for EE")
        with open(path_to_credentials, "r") as f:
            credentials_json = json.load(f)
            service_account = credentials_json["client_email"]

        credentials = ee.ServiceAccountCredentials(service_account, path_to_credentials)
        ee.Initialize(credentials)

    ee_initialized = True
