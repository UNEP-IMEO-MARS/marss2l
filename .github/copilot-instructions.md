# GitHub Copilot instructions — marss2l

`marss2l` is a **public** package (PyPI + public GitHub). Apply these rules to
every code completion, chat answer, and pull-request review.

## 🔒 Never expose internal information

Do not introduce, and flag in review, any reference to the project's internal /
non-public infrastructure:

- **Private repositories** — do not name or link the organization's non-public
  repos. Only this package and its **public** dependencies (e.g. `marshsi`,
  `georeader`) may be referenced by name.
- **Internal URLs** — SharePoint / OneDrive (`*.sharepoint.com`), internal wikis
  or drives, the internal git server, internal portals / dashboards,
  Keycloak / SSO, or internal API endpoints.
- **Azure resource identifiers** — storage account names,
  `*.blob.core.windows.net` / `*.dfs.core.windows.net`, container names, SAS
  tokens, Key Vault names, DB hosts (`*.postgres.database.azure.com`), database
  or schema names, or connection strings. Use env vars locally and
  `${{ secrets.* }}` in CI.
- **Credentials** — API keys, tokens, passwords, service-principal IDs. Use a
  git-ignored `.env` with a mock `.env.sample`.
- **Absolute local paths** — home directories, usernames, or Azure ML
  cluster / compute paths. Prefer relative paths or environment variables.

## Reviewing pull requests
Treat any newly added internal URL, storage account, DB host / name,
private-repo reference, credential, or absolute home path as a **blocking**
issue and request changes.

## Enforcement
Pre-commit hooks (`gitleaks` + the local `mars-internal-refs` scan) enforce
these rules; never suggest bypassing them with `--no-verify`.

## Tests

- A change is done only when `make test` (unit tests) and `make test-integration`
  (integration-marked pytest plus every notebook in the `Makefile` list, via
  `nbmake`) pass.
- The notebooks read the published Hugging Face data through
  `marss2l.huggingface.HfCachedFileSystem`. Before uploading changed CSVs, check
  them by running the notebooks with `HfCachedFileSystem(local_dir=<new files>)`.
