# CLAUDE.md — marss2l

Guidance for Claude / AI agents working in this repository (also see `AGENTS.md`
and `.github/copilot-instructions.md`, which carry the same rules).

`marss2l` is a Sentinel-2 / Landsat ML plume-detection package. Environment,
tests and linting are documented in `README.md` and the `Makefile`.

## 🔒 This is a PUBLIC repository — never expose internal information

`marss2l` is published to PyPI and its source is public on GitHub. No commit,
code suggestion, comment, docstring, notebook, or PR review may introduce
references to the project's internal / non-public infrastructure. Never add:

- **Private repositories** — do not name or link the organization's non-public
  repositories. Only this package and its **public** dependencies (e.g.
  `marshsi`, `georeader`) may be referenced by name.
- **Internal URLs** — SharePoint / OneDrive (`*.sharepoint.com`), internal wikis
  or shared drives, the internal git server, internal portals/dashboards,
  Keycloak / SSO, or internal API endpoints.
- **Azure resource identifiers** — storage account names,
  `*.blob.core.windows.net` / `*.dfs.core.windows.net`, container names, SAS
  tokens, Key Vault names, database hosts (`*.postgres.database.azure.com`),
  database or schema names, or connection strings. Read these from environment
  variables locally and from `${{ secrets.* }}` in CI.
- **Credentials** — API keys, tokens, passwords, or service-principal IDs. Use a
  git-ignored `.env` with a mock `.env.sample`.
- **Absolute local paths** — home directories, usernames, or Azure ML
  cluster / compute paths. Prefer relative paths or environment variables.

### When reviewing a pull request
Flag and request changes on any of the above. Treat a newly added internal URL,
storage account, DB host / name, private-repo reference, credential, or absolute
home path as a blocking issue.

### Enforcement
Pre-commit hooks (`.pre-commit-config.yaml`: `gitleaks` + the local
`mars-internal-refs` scan) enforce these rules. Do not bypass them with
`git commit --no-verify`. Run them with `make pre-commit` (or
`pre-commit run --all-files`). Maintainers can list real internal literals to
block in a git-ignored `scripts/.internal-blocklist` (see the tracked
`.template`).
