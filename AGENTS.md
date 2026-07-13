# AGENTS.md — marss2l

Instructions for AI coding agents. The full guidance is in `CLAUDE.md`; the
security rules are restated here because some tools read only this file.

## 🔒 This is a PUBLIC repository — never expose internal information

`marss2l` is published to PyPI and public on GitHub. Do not introduce — and flag
in any PR review — references to internal / non-public infrastructure:

- **Private repositories**: do not name or link the organization's non-public
  repos. Only this package and its **public** dependencies (e.g. `marshsi`,
  `georeader`) may be named.
- **Internal URLs**: SharePoint / OneDrive (`*.sharepoint.com`), internal wikis /
  drives, the internal git server, internal portals, Keycloak / SSO, internal
  APIs.
- **Azure resource identifiers**: storage accounts, `*.blob.core.windows.net` /
  `*.dfs.core.windows.net`, containers, SAS tokens, Key Vault names, DB hosts
  (`*.postgres.database.azure.com`), DB / schema names, connection strings. Use
  env vars locally and `${{ secrets.* }}` in CI.
- **Credentials**: API keys, tokens, passwords, service-principal IDs — use a
  git-ignored `.env` + mock `.env.sample`.
- **Absolute local paths**: home dirs, usernames, Azure ML cluster / compute
  paths — prefer relative paths / env vars.

**Reviewing PRs:** treat any of the above as a blocking issue. **Enforcement:**
pre-commit hooks (`gitleaks` + `mars-internal-refs`) — never bypass with
`--no-verify`; run `make pre-commit`.
