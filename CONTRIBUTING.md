# Contributing to marss2l

Thanks for contributing! `marss2l` is a **public** package, so please keep all
internal / non-public information out of commits, code, notebooks, and PRs.

## Set up the leak-prevention hooks (one time)

After creating the dev environment (`make condaenv`), install the pre-commit
hooks so they run automatically on every commit:

```bash
pre-commit install
```

Run them on demand across the whole repo with:

```bash
make pre-commit          # == pre-commit run --all-files
```

The hooks are:

- **gitleaks** — detects committed secrets (keys, tokens, SAS, connection
  strings). Mock secrets in `.env.sample` are allow-listed in `.gitleaks.toml`.
- **mars-internal-refs** (`scripts/check_internal_refs.sh`) — blocks internal
  MARS references: Azure storage / DB / Key Vault identifiers, SAS tokens,
  SharePoint / internal URLs, Azure ML paths, and absolute home paths.

Maintainers can additionally block real org-specific literals (private-repo
names, internal host prefixes, usernames) by copying
`scripts/.internal-blocklist.template` to `scripts/.internal-blocklist` (which is
git-ignored) and filling it in.

## Do not commit

Private-repo names, internal URLs (SharePoint, internal git server, portals),
Azure storage accounts / DB hosts / DB names, SAS tokens or other credentials,
and absolute local paths (home dirs, usernames, Azure ML compute paths). Read
secrets from a git-ignored `.env` locally and from CI secrets in workflows. See
`CLAUDE.md` for the full policy.
