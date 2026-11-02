#!/usr/bin/env bash
# Block internal MARS references from entering this PUBLIC repository.
#
# Scans the files passed as arguments (pre-commit passes the staged files; a
# `--all-files` run passes everything) and fails if any line looks like internal
# infrastructure: Azure storage/DB/Key-Vault, SAS tokens, connection strings,
# SharePoint/OneDrive or other internal UNEP URLs, Azure ML paths, or absolute
# local paths that leak a username. Prints
#   BLOCKED: possible <category>
# for each category hit (never the matched value) and exits 1; exits 0 if clean.
#
# Design notes:
#   - GENERIC, commit-safe patterns only. Do NOT hardcode real repo names, hosts,
#     storage accounts or DB names here — this file is committed to a public repo,
#     so that would itself leak them.
#   - Lines containing a <PLACEHOLDER> token, known-public hosts, and generic
#     CI/home paths are ignored so templates and docs never trip the scan.
#   - Real org-specific literals may be listed in an OPTIONAL, git-ignored
#     scripts/.internal-blocklist (one term per line, case-insensitive substring).
#     If that file is absent (CI, outside contributors) the sub-check is skipped
#     and the generic patterns still apply.
set -uo pipefail

[ "$#" -eq 0 ] && exit 0

HERE="$(cd "$(dirname "$0")" && pwd)"
BLOCKLIST="$HERE/.internal-blocklist"

# Candidate content: drop placeholder lines, known-public hosts, and generic
# CI/home paths so they never trip the scan.
CANDIDATES="$(cat "$@" 2>/dev/null \
  | grep -vE '<[A-Za-z0-9_]+>' \
  | grep -vE '(methanedata|www)\.unep\.org' \
  | grep -vE '/(home|Users)/(user|runner|runneradmin|vscode|CI)/')"

hit=0
flag () {  # $1 = category label, $2 = regex
  if printf '%s\n' "$CANDIDATES" | grep -Eiq "$2"; then
    echo "BLOCKED: possible $1"
    hit=1
  fi
}

# --- Azure storage / database / secrets (shape-based) ---
flag "Azure storage endpoint"        '\.(blob|dfs|file|queue|table)\.core\.windows\.net'
flag "Azure Postgres host"           '\.postgres\.database\.azure\.com'
flag "Azure Key Vault"               '\.vault\.azure\.net'
flag "Azure SAS token"               'sig=[A-Za-z0-9%+/]{16,}'
flag "Azure SAS token"               'sp=[a-z]+&.*sig='
flag "Azure storage account key"     'accountkey=[A-Za-z0-9+/=]{20,}'
flag "Postgres connection string"    '(postgres|postgresql)://[^[:space:]:@/]+:[^[:space:]@/]+@'
flag "database name/credential"      '(database_password|db_password|pgpassword|database_name|dbname)[[:space:]]*[:=][[:space:]]*"?[^<[:space:]"'"'"']{2,}'

# --- Internal collaboration / portals ---
flag "SharePoint/OneDrive URL"       '[a-z0-9-]+(-my)?\.sharepoint\.com'
flag "internal UNEP host"            'https?://[a-z0-9.-]*\.unep\.org'

# --- Azure ML / absolute local paths that leak infra or usernames ---
flag "Azure ML batch path"           '/mnt/batch/tasks/shared/LS_root'
flag "absolute home path"            '(/home/[a-z0-9._-]+/|/Users/[A-Za-z0-9._-]+/|C:\\\\Users\\\\)'

# --- Optional org-specific literal blocklist (git-ignored) ---
if [ -f "$BLOCKLIST" ]; then
  while IFS= read -r term; do
    case "$term" in ''|\#*) continue;; esac
    if printf '%s\n' "$CANDIDATES" | grep -Fiq "$term"; then
      echo "BLOCKED: possible internal identifier"
      hit=1
    fi
  done < "$BLOCKLIST"
fi

if [ "$hit" -eq 1 ]; then
  echo "  -> This is a PUBLIC repository. Remove the internal reference(s) above before committing."
fi
exit $hit
