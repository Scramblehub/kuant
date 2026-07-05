# GitHub Rulesets

Branch protection rulesets for the `kuant` repo. Each JSON file here can be
imported at Settings → Rules → Rulesets on GitHub, or applied via the API.

## `main-protection.json`

Protects the `~DEFAULT_BRANCH` (currently `main`).

**Rules enforced:**

| Rule | What it does |
|---|---|
| `deletion` | Prevents accidental branch deletion. |
| `non_fast_forward` | Blocks force pushes to `main`. |
| `creation` | Blocks new branches from being created *as* `main` on the remote. |
| `pull_request` | Requires PR merge. Zero required approvals (solo owner). Merge methods limited to squash + rebase (no merge commits). |
| `required_status_checks` | All six CI jobs must pass: `pytest (3.10/3.11/3.12/3.13)`, `ruff`, `package build check`. Strict policy — PR head must be up to date with base. |
| `required_linear_history` | No merge commits on `main`. Fast-forward, rebase, or squash only. |

**Bypass:** repository role 5 (Maintain and above) can bypass. Adjust
`bypass_actors[].actor_id` in the JSON if a different bypass role is needed.

## How to apply

### Option 1 — UI (recommended for one-off setup)

1. Settings → Rules → Rulesets → New ruleset → Import a ruleset
2. Upload `main-protection.json`
3. Confirm

### Option 2 — API via `gh`

```bash
gh api \
  --method POST \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  /repos/Scramblehub/kuant/rulesets \
  --input .github/rulesets/main-protection.json
```

To update an existing ruleset, use PUT with the ruleset id:

```bash
# List rulesets to find the id
gh api /repos/Scramblehub/kuant/rulesets

# Update
gh api \
  --method PUT \
  --input .github/rulesets/main-protection.json \
  /repos/Scramblehub/kuant/rulesets/<ID>
```

### Option 3 — Terraform

The [GitHub provider](https://registry.terraform.io/providers/integrations/github/latest/docs/resources/repository_ruleset)
has a `github_repository_ruleset` resource that reads this JSON directly.

## Required status check names

The `required_status_checks` list must match the CI job names verbatim.
Current job names in [.github/workflows/ci.yml](../workflows/ci.yml):

- `pytest (Python 3.10)`, `pytest (Python 3.11)`, `pytest (Python 3.12)`, `pytest (Python 3.13)` — from the `test` job's matrix.
- `ruff` — the lint job.
- `package build check` — the `build` job.

If the CI job names change, update this JSON to match or new PRs will
report a stale check as required.
