---
title: "GRC Dashboard v2: Access Controls and Auto-Generated Policies"
date: 2026-04-15
slug: grc-dashboard-access-controls-policies
description: "The compliance scanner now sees real branch protection details and auto-ships generated policies with your PR. Here's what filled in two big gaps."
tags: [grc, security, compliance, automation, nist, devops, dashboard]
---

# GRC Dashboard v2: Access Controls and Auto-Generated Policies

> Last post I built a scanner that told me everything wrong with my personal site and walked through adding security headers to fix it. This time I filled in the biggest gaps: a dashboard to see compliance across repos, real branch protection detection, and policies that actually ship with your code.
>
> *This is part of an ongoing series where I work through governance, risk, and compliance fundamentals by automating them.*

---

## Quick recap

The scanner is a GitHub Action that runs on every push and PR. It reads your repo, optionally hits your live site, and produces a structured manifest of findings: data collection points, third-party services, dependency vulnerabilities, exposed secrets, security headers, TLS config, access controls, whether governance documents exist. From that manifest it generates 10 compliance reports and policy templates.

The first version got that loop working but had three problems I wanted to solve next:

1. Reports evaporated after each run. There was no place to *see* compliance across time or across repos.
2. Branch protection detection was almost useless - we could tell it was on or off, but not the details.
3. Generated policies went into a gitignored directory and were never deployed. The scanner was producing compliance documents and throwing them away.

This post is about fixing those three.

## Persistent view: the dashboard

Every repo POSTs its scan manifest to a central dashboard running on Cloudflare Workers with KV storage. One URL shows compliance across everything.

It's at [grc-dashboard.jdeftekhari.workers.dev](https://grc-dashboard.jdeftekhari.workers.dev).

<!-- IMAGE: screenshot of the dashboard homepage showing the retro theme, HP bars, and repo cards -->

Per-repo view:
- **Overview**: data collection, third-party services, security headers, TLS, dependency vulns, access controls, governance artifacts
- **NIST CSF**: 18 controls mapped, cross-referenced to SOC 2 and ISO 27001, highlighted gaps
- **Branches**: side-by-side compliance comparison so you can catch regressions before merging
- **Trends**: compliance score, NIST score, and vulnerability count over time

There's also a **"Check Production"** button that hits your live URL on demand and reports current header/HTTPS status. Useful when you want to confirm a fix is live without waiting for the next scheduled scan.

Retro video game theme. HP bars. `[OK]` and `[XX]`. If I'm staring at compliance data, it should at least look like a game.

## Access controls: from "enabled" to useful

**The gap.** The first version of the scanner could tell you whether a repo had branch protection on `main`. That's it. A boolean. It couldn't tell you *how many reviewers were required*, whether *signed commits were enforced*, or which specific rule types were active. Those details live behind a GitHub API endpoint that requires admin scope, and `GITHUB_TOKEN` doesn't have admin scope in a workflow.

The dashboard showed "Branch Protection: ENABLED" next to "Required Reviews: --" and "Signed Commits: --" which looked broken. Useful check, useless output.

**The fix.** GitHub has a newer **Rulesets API** (`GET /repos/:owner/:repo/rules/branches/:branch`). It was designed to answer "what rules apply to me?" for anyone on the repo, not just admins. Standard read scope works fine.

The response is an array of effective rule objects:

```json
[
  { "type": "deletion" },
  { "type": "non_fast_forward" },
  {
    "type": "pull_request",
    "parameters": { "required_approving_review_count": 1 }
  },
  { "type": "required_linear_history" }
]
```

From that the scanner pulls:
- **Required reviews** from the `pull_request` rule's `required_approving_review_count`
- **Signed commits** from the presence of a `required_signatures` rule type
- **Other active rules** by their `type` names

Rules can come from both an **org-level ruleset** and a **repo-level ruleset**. If both have review-count rules, we take the strictest because that's what actually gets enforced when you try to merge.

**What it looks like now.**

```
Branch Protection: ENABLED
  Required reviews: 1
  Signed commits: not required (advisory)
  Rules: deletion, non_fast_forward, pull_request, required_linear_history
```

If signed commits aren't enforced, the scanner surfaces it as an advisory finding - not a failure, just a suggestion. Supply-chain integrity is one of those things most orgs don't enable until they have to.

## Governance artifacts: from "generated" to "shipped"

**The gap.** The scanner has always been able to generate privacy policies, terms of service, security.txt, vulnerability disclosure pages, and incident response plans. Each is populated from what the scanner detects - your actual data collection points, third-party processors, jurisdiction config, and deployed endpoints. So the generated privacy policy is specific to *your* site, not a generic template.

But the files were getting written into `.grc/` on the runner's filesystem, which is gitignored. When the GitHub Action finished, the runner evaporated and the files went with it. Meanwhile the artifact check looked at common repo paths, couldn't find any policies there, and reported them as "missing." The scanner was producing compliance documents and then deleting them.

**The fix.** Add one field to your `.grc/config.yml`:

```yaml
output_dir: docs/policies   # optional, this is the default
```

On a PR, the scanner now writes policies into your actual repo at `output_dir`:

```
docs/policies/
  privacy-policy.md
  terms-of-service.md
  vulnerability-disclosure.md
  incident-response-plan.md
.well-known/
  security.txt   # RFC 9116 requires this exact path
```

The action then stages those files, diffs against what's in the repo, and if anything changed, commits the update to your PR branch as `grc-bot`. You review the policies alongside your code changes in the same PR, merge everything together, and the policies ship with your deploy.

Consuming repos need `contents: write` in their workflow for this to work. The README documents it. If permission is missing, the action logs a warning and continues - the scan still runs, the dashboard still updates, the auto-commit just doesn't happen.

The policies are **idempotent by design**: no timestamps or commit hashes in the content, security.txt's `Expires` pinned to a calendar year. Three consecutive scans produce byte-identical output. Git sees no diff, the action doesn't commit, no noise in your PR.

## Verifying policies are actually served

**The gap.** Files in your repo don't mean URLs on your site. A privacy policy sitting at `docs/policies/privacy-policy.md` is only useful if visitors can reach it at something like `yoursite.com/privacy-policy`.

Different frameworks serve files differently. Express routes. Next.js pages. Hugo permalinks. Static HTML. Some sites use `/legal/privacy-policy`. We can't assume a universal URL pattern.

**The fix.** Let users declare their URLs:

```yaml
policy_urls:
  privacy_policy: /privacy-policy
  terms_of_service: /legal/terms
  vulnerability_disclosure: https://vdp.example.com/
  security_txt: /.well-known/security.txt
```

The "Check Production" button reads `policy_urls` from the manifest, fetches each one, and reports three states per policy:

- **served** - URL returned 2xx
- **unreachable** - URL configured but failed
- **not-configured** - URL not set (not a failure, just not checked)

Fully opt-in. No defaults. If you don't configure `policy_urls`, that section just doesn't get checked.

<!-- IMAGE: screenshot of dashboard showing a repo with generated policies and Check Production results -->

## Separating static scans from live checks

Scans break down naturally into two kinds:
- **Static**: read the code, check dependencies, look for secrets, verify files exist. Works anytime.
- **Live**: hit the deployed URL, check security headers, verify HTTPS enforcement, confirm policies are served. Only works after a deploy.

Tangling these in one run created a race with your deploy pipeline - sometimes the scan hit the old version, sometimes the new, depending on timing.

Now the split is clean. Static scans run in the GitHub Action on every push and PR, no coupling to deploy state. Live checks happen on-demand via the dashboard's "Check Production" button, which you click when you know a deploy is complete. The scanner stores your site URL during the action POST so the Worker knows where to check later.

## Open-source ready

The dashboard is designed to be forked. No hardcoded org names or personal KV IDs in the repo. Setup for a new org:

1. Fork the repo
2. `npx wrangler login && npx wrangler kv namespace create GRC_KV`
3. Add your KV ID and Cloudflare API token as repo secrets
4. `npx wrangler deploy`
5. Add a 15-line workflow to each repo you want scanned

No database to run, no server to host, no Docker. Just a Cloudflare Worker, KV storage, and GitHub Actions.

## What a full scan looks like now

My personal site scanned today:

```
shipstuff/joeeftekhari.com (main)
  Compliance: 93%   NIST CSF: 89%

  Security headers: 6/6
  Branch protection: ENABLED
    Required reviews: 1
    Signed commits: not required
  Dependencies: 4 high/critical
  Secrets detected: 0

  Artifacts:
    Privacy Policy:         generated
    Terms of Service:       generated
    security.txt:           present
    Vulnerability Disclosure: present
    Incident Response Plan:   present
```

<!-- IMAGE: screenshot of the dashboard with joeeftekhari.com expanded, showing the branch dropdown and populated compliance data -->

Each piece comes from a different source. Compliance score is an aggregate. NIST percentage comes from 18 mapped controls. Security headers come from the live check. Branch protection details come from the Rulesets API. Artifacts are file-existence checks. The dashboard stitches them together.

I still haven't read all of the NIST CSF 2.0 spec, and the 18 controls I've mapped are a slice of the ~100 total. The scanner can tell me whether something is enabled. It can't yet tell me whether my interpretation of the framework is correct. There's more learning to do.

---

*Next up: adding the AI compliance layer. EU AI Act detection and risk tiering, a new AI compliance tab on the dashboard, and auto-generated model cards and FRIAs for anything that needs them.*
