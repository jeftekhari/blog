# HSFCU Interview — Jason Martinson Supplement

**Date:** 2026-04-28 | **Interviewer:** Jason Martinson, SVP Consumer & Mortgage Loan Division

---

## Who he is (1-minute read)

- **SVP, Consumer & Mortgage Loan Division Manager** at HSFCU (promoted June 2024)
- Oversees HELOC, centralized lending, consumer loan servicing — full processing/underwriting/funding/servicing chain
- **Economics, Pomona College** (elite liberal arts — thoughtful generalist)
- Career: Central Pacific Bank → HomeStreet → loanDepot → Academy Mortgage → HSFCU
- **⭐ At Central Pacific Bank, he personally led a full LOS replacement — design, testing, implementation, training**
- President (past/present) of Mortgage Bankers Association of Hawaii
- Board: Hawaii HomeOwnership Center Land Trust + Nanakuli Housing Corp (affordable housing mission)
- Appeared on Alec Hanson's "Modern Lending Playbook" podcast (2020)

**Translation:** Lending + operations executive, not an IT leader. Has bought/rebuilt loan tech from the business side. Values operational efficiency, clear communication, and delivery. Mission-driven about homeownership.

---

## Your strongest opening — the LOS parallel

> "I saw you led the LOS replacement at Central Pacific Bank — design through training. At SBA I was on the engineering side of the same motion for the federal government's largest commercial lending program. I led the database migration from SQLite to PostgreSQL with zero data loss, redesigned the schemas, built the validation suite, and trained downstream teams. We were probably fighting the same problems from opposite sides of the table — business people wanting the system to match how they actually work, engineers needing the business to clarify edge cases they assumed were obvious. That's the work I want to keep doing."

---

## ⚠️ Your honest gap — never personally serviced a loan

You built the systems; you weren't a loan officer. Handle it directly when it comes up:

> "I should be upfront — I've been on the engineering and systems side of lending for 5+ years, but I've never personally originated, underwritten, or serviced a loan. What I do have is deep systems-level understanding of the workflows: I can tell you exactly what data flows from LOS to core, what hits the GL, what triggers an escrow analysis, where the regulatory touchpoints sit. But if you ask me the human feel of working a delinquency queue at 7am on a Monday, I'd be making it up. I'd rather learn that from your team than pretend."

**Why this works:**
- Martinson values judgment over bravado
- Acknowledging the gap disarms the follow-up question before he asks it
- Pivot to what you DO know gives him a systems answer that's credible
- "I'd rather learn from your team" is respectful framing

---

## Likely loan-specific questions & suggested answers

### Q: "Walk me through the lifecycle of a consumer loan."
> "Application → prequalification → formal submission → underwriting (credit + DTI + collateral if secured) → approval and disclosures (TRID for mortgage, TILA for consumer) → closing and funding → boarding to servicing system → payment processing, escrow if applicable, delinquency management → payoff or charge-off. Each stage writes to the LOS and eventually boards to the core — that integration point is where a lot of my engineering attention went at SBA."

### Q: "What's the difference between origination and servicing?"
> "Origination is everything up to funding — application, underwriting, closing, disbursement. Servicing is everything after — payment collection, escrow, statements, delinquency, modifications, payoffs. Different systems, different teams, different regulatory regimes. At SBA our platform handled both and the seam between them was always the hardest integration to keep clean."

### Q: "What's TRID / RESPA / Reg Z / Reg X and why do they matter?"
> "TRID integrates RESPA and TILA disclosure requirements — Loan Estimate within 3 days of application, Closing Disclosure at least 3 days before closing. Reg Z is Truth in Lending — APR accuracy, right of rescission on refinances. Reg X is RESPA servicing rules — escrow, error resolution, loss mitigation. From a systems perspective they all mean the same thing: specific events trigger specific disclosures within specific windows, and missing a window is a reportable violation. I built the audit trails at SBA to survive exactly this kind of scrutiny."

### Q: "How do you handle payment processing and escrow?"
> "Honestly — I've designed the data models and integrations for payment processing but I haven't worked the exception queue day-to-day. From the systems side: payment hits, splits into principal/interest/escrow/late fees/unpaid fees per the waterfall in the note, GL entries fire, if escrow-backed the analysis runs annually or on trigger events (tax bill, insurance change). Where I'd lean on your team is the feel for when an exception is 'fix it quietly' versus 'escalate immediately.'"

### Q: "What's an LOS, and have you used [Encompass / Black Knight Empower / MeridianLink / Symitar PowerOn]?"
> "LOS is Loan Origination System — the workflow layer that tracks an application from intake through funding. I haven't worked inside Encompass or Empower directly, but I've done the integration work on SBA's system — which functionally performed the same role: workflow state, document management, underwriting logic, disclosures, funding. I'd expect a few weeks to be productive in Symitar PowerOn or whatever you're on; the concepts transfer, the syntax doesn't."

### Q: "How do delinquency buckets work?"
> "Standard buckets are 1-29, 30-59, 60-89, 90-119, 120+, with charge-off typically at 120 or 180 depending on loan type and policy. The bucket drives collections intensity, reporting (NCUA 5300 call report line items), and allowance for loan loss accounting. From a systems angle what I've paid attention to is: date math has to be bulletproof because the bucket is legally consequential, and the rule engine has to handle holidays, partial payments, and modification resets correctly."

### Q: "What's a HELOC vs. a standard mortgage in system terms?"
> "HELOC is revolving — draw period, then repayment period, variable rate usually, interest-only payments often allowed during the draw. First mortgage is typically fully amortizing, fixed or ARM, with escrow. System-wise: HELOC needs credit-line tracking, draw management, and a phase transition when the draw period ends. A first mortgage is a simpler amortization schedule but more escrow complexity. They share compliance frameworks but the data models diverge."

### Q: "What's the relationship between the LOS and the core?"
> "LOS handles origination workflow; core handles the loan after boarding — balances, payments, statements, GL, member-facing access. Integration is usually via nightly batch or real-time API — loan boards from LOS, payments and statements flow back from core, modifications can originate in either and need to reconcile. The seam is a classic place for data drift — I've spent real time on exactly this problem."

### Q: "If we gave you our loan systems on day one, what would you look for first?"
> "Three things, in order: (1) what integrations are the most fragile — where does data silently diverge between systems? (2) what reports do the business teams work around because the system can't produce them natively? and (3) what audit or regulatory processes are manual that shouldn't be? Those three questions usually surface 80% of the technical debt and 100% of the team's daily pain. I'd spend the first two weeks listening before proposing anything."

### Q: "What's one thing you'd change about how loan systems teams usually operate?"
> "The biggest gap I've seen is that loan systems teams often treat the business team as requirements-deliverers rather than partners. The engineers who ship the best systems sit in on servicing calls, watch a loan officer close a file, and understand what the 'quick fix' they're being asked for actually unblocks. At SBA the times I delivered my best work were after spending a day shadowing the team I was building for."

### Q: "How do you handle regulatory compliance and audit?"
> "Audit-ready by design, not bolted on. Every state change in a loan lifecycle has to produce an immutable record with timestamp, actor, before/after values. Access control has to be role-based and enforceable at the query level, not just the UI. And the compliance team should be able to self-serve their reports — if they have to ask engineering for every audit pull, something's broken. I built exactly this at SBA and open-sourced a GRC scanner that maps code-level findings to NIST CSF, SOC 2, and ISO 27001 — NCUA Part 748 is the same playbook."

### Q: "Why should we hire you for this role specifically?"
> "Three things. First, the technical stack match is direct — C#/.NET, SQL Server, PowerShell, the SBA environment is the same shape as Symitar's world with different labels. Second, I've lived through the exact system modernization you led at Central Pacific — same motion, different side of the table. Third, I'm local, I want this job, and I understand that the value of a loan systems role isn't the code — it's the judgment about which problems to solve and in what order. Three weeks into this role, I'd rather be known as the person who listened well than the person who shipped fast."

---

## The question to ask him at the end

> "You led the LOS replacement at Central Pacific Bank and revamped lending procedures here at HSFCU. What's the biggest gap you see right now between what the lending team needs from their systems and what the systems actually deliver?"

**Backup:** "What does a great year look like for your division, and where does IT fall short of supporting that today?"

---

## Quick-reference cheat sheet

| Acronym | Meaning |
|---------|---------|
| LOS | Loan Origination System |
| GL | General Ledger |
| DTI | Debt-to-Income ratio |
| LTV | Loan-to-Value |
| TRID | TILA-RESPA Integrated Disclosure |
| TILA | Truth in Lending Act (Reg Z) |
| RESPA | Real Estate Settlement Procedures Act (Reg X) |
| HMDA | Home Mortgage Disclosure Act |
| LAR | Loan Application Register (HMDA) |
| ARM | Adjustable Rate Mortgage |
| HELOC | Home Equity Line of Credit |
| Reg B | Equal Credit Opportunity Act |
| 5300 | NCUA Call Report |
| ACET | Automated Cybersecurity Evaluation Toolbox |
| Part 748 | NCUA info security rule |
| NCUA | National Credit Union Administration |
| Symitar Episys | HSFCU's core banking system |
| PowerOn | Symitar's scripting language |

---

## Do NOT

- Brag about self-hosted Llama / K8s / agentic pipelines unprompted — misreads the room
- Volunteer Oahu residence before it matters
- Assume he's evaluating you only for Loan Systems II — he's a division exec sizing you up for his team (Loan Systems II, BSA III, or Dept Manager)
- Overclaim loan servicing experience — honest gap framing beats fake confidence
