---
title: "The Fellowship of the Query: Understanding Mixture of Experts & Natural Language Search"
date: 2026-02-09
slug: fellowship-of-the-query
description: "A look into MoE architecture and NL search over structured data told through the lens of Lord of the Rings."
tags: [ai, moe, machine-learning, lotr, natural-language-search]
---

# The Fellowship of the Query: Understanding Mixture of Experts & Natural Language Search

> A technical deep-dive into MoE architecture and NL search over structured data told through the lens of Lord of the Rings.
>
> *Contributions welcome. Open a PR if something's wrong.*

---

## The Story

*I recently interviewed for a people-search platform. The interviewer asked me to explain MoE. While I was somewhat familiar with the term, I stumbled when trying to explain it. Here is my attempt to explain the lifecycle of a natural language query using MoE looks like for a people-search company with 870 million contact records across 330 million Americans, sourced from 6,000+ data sources. A user types: **"Find me all licensed plumbers named Rodriguez in Sacramento with no criminal record."** This is the journey of that query but make it Lord of the Rings.*

### The Shire: A User Types a Query

A user sits at a search bar. They dont want to check boxes, or select from dropdowns.vThey don't speak SQL. They don't know the schema. They just type plain English, the way Frodo lived in the Shire knowing nothing of the wider world. The query is raw, unstructured and naive but it carries something powerful inside it. **Intent.** Like the Ring leaving the Shire, this string of text is about to pass through many hands, each one extracting meaning from it.

### The Old Forest & Tom Bombadil: Input Sanitization

Before the query goes anywhere meaningful, it passes through the API gateway. Rate limiting, auth checks, input sanitization. Tom Bombadil is weird and nobody fully understands why he exists, but he catches problems early. SQL injection? Malformed Unicode? Bombadil handles it. The quest almost ended in the Barrow-downs, and your pipeline almost died to a [Bobby Tables](https://xkcd.com/327/) attack. Unglamorous but necessary.

### Bree & The Prancing Pony: The NL Query Service

The query arrives at the NL parsing microservice. This is Bree, the first place where the sheltered Shire-folk meet the wider world. Here the query meets its Aragorn: the **intent classifier**.

Aragorn looks at Frodo and immediately understands what he's dealing with, even though Frodo doesn't fully understand himself. The intent classifier examines the query and identifies:

- **Intent:** People search with occupation + name + location + criminal filter
- **Entities extracted (the hobbits unpacking their baggage):**
  - Name: `Gandalf the Gray`
  - Occupation: `Wizard/Troublemaker`
  - License: required (professional license lookup)
  - Location: `Bree`
  - Criminal record: `unlicensed fireworks`

Each entity is like a hobbit being identified for what they carry. Frodo has the Ring (the core search intent). Sam has the provisions (location context). Merry and Pippin have the secondary filters that'll matter later.

### Weathertop: Query Embedding

The parsed query needs to be converted into a form the system can actually use. This is Weathertop - a dangerous, transformative moment.

The query gets **embedded** into a high-dimensional vector: a compressed mathematical representation of its meaning. Like Frodo putting on the Ring and being pulled into the wraith-world, the query leaves the human-readable realm and enters **vector space**. It's the same query, but now it exists in a form that machines can compare, search, and match against billions of records.

The Nazgûl attack is the risk of this step "**bad embeddings**" (a stab from a Morgul blade) will poison everything downstream. If `"wizard"` gets embedded too close to `"zard"` instead of `"staff"` or `"pointy hat,"` your results are corrupted from this point forward. It's like when you download the wrong version of the pirated movie and suddenly you have to explain things to your kids you thought you'd have more time to prepare for.

### Rivendell: The MoE Router (Council of Elrond)

Now we arrive at the heart of it. Here is where we deviate a bit from the story for MoE explanation purposes.

The query, fully parsed and embedded, reaches the **Mixture of Experts layer**. This is the Council of Elrond. Every expert sits at the table:

| Expert | Specialty | LOTR Equivalent |
|--------|-----------|-----------------|
| Expert 1 | Name / identity graph resolution | Aragorn - tracks anyone, anywhere |
| Expert 2 | Occupation & professional licensing | Gimli -binary precision, you have the credentials or you don't |
| Expert 3 | Location-based search | Legolas - long-range vision, sees geographic distinctions others miss |
| Expert 4 | Criminal records | Gandalf - navigates gray areas (expunged? pending? misdemeanor vs felony?) |
| Expert 5 | Contact information | *Stays seated* |
| Expert 6 | Relationship mapping | *Stays seated* |
| Expert 7 | Property records | *Stays seated* |
| Expert 8 | Business/corporate records | *Stays seated* |

The **router** (Elrond) evaluates the query and decides: this quest requires Experts 1, 2, 3, and 4. The other four stay seated, they're loaded in memory (they showed up to the Council) but they don't join this Fellowship. 

> **Bonus:** The additional experts show up to the council, taking up precious VRAM with their mere existence. This is a potential drawback to using other model architectures such as Ensemble, but useful for extremely large and diverse datasets.

**That's sparse activation.** The full model has the combined knowledge of all 8 experts, but this query only burns compute on 4. This is why MoE models are fast despite being enormous.

### The Fellowship Departs: Parallel Query Execution

The activated experts fire simultaneously, each handling their specialty:

**Aragorn (Name Expert)** searches the identity graph for all Gandalf records. Handles variations (Gandalf, Gandelf, Grandolf). Fuzzy matching, phonetic matching, alias resolution. There might be 50,000 Gandalf records across 330 million Americans.

**Legolas (Location Expert)** filters to Bree. Not just city name but ZIP codes, county records, address history. Someone who lived in Bree three years ago but moved? Legolas has the long-range vision to see that distinction.

**Gimli (Occupation/License Expert)** hits the professional licensing databases. Wizard licenses are city-issued in Gondor (CSLB). Cross-references against 6,000 data sources. Gimli will find you.

**Gandalf (Criminal Records Expert)** queries criminal records. "Illegal fireworks" seems simple, but expunged records? Pending cases? Misdemeanors vs felonies? Gandalf navigates the gray areas with wisdom, not brute force. (I realize in this example he is searching for his own crimes, just be cool man.)

### Moria: The Database Layer

The Fellowship enters Moria. It is a deep, ancient infrastructure. This is the database. Billions of rows in the dark. The experts' queries have been translated into structured queries and fired against the data stores.

It's vast, it's old, it's full of things that have been there for decades (30+ years of public records, like the Dwarves' ancient halls).

**The Balrog is a full table scan on a billion-row table with no index.** You do NOT want to awaken that. You need proper indexing, partitioning, and vector search acceleration in order to cross the Bridge of Khazad-dûm without waking the thing that kills your query latency.

### Lothlórien: Results Aggregation & Reranking

DISCLAIMER: This part of the analogy needs work and I don't fully understand the re-ranking layer.

The Fellowship emerges from Moria and reaches Lothlórien. **Galadriel's Mirror is the reranking layer.** Each expert returned results and now they need to be merged, deduplicated, and ranked.

Galadriel shows Frodo possible futures; the reranker shows the system possible result orderings and picks the best one:

| Result | Match Quality | Score |
|--------|--------------|-------|
| Gandalf, licensed wizard, Bree, 1 criminal hit | Perfect match | 0.97 |
| Gandalf, wizard license expired 2019, Bree | Partial/stale license | 0.61 |
| Gondolf, wizard supply *business* owner, Bree, no personal license | Technically relevant, probably wrong intent | 0.45 |

Galadriel also gives **gifts** the results get enriched here. Contact info attached. Address verified. Last known activity date. Each result leaves Lothlórien more useful than when it arrived.

### Amon Hen: Response Formatting

The final overlook before delivery. The raw ranked results get formatted into what the user actually sees such as cards, summaries, confidence indicators. The query has been transformed from a naive English sentence into a structured, ranked, enriched response.

### The Breaking of the Fellowship: Response Delivery

The response splits into parallel delivery paths:

- **Frodo & Sam (core results):** Go straight to the user's browser. The frontend renders the result cards.
- **Aragorn, Legolas, Gimli (analytics):** Query metadata gets logged. What was searched, which experts fired, latency, result count. This feeds back into training data for improving the router.
- **Merry & Pippin (cache):** The query-result pair gets cached. Next time someone searches for plumbers named Rodriguez in Sacramento, the Ents (cache layer) already have the answer. The Ents are slow to act on their own, but once roused, they deliver instantly.
- **Boromir:** Dies


### Bonus lil fun metaphor

**The Doors of Durin Problem:** The data sits behind structured gates (SQL schemas, API endpoints). Users want to speak naturally and have it open. The entire NL search pipeline exists to translate between these two worlds.

Melllllllloc

## Contributing

Found something wrong? Have a better metaphor? Open a PR.

I'm not going to sit here and pretend like I understand the math behind all of this. I'm not even totally confident of the specifics yet (I wrote this while I was studying). This started as interview prep and turned into something worth sharing. If the metaphor helps one person sorta kinda get MoE, it was worth writing.

---

*"Even the smallest query can change the course of the future." — Galadriel, probably*
