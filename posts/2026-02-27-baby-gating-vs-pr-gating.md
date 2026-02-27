# Baby Gating vs PR Gating

It’s Pokémon Day’s 30th anniversary, and somehow Pikachu has outlived half the frameworks I’ve shipped to prod.

I’m writing this while installing baby gates and outlet covers, which is either peak parenting or peak irony depending on your sleep score.

Somewhere between “why are there 900 outlets in this house” and “where did the tiny screwdriver go,” it clicked: this is exactly how we run PR gating in **Permitted**. Permitted is a public data normilzation and visualiation project we're building but that's not really the point of this.. 

A baby gate is not there because you think your kid is dumb.
It’s there because your kid is fast, creative, and has zero fear.
So… basically every engineer shipping code on a Friday night. Turns out both toddlers and engineers will immediately pathfind to the one unsafe option you forgot to guard.

Same deal with PR gating.

We don’t gate because devs are bad.
We gate because humans are tired, moving quickly, context-switching, and occasionally convinced a risky refactor is “just a tiny cleanup.” Outlet covers are just branch protection for tiny chaos agents.

Outlet covers are the best analogy for required checks.
Nobody gets excited about outlet covers. Nobody tweets about outlet covers.
But they quietly prevent one very bad day.
That’s exactly what required status checks do on ~~master~~ `main`.

In Permitted, we’ve been trying to keep this practical:
- run tests based on what changed (path-scoped), not “run every test known to man”
- layer checks so one failure doesn’t have to be production to be discovered
- keep the gates fast enough that people don’t try to route around them

Because if CI is slow and noisy, people stop trusting it.
And once trust is gone, your “gates” are just decorative baby furniture.

Another parenting parallel: you don’t baby-proof *everything*. There's no need to baby-proof a cieling fan... right?
You baby-proof the places your kid can actually reach and break themselves on today.

Software equivalent:
if a PR touches frontend copy, don’t make it wait on some giant backend integration marathon from 2019.
if a PR touches API contracts, yeah, now we do the heavier stuff.
Risk-based gating > ritual-based gating.

Also, this one’s important: gates should fail loudly and clearly.
At 2am, nobody wants to parse cryptic CI poetry.
Tell me what failed, why, and where. Preferably in one screen, not twelve tabs and a terminal.

The goal isn’t “more checks.”
The goal is “fewer surprises in prod.”

That’s the whole thing.

Baby gates don’t make your house perfect.
PR gates don’t make your code perfect.
They just dramatically lower the odds that your worst day starts with, “huh, that’s weird…”

Anyway, I’m off to install more safety gear while a tiny human attempts to speedrun self-destruction.


*Remember, if you don’t gate PRs, you’re just raw-dogging `main` and hoping vibes are enough.*
