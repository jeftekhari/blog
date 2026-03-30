---
title: "Pretext Feels Like a Missing Layout Primitive"
date: 2026-03-30
slug: pretext-is-a-layout-primitive
description: "I wanted to see if I could keep writing blog posts in plain GitHub markdown while still dropping in a layout-aware demo. Pretext made that surprisingly straightforward."
tags: [engineering, typography, ui, javascript, blogging, pretext]
---

# Pretext Feels Like a Missing Layout Primitive

Every once in a while a library shows up and makes a weird part of the platform feel embarrassingly underpowered.

That was my reaction to [Pretext](https://github.com/chenglou/pretext).

I kept seeing slick little demos and examples floating around from folks like Alyx, Vlad, Sirokos, and birdabo, and the thing that grabbed me was not just "wow, neat text rendering."

It was:

**oh. this turns paragraph layout into data.**

That’s the part that feels important.

The browser is great at painting text.
It is much less great at letting you *work with* multiline layout as a first-class thing.

Usually the moment you want to know:
- how tall this paragraph will be
- how many lines it will wrap into
- what the actual line breaks are
- what the narrowest "nice" width is for a message bubble or pullquote

...you end up poking the DOM, measuring boxes, forcing layout, or building some cursed little cache that you already know will betray you later.

Pretext takes a different angle.

At the simple end, you can do this:

```ts
import { prepare, layout } from '@chenglou/pretext'

const prepared = prepare(text, '16px Inter')
const { height, lineCount } = layout(prepared, width, 24)
```

That alone is already useful.
You get paragraph height and line count without asking the DOM to reflow so you can inspect it after the fact.

But the part I really like is the next layer up:

```ts
import { prepareWithSegments, layoutWithLines } from '@chenglou/pretext'

const prepared = prepareWithSegments(text, '600 18px Inter')
const { lines } = layoutWithLines(prepared, width, 28)
```

Now you do not just know the paragraph dimensions.
You have the actual lines.

That means you can paint them however you want:
- DOM rows
- SVG
- canvas
- WebGL
- some weird editorial layout that would normally make CSS look at you like you asked it to solve taxes

That’s what I wanted to try here.

This blog still works the same way it did before:
- the post source of truth is still plain markdown in my [`jeftekhari/blog`](https://github.com/jeftekhari/blog) repo
- my site fetches that markdown remotely and renders it
- this article adds one tiny embed marker that the site hydrates on the client

So the post still lives in GitHub-flavored markdown.
It just happens to contain a small interactive layout demo powered by Pretext.

Here’s the demo:

<div class="pretext-demo" data-pretext-demo="compare">
  <script type="application/json">
    {
      "title": "Pretext in this actual blog post",
      "note": "Drag the width slider. The left panel is regular paragraph flow. The right panel is the same text manually laid out line-by-line from Pretext output.",
      "text": "Pretext turns paragraph layout into reusable data. That makes it great for message bubbles, pull quotes, weird editorial blocks, or any UI where you want the browser's line breaking behavior without asking the DOM to reflow on every pass. AGI 春天到了. بدأت الرحلة 🚀",
      "font": "600 18px Inter",
      "lineHeight": 28,
      "minWidth": 220,
      "maxWidth": 540,
      "initialWidth": 360,
      "step": 10
    }
  </script>
</div>

That little box is the whole trick.

The markdown file contains the content and the config.
The site sees the `data-pretext-demo` marker, loads a tiny browser module, runs Pretext, and manually renders the right-hand version line by line.

I like this approach because it keeps the boundaries clean:

- authoring stays in markdown
- the blog repo stays the source of truth
- Pretext only appears where I explicitly ask for it
- I don’t need to rebuild the whole blog system around a new content format

That feels like the right amount of ceremony for a personal site.

## Why This Is More Interesting Than "Fancy Text"

The flashy demos are fun, but I think the real value is boring, practical UI engineering:

- virtualization without guessing paragraph heights
- stable scroll anchoring when text arrives late
- message bubbles that shrink-wrap to the widest wrapped line
- editorial layouts where width changes from row to row
- preflight checks that tell you a label is going to wrap before it ships

Basically: anywhere text layout is part of your product logic instead of just a side effect of CSS.

That’s the thing I keep coming back to.
Pretext is not trying to replace the browser’s renderer.
It is exposing just enough of paragraph layout to let you build better stuff around it.

## The Version I Actually Wanted

I did *not* want a giant CMS rewrite for this.

I wanted:
- one npm package
- one small blog-side script
- one markdown post in GitHub
- one example that proves the idea is real

That bar matters.

A lot of otherwise-interesting frontend tools die the second they demand you reorganize your whole app around them.
This one didn’t.

I got to keep the existing flow and make one post a little more alive.

That’s usually my favorite kind of experiment:

small enough to ship,
weird enough to be memorable,
useful enough that I’ll probably steal the pattern again later.

If you want to try it yourself, the starting point is dead simple:

```sh
npm install @chenglou/pretext
```

Then start with `prepare()` and `layout()` if you only need paragraph height.
If you need the actual lines, jump straight to `prepareWithSegments()` and `layoutWithLines()`.

That second path is where the fun starts.
